import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import json
from torch import nn
from dataclasses import dataclass
from typing import List, Dict
from transformers import (
    AutoTokenizer,
    ModernBertModel,            # 直接匯入官方 ModernBertModel
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    EvalPrediction
)
from torch.utils.data import Dataset
import numpy as np

# def compute_metrics(eval_pred):
#     logits, labels, losses = eval_pred
#     logits = torch.tensor(logits)
#     labels = torch.tensor(labels)
#     loss = nn.CrossEntropyLoss()(logits, labels)
#     return {"eval_loss": float(losses.mean())}

# ---------- compute_metrics ----------
from sklearn.metrics import f1_score
def my_compute_metrics(eval_pred):
    # 取出 logits, label_ids, losses
    logits, labels, losses = (
        torch.tensor(eval_pred.predictions),
        torch.tensor(eval_pred.label_ids),
        torch.tensor(eval_pred.losses),
    )

    preds = logits.argmax(dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    f1 = f1_score(labels_np, preds, average="macro")   # or "weighted"

    metrics_out = {
        "eval_loss": float(losses.mean()),
        "f1": f1,
    }
    print("DEBUG metrics:", metrics_out)               # ← 確認有被叫到
    return metrics_out

# def compute_metrics(eval_pred: EvalPrediction):
#     return {"loss": float(np.mean(eval_pred.losses))}

# ----------- Dataset -----------
class ContrastiveDataset(Dataset):
    def __init__(self, json_path: str, doc_folder: str):
        # 載入 JSON 資料 (已由 create_bm25_hard_negative_data.py 生成)
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.doc_folder = doc_folder
        print(f"🔹 對比式資料（{os.path.basename(json_path)}）共載入 {len(self.data)} 筆樣本")

    def __len__(self):
        return len(self.data)

    def load_text(self, doc_id: str) -> str:
        # 讀 text 檔（判決書內容），只取純文字
        path = os.path.join(self.doc_folder, f"{doc_id}.txt")
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "query_text": self.load_text(sample["query_id"]),
            "positive_text": self.load_text(sample["positive_id"]),
            "negative_texts": [self.load_text(nid) for nid in sample["negative_ids"]]
        }

# ----------- Collator -----------
@dataclass
class ContrastiveCollator:
    tokenizer: AutoTokenizer
    max_length: int = 8192

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # batch 裡面每個 item = {"query_text": str, "positive_text": str, "negative_texts": [str, ...]}
        q_texts = [item["query_text"] for item in batch]
        p_texts = [item["positive_text"] for item in batch]
        n_texts = [neg for item in batch for neg in item["negative_texts"]]

        # 呼叫 tokenizer 取得 input_ids, attention_mask
        anchor_enc = self.tokenizer(
            q_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        positive_enc = self.tokenizer(
            p_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        negative_enc = self.tokenizer(
            n_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "anchor_input": anchor_enc,
            "positive_input": positive_enc,
            "negative_input": negative_enc
        }

# ----------- Model with InfoNCE loss -----------
class ModernBERTContrastive(nn.Module):
    def __init__(self, model_name: str, device, temperature: float = 0.05):
        super().__init__()
        # 使用官方 ModernBertModel，並傳入官方文件建議的各項參數
        self.encoder = ModernBertModel.from_pretrained(
            model_name,
            device_map=device,                  # 指定 GPU 或 CPU
            #torch_dtype=torch.float16,          # 不要使用 FP16，參數權重照樣載入FP32，TrainingArguments(fp16=True) 照留，AMP 會自動把運算 cast 成 FP16
            attn_implementation="flash_attention_2",  # 啟用 Flash Attention 2
        )
        self.temperature = temperature
        self.encoder.config.use_cache = False
        self.encoder.enable_input_require_grads()   # 讓輸入 tensor 也進入 checkpoint 機制
        self.encoder.gradient_checkpointing_enable()  # 開啟梯度檢查點

    def encode(self, input_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        把 tokenizer 回傳的 dict(**batch) 丟給 ModernBertModel，並取 CLS token 的 hidden state
        input_batch 範例：{"input_ids": Tensor, "attention_mask": Tensor, ...}
        """
        output = self.encoder(**input_batch)
        # last_hidden_state shape = (batch_size, seq_len, hidden_size)
        # CLS token 在 index 0
        return output.last_hidden_state[:, 0, :]  # 回傳 shape = (batch_size, hidden_size)

    def forward(
        self,
        anchor_input: Dict[str, torch.Tensor],
        positive_input: Dict[str, torch.Tensor],
        negative_input: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # 1. 分別取得 anchor / positive / negative 向量
        anchor_vec = self.encode(anchor_input)
        positive_vec = self.encode(positive_input)
        negative_vec = self.encode(negative_input)

        bsz = anchor_vec.size(0)
        # 假設 negative_vec shape = (bsz * num_negatives, hidden_size)
        neg_count = negative_vec.size(0) // bsz
        negative_vec = negative_vec.view(bsz, neg_count, -1)  # (bsz, neg_count, hidden_size)

        # 2. 計算 cosine 相似度：anchor vs positive 以及 anchor vs each negative
        # anchor vs positive → (bsz,)
        pos_sim = torch.cosine_similarity(anchor_vec, positive_vec, dim=-1).unsqueeze(1)  # → (bsz, 1)
        # anchor vs negative → 比對維度 (bsz, 1, hidden) vs (bsz, neg_count, hidden) → (bsz, neg_count)
        neg_sim = torch.cosine_similarity(anchor_vec.unsqueeze(1), negative_vec, dim=-1)  # → (bsz, neg_count)

        # 3. 拼接 logits，第一欄放正樣本，後面欄放負樣本
        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature  # shape = (bsz, 1 + neg_count)
        labels = torch.zeros(bsz, dtype=torch.long).to(logits.device)     # 正樣本在 index 0

        loss = nn.CrossEntropyLoss()(logits, labels)
        return {
            "loss": loss,
            "labels": labels,          # 關鍵：給 Trainer “看到” labels
            "logits": logits           # logits 想省也可省
        }

# ----------- Main Training -----------  
def main():
    # 1. 檢查 CPU / GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ 使用 CPU")

    model_name = "answerdotai/ModernBERT-base"
    print("🔹 載入 tokenizer 與模型...")
    # tokenizer 仍然用 AutoTokenizer，即可支援 ModernBERT
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. 初始化自定義的 Contrastive Model
    model = ModernBERTContrastive(model_name, device)
    print("✅ Tokenizer 與 Model 初始化完成\n")

    # 3. 載入訓練 / 驗證資料集
    train_json_path = "./coliee_dataset/task1/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_train.json"
    valid_json_path = "./coliee_dataset/task1/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_valid.json"
    doc_folder = "./coliee_dataset/task1/processed"

    train_dataset = ContrastiveDataset(json_path=train_json_path, doc_folder=doc_folder)
    valid_dataset = ContrastiveDataset(json_path=valid_json_path, doc_folder=doc_folder)

    # 裁切訓練集與驗證集前 10 筆樣本，快速測試用
    train_dataset.data = train_dataset.data[:10]
    valid_dataset.data = valid_dataset.data[:10]
    print("測試模式：裁切前 10 筆樣本進行訓練")

    # 4. 設定 TrainingArguments，啟用 validation、早停與模型儲存
    args = TrainingArguments(
        output_dir="./modernBERT_contrastive",
        per_device_train_batch_size=1,
        #gradient_accumulation_steps=4,   # GPU 只需放 1 樣本；4 步再做一次 optimizer.step()。
        per_device_eval_batch_size=1,
        fp16=True,                                  # 使用半精度訓練
        learning_rate=5e-6,
        num_train_epochs=20,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",                 # 線性衰減學習率
        optim="adamw_torch_fused",                  # 有 CUDA 11.8 可用，速度比 adamw_torch 好一點
        logging_steps=10,                           # 每 10 step 打一次 log
        eval_strategy="epoch",                      # 每個 epoch 結束時跑一次驗證 (use eval_strategy for newer versions)
        save_strategy="epoch",                      # 每個 epoch 結束時儲存一次 checkpoint
        save_total_limit=2,                         # 最多保留 2 個 checkpoint
        load_best_model_at_end=True,                # 訓練結束後自動載入驗證集最佳模型
        metric_for_best_model="eval_loss",               # 以 validation loss 作為挑最佳模型指標
        greater_is_better=False,                    # 因為 loss 越小越好
        remove_unused_columns=False,                # <--- ADD THIS LINE
        report_to="none",                           # 不額外回報到 WandB/Comet 等
        #gradient_checkpointing=True,               # 梯度檢查點 (activation recompute) -30 %～-40 % activation 顯存，訓練速度 ↓10–15 %。
        include_for_metrics=["loss"],               # <-- 這行自動蒐集 eval loss
        prediction_loss_only=False                   # 讓 Trainer 只蒐集 loss
    )

    # 5. 建立 Trainer，並**移除 tokenization 自動流程**，改交給 ContrastiveCollator
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=ContrastiveCollator(tokenizer),
        compute_metrics=my_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("🔹 Trainer 設定完成，開始訓練並驗證...\n")
    trainer.train()
    print(f"\n✅ 訓練與驗證完成！模型已儲存於：{args.output_dir}")

if __name__ == "__main__":
    main()
