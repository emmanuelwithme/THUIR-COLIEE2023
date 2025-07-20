import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
#torch.set_float32_matmul_precision('high')     # 啟用 TF32 的高精度加速，加這行不能設定fp16=True
import json
from torch import nn
from dataclasses import dataclass
from typing import List, Dict, Optional
from transformers import (
    AutoTokenizer,
    ModernBertModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    EvalPrediction
)
from torch.utils.data import Dataset
import numpy as np

from sklearn.metrics import accuracy_score
import random
random.seed(289) # 設定隨機種子以保證可重現性（可選）

# ------------------------
# 先定義能「隨機抽 10 筆」的 Dataset
# ------------------------
class RandomSubsetDataset(Dataset):
    """
    這個 Dataset 只封裝一個「已經抽出的 samples list」,
    讓 Trainer.evaluate() 可以直接呼叫 DataLoader 去抓這 10 筆。
    """
    def __init__(self, samples: List[Dict]):
        self.data = samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ------------------------
# 自訂 Trainer：把 evaluate() 改為「先抽 num_eval_samples 個索引，再拿 base_dataset[idx]」
# ------------------------
class RandEvalTrainer(Trainer):
    def __init__(self, *args, num_eval_samples: int = 10, **kwargs):
        """
        num_eval_samples: 每次做 evaluate 時，隨機抽幾筆驗證集來計算指標。
        其餘參數請一律傳給父類別 Trainer。
        """
        super().__init__(*args, **kwargs)
        self.num_eval_samples = num_eval_samples

    def evaluate(self, eval_dataset=None, *args, **kwargs):
        # 1. 選擇要用的基底驗證集
        base_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        # 2. 確認 base_dataset 支援 __len__() 跟 __getitem__()
        dataset_size = len(base_dataset)
        if dataset_size == 0:
            raise ValueError("基底驗證集長度為 0，無法抽樣。")

        # 3. 決定真正要抽的索引數量
        k = min(self.num_eval_samples, dataset_size)
        picked_indices = random.sample(range(dataset_size), k)

        # 4. 用這些索引去呼叫 base_dataset.__getitem__(idx)，取得 dict list
        subset = [base_dataset[i] for i in picked_indices]

        # 5. 將抽出的 subset 包成 RandomSubsetDataset，交給父類別 evaluate
        rand_subset = RandomSubsetDataset(subset)
        return super().evaluate(eval_dataset=rand_subset, *args, **kwargs)

# 生成假的極限長度文本看會不會Out of memory，會被切到8192 tokens長度
def generate_fake_text(word_count=20000):
    # 常見英文單字集
    vocab = ["city", "building", "traffic", "light", "road", "car", "signal", "street", "corner", "park",
             "tree", "bridge", "sky", "people", "crosswalk", "bus", "train", "station", "bike", "walk"]
    
    # 隨機挑 word_count 個字並串成一段
    return " ".join(random.choices(vocab, k=word_count))
# 每筆資料包含 query、positive、15 negatives，每段各 20,000 字
def generate_fake_sample():
    return {
        "query_text": generate_fake_text(),
        "positive_text": generate_fake_text(),
        "negative_texts": [generate_fake_text() for _ in range(15)]
    }
# 建立 fake dataset 類別
class FakeContrastiveDataset(Dataset):
    def __init__(self, n_samples: int = 4):
        self.data = [generate_fake_sample() for _ in range(n_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ---------- compute_metrics: 用 accuracy@1 而不是 macro‐F1 ----------
def acc1_compute_metrics(eval_pred):
    logits = torch.tensor(eval_pred.predictions)   # (bsz, 16)
    labels = torch.tensor(eval_pred.label_ids)     # (bsz,)

    # Accuracy@1
    preds_top1 = logits.argmax(dim=1).cpu().numpy()     # (bsz,)
    acc1 = accuracy_score(labels.cpu().numpy(), preds_top1)

    # Accuracy@5
    top5_preds = torch.topk(logits, k=5, dim=1).indices  # (bsz, 5)
    labels_expanded = labels.view(-1, 1).expand_as(top5_preds)
    acc5 = (top5_preds == labels_expanded).any(dim=1).float().mean().item()

    if hasattr(eval_pred, "losses") and eval_pred.losses is not None:
        loss_val = float(np.mean(eval_pred.losses))
    else:
        loss_val = nn.CrossEntropyLoss()(logits, labels).item()

    return {"eval_loss": loss_val, "eval_acc1": acc1, "eval_acc5": acc5}

# ----------- Dataset -----------
class ContrastiveDataset(Dataset):
    def __init__(self, json_path: str, doc_folder: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.doc_folder = doc_folder
        print(f"🔹 對比式資料（{os.path.basename(json_path)}）共載入 {len(self.data)} 筆樣本")

    def __len__(self):
        return len(self.data)

    def load_text(self, doc_id: str) -> str:
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
    max_length: int = 4096 #最大8192

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        batch 裡每個項目都沒有 labels，要在這裡幫它加 labels（全部設為 0）。
        """
        q_texts = [item["query_text"] for item in batch]
        p_texts = [item["positive_text"] for item in batch]
        n_texts = [neg for item in batch for neg in item["negative_texts"]]

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

        bsz = len(batch)
        # contrastive label = 0（表示第一個位置是正樣本），Trainer 會把它當成 label_ids
        labels = torch.zeros(bsz, dtype=torch.long)

        return {
            "anchor_input": anchor_enc,
            "positive_input": positive_enc,
            "negative_input": negative_enc,
            "labels": labels, 
        }


# ----------- Model with InfoNCE loss，仅保留 backbone，不再有 projector -----------
class ModernBERTContrastive(nn.Module):
    """
    这个版本只用 ModernBertModel，把 CLS 输出当作 embedding（可做 L2 归一化），
    再直接在 forward 里计算正负样本相似度并算 InfoNCE loss。
    """
    def __init__(self, model_name: str, device, temperature: float = 0.1):
        super().__init__()
        # 1) Backbone：ModernBertModel，启用 flash_attention_2 并放到 device
        self.encoder = ModernBertModel.from_pretrained(
            model_name,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
        # 2) 只保留一个 temperature 参数
        self.temperature = temperature

        # 3) 禁用 use_cache、enable grad checkpointing（跟原来一致）
        self.encoder.config.use_cache = False
        self.encoder.enable_input_require_grads()
        self.encoder.gradient_checkpointing_enable()

    def encode(self, input_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        input_batch: tokenizer 返回的 dict，比如 {"input_ids": Tensor, "attention_mask": Tensor}
        直接把 backbone 的 CLS 向量（last_hidden_state[:,0,:]）拿出来，
        然后做 L2 归一化（因为对比学习里通常这么做）。
        返回形状 (bsz, hidden_size)
        """
        output = self.encoder(**input_batch)
        cls_vec = output.last_hidden_state[:, 0, :]               # (bsz, hidden_size)
        cls_norm = torch.nn.functional.normalize(cls_vec, p=2, dim=-1)
        return cls_norm

    def forward(
        self,
        anchor_input:   Dict[str, torch.Tensor],
        positive_input: Dict[str, torch.Tensor],
        negative_input: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        anchor_input, positive_input, negative_input 都是 tokenizer 返回的 dict。
        labels 全部为 0，表示 logits[:,0] 是正样本。
        """
        anchor_vec   = self.encode(anchor_input)    # (bsz, H)
        positive_vec = self.encode(positive_input)  # (bsz, H)
        negative_vec = self.encode(negative_input)  # (bsz * neg_count, H)

        bsz = anchor_vec.size(0)
        neg_count = negative_vec.size(0) // bsz
        negative_vec = negative_vec.view(bsz, neg_count, -1)  # (bsz, neg_count, H)

        # 正样本相似度 (bsz, 1)
        pos_sim = torch.cosine_similarity(anchor_vec, positive_vec, dim=-1).unsqueeze(1)
        # 负样本相似度 (bsz, neg_count)
        neg_sim = torch.cosine_similarity(
            anchor_vec.unsqueeze(1), negative_vec, dim=-1
        )
        # 拼成 (bsz, 1 + neg_count)，再除以 temperature
        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature

        # 直接用交叉熵 loss，labels=0 → 第一列是正样本
        loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


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
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. 初始化自定義的 Contrastive Model
    model = ModernBERTContrastive(model_name, device)
    print("✅ Tokenizer 與 Model 初始化完成\n")

    # 3. 載入訓練 / 驗證資料集
    # train_json_path = "./coliee_dataset/task1/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_train.json"
    # valid_json_path = "./coliee_dataset/task1/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_valid.json"
    train_json_path = "./coliee_dataset/task1/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_train.json"
    valid_json_path = "./coliee_dataset/task1/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_valid.json"
    doc_folder = "./coliee_dataset/task1/processed"

    train_dataset = ContrastiveDataset(json_path=train_json_path, doc_folder=doc_folder)
    valid_dataset = ContrastiveDataset(json_path=valid_json_path, doc_folder=doc_folder)

    # 隨機打亂資料順序
    random.shuffle(train_dataset.data)
    random.shuffle(valid_dataset.data)

    # 隨機抽 10 筆做試驗
    # train_dataset.data = train_dataset.data[:10]
    # valid_dataset.data = valid_dataset.data[:10]
    # print("測試模式：隨機抽樣 10 筆樣本進行訓練及驗證")

    # 產生單筆極長文本樣本（query、positive、15 個 negative）
    # train_dataset = FakeContrastiveDataset(n_samples=4)
    # valid_dataset = FakeContrastiveDataset(n_samples=4)

    # 4. 設定 TrainingArguments
    args = TrainingArguments(
        output_dir="./modernBERT_contrastive_noprojector",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        fp16=True,
        learning_rate=5e-6,
        num_train_epochs=20,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_torch_fused",

        logging_strategy="steps",
        # logging_steps=1,
        logging_steps=50,   # 每 50 步就印一次訓練 loss

        eval_strategy="epoch",
        save_strategy="epoch",
        # eval_strategy="steps",        # 改為「按步數」評估
        # eval_steps=200,               # 例如：每 200 步就跑一次驗證
        # save_strategy="steps",        # 模型存檔也可以改成「按步數」存
        # save_steps=200,               # 每 200 步存一次 Checkpoint
        save_total_limit=10,           # 最多保留 2 個 Checkpoint

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",   # ← 回傳 compute_metrics 的 key「loss」，Trainer 會找「eval_loss」
        greater_is_better=False,

        remove_unused_columns=False,
        report_to="none",
        include_for_metrics=["loss"],   # 收集各 batch 的 loss，放到 EvalPrediction.losses
        prediction_loss_only=False
    )

    # 5. 建立 Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=ContrastiveCollator(tokenizer),
        compute_metrics=acc1_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        #num_eval_samples=500  # (自定義RandEvalTrainer)每次 evaluate 要抽 10 筆
    )

    print("🔹 Trainer 設定完成，開始訓練並驗證...\n")
    trainer.train()
    print(f"\n✅ 訓練與驗證完成！模型已儲存於：{args.output_dir}")


if __name__ == "__main__":
    main()
