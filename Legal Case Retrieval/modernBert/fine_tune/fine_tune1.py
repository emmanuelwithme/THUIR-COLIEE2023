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
import pynvml
pynvml.nvmlInit()
nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0

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
    max_length: int = 4096  # 或你想要的長度

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        bsz = len(batch)
        q_texts = [item["query_text"] for item in batch]
        p_texts = [item["positive_text"] for item in batch]
        n_texts = [neg for item in batch for neg in item["negative_texts"]]

        # 一次把所有  q + p + n 都 tokenize
        all_texts = q_texts + p_texts + n_texts
        all_enc = self.tokenizer(
            all_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 切回三份：anchor (bsz)、positive (bsz)、negative (bsz*neg_count)
        neg_count = len(n_texts) // bsz
        sizes = [bsz, bsz, bsz * neg_count]
        anchor_ids, positive_ids, negative_ids = all_enc["input_ids"].split(sizes, dim=0)
        anchor_mask, positive_mask, negative_mask = all_enc["attention_mask"].split(sizes, dim=0)

        labels = torch.zeros(bsz, dtype=torch.long)
        return {
            "anchor_input":   {"input_ids": anchor_ids,   "attention_mask": anchor_mask},
            "positive_input": {"input_ids": positive_ids, "attention_mask": positive_mask},
            "negative_input": {"input_ids": negative_ids, "attention_mask": negative_mask},
            "labels": labels,
        }


# ----------- Model with InfoNCE loss -----------
class ModernBERTContrastive(nn.Module):
    def __init__(self, model_name: str, device, temperature: float = 1):
        super().__init__()
        self.encoder = ModernBertModel.from_pretrained(
            model_name,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
        # SimCLR、MoCo、Sentence‐BERT 等论文里都建议：把 BERT/ResNet 的输出 CLS 向量再过一个小 MLP（通常是一层或两层线性+ReLU）→ 再做归一化 → 最后去算 InfoNCE loss。
        hidden_dim = self.encoder.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temperature = temperature
        self.encoder.config.use_cache = False
        self.encoder.enable_input_require_grads()
        self.encoder.gradient_checkpointing_enable()

    def encode(self, input_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        input_batch 會是 tokenizer 回傳的 dict，以下例：
        {"input_ids": Tensor, "attention_mask": Tensor}
        """
        output = self.encoder(**input_batch)
        # CLS token 在 index 0
        vec = output.last_hidden_state[:, 0, :]             # (bsz, hidden_size)
        vec_project = self.projector(vec)
        vec_l2_norm = torch.nn.functional.normalize(vec_project, p=2, dim=-1) #
        return vec_l2_norm
    
    def encode_in_chunks(self, negatives_input, chunk_size):
        ids = negatives_input["input_ids"]
        attn = negatives_input["attention_mask"]
        all_vecs = []
        for start in range(0, ids.size(0), chunk_size):
            end = start + chunk_size
            out = self.encoder(
                input_ids=ids[start:end],
                attention_mask=attn[start:end]
            )
            cls = out.last_hidden_state[:,0,:]
            proj = self.projector(cls)
            all_vecs.append(torch.nn.functional.normalize(proj, p=2, dim=-1))
        return torch.cat(all_vecs, dim=0)

    def forward(
        self,
        anchor_input: Dict[str, torch.Tensor],
        positive_input: Dict[str, torch.Tensor],
        negative_input: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        注意：這裡把 labels 收進 forward，Trainer 會自動把 collator 的
        labels 參數 pop 出來並傳進來。
        """
        print(f"forward...")
        
        bsz = anchor_input["input_ids"].size(0)
        neg_count = negative_input["input_ids"].size(0) // bsz

        # 1. 合併成一個大 batch
        merged_batch = {
            "input_ids": torch.cat([
                anchor_input["input_ids"],
                positive_input["input_ids"],
                negative_input["input_ids"],
            ], dim=0),
            "attention_mask": torch.cat([
                anchor_input["attention_mask"],
                positive_input["attention_mask"],
                negative_input["attention_mask"],
            ], dim=0),
        }

        # 2. 一次呼叫 encode()（裡面已經有 projector + normalize）
        vec_all = self.encode(merged_batch)   # shape: (bsz*2 + bsz*neg_count, H)

        # 3. 拆回 anchor / positive / negative
        anchor_vec   = vec_all[:bsz]
        positive_vec = vec_all[bsz:bsz*2]
        neg_flat     = vec_all[bsz*2:]
        negative_vec = neg_flat.view(bsz, neg_count, -1)

        self.print_gpu_status("all encoded in one pass")

        # 正樣本相似度
        pos_sim = torch.cosine_similarity(anchor_vec, positive_vec, dim=-1).unsqueeze(1)  # (bsz, 1)
        # 負樣本相似度
        neg_sim = torch.cosine_similarity(anchor_vec.unsqueeze(1), negative_vec, dim=-1)  # (bsz, neg_count)
        self.print_gpu_status("cosine similarity computed")

        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature  # (bsz, 1+neg_count)，temperature通常是 0.05 ~ 0.2
        # collator 裡已經給了 labels=0
        loss = nn.CrossEntropyLoss()(logits, labels)
        print(f"labels: {labels}")

        return {"loss": loss, "logits": logits}
    
    def print_gpu_status(self, tag=""):
        util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
        print(f"🟩 [{tag}] GPU 使用率: {util.gpu}% │ 記憶體: {mem.used / 1024**2:.0f} MB / {mem.total / 1024**2:.0f} MB")


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

    print(f"train_dataset: {len(train_dataset)}")

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
        output_dir="./modernBERT_contrastive",
        dataloader_num_workers=8,  # 多核讀資料，減少GPU等CPU、I/O
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
