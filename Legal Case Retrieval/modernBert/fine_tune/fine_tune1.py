import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
# torch.set_float32_matmul_precision('high')
import json
from torch import nn
from dataclasses import dataclass
from typing import List, Dict, Optional
from transformers import (
    AutoTokenizer,
    ModernBertModel,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    EvalPrediction
)
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score
import random
random.seed(289)
import pynvml
pynvml.nvmlInit()
nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# 自訂 Callback，把紀錄寫到 TensorBoard
class TensorBoardExtras(TrainerCallback):
    def __init__(self):
        self.writer = None

    def _ensure_writer(self, args):
        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            logdir = os.path.join(args.output_dir, "tb", "extras")  # 與官方 writer 分開存
            os.makedirs(logdir, exist_ok=True)
            self.writer = SummaryWriter(logdir)

    def on_train_begin(self, args, state, control, **kwargs):
        self._ensure_writer(args)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Trainer 每個 logging step 會呼叫這裡，logs 內含 loss、learning_rate 等
        self._ensure_writer(args)
        if logs:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"train/{k}", v, state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        # 這裡額外寫 temperature 與每組學習率
        self._ensure_writer(args)
        model = kwargs.get("model", None)
        optimizer = kwargs.get("optimizer", None)

        if model is not None and hasattr(model, "log_temperature"):
            temp = model.log_temperature.exp().item()
            self.writer.add_scalar("train/temperature", temp, state.global_step)

        if optimizer is not None:
            for i, g in enumerate(optimizer.param_groups):
                lr = float(g.get("lr", 0.0))
                self.writer.add_scalar(f"train/lr_group_{i}", lr, state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # 每次 evaluate() 後把 eval_* 寫到 TensorBoard
        self._ensure_writer(args)
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"eval/{k}", v, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.writer:
            self.writer.flush()
            self.writer.close()

# ------------------------
# Dataset (隨機子集用)
# ------------------------
class RandomSubsetDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.data = samples
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# ------------------------
# 隨機抽樣評估的 Trainer（保留）
# ------------------------
class RandEvalTrainer(Trainer):
    def __init__(self, *args, num_eval_samples: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_eval_samples = num_eval_samples
    def evaluate(self, eval_dataset=None, *args, **kwargs):
        base_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        dataset_size = len(base_dataset)
        if dataset_size == 0:
            raise ValueError("基底驗證集長度為 0，無法抽樣。")
        k = min(self.num_eval_samples, dataset_size)
        picked_indices = random.sample(range(dataset_size), k)
        subset = [base_dataset[i] for i in picked_indices]
        rand_subset = RandomSubsetDataset(subset)
        return super().evaluate(eval_dataset=rand_subset, *args, **kwargs)

# 生成假的極限長度文本
def generate_fake_text(word_count=20000):
    vocab = ["city", "building", "traffic", "light", "road", "car", "signal", "street", "corner", "park",
             "tree", "bridge", "sky", "people", "crosswalk", "bus", "train", "station", "bike", "walk"]
    return " ".join(random.choices(vocab, k=word_count))

def generate_fake_sample():
    return {
        "query_text": generate_fake_text(),
        "positive_text": generate_fake_text(),
        "negative_texts": [generate_fake_text() for _ in range(15)]
    }

class FakeContrastiveDataset(Dataset):
    def __init__(self, n_samples: int = 4):
        self.data = [generate_fake_sample() for _ in range(n_samples)]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# ---------- compute_metrics ----------
def acc1_compute_metrics(eval_pred):
    logits = torch.tensor(eval_pred.predictions)   # (bsz, 16)
    labels = torch.tensor(eval_pred.label_ids)     # (bsz,)
    preds_top1 = logits.argmax(dim=1).cpu().numpy()
    acc1 = accuracy_score(labels.cpu().numpy(), preds_top1)
    top5_preds = torch.topk(logits, k=5, dim=1).indices
    labels_expanded = labels.view(-1, 1).expand_as(top5_preds)
    acc5 = (top5_preds == labels_expanded).any(dim=1).float().mean().item()
    if hasattr(eval_pred, "losses") and eval_pred.losses is not None:
        loss_val = float(np.mean(eval_pred.losses))
    else:
        loss_val = nn.CrossEntropyLoss()(logits, labels).item()
    return {"eval_loss": loss_val, "eval_acc1": acc1, "eval_acc5": acc5}

# ----------- Dataset 讀檔 -----------
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
    max_length: int = 4096
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        bsz = len(batch)
        q_texts = [item["query_text"] for item in batch]
        p_texts = [item["positive_text"] for item in batch]
        n_texts = [neg for item in batch for neg in item["negative_texts"]]
        all_texts = q_texts + p_texts + n_texts
        all_enc = self.tokenizer(
            all_texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        neg_count = len(n_texts) // bsz
        sizes = [bsz, bsz, bsz * neg_count]
        anchor_ids,  positive_ids,  negative_ids  = all_enc["input_ids"].split(sizes, dim=0)
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
    def __init__(self, model_name: str, device, temperature: float = 0.55555):
        super().__init__()
        self.encoder = ModernBertModel.from_pretrained(
            model_name,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
        hidden_dim = self.encoder.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 可學習的 temperature（log-param 保證 >0）
        self.log_temperature = nn.Parameter(
            torch.tensor(np.log(float(temperature)), dtype=torch.float32)
        )
        self.temperature_min = 1e-3
        self.temperature_max = 2.0

        self.encoder.config.use_cache = False
        self.encoder.enable_input_require_grads() # 打開訓練效果會好點，可以學習id->embedding
        self.encoder.gradient_checkpointing_enable()

    def encode(self, input_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.encoder(**input_batch)
        vec = output.last_hidden_state[:, 0, :]
        vec_project = self.projector(vec)
        vec_l2_norm = torch.nn.functional.normalize(vec_project, p=2, dim=-1)
        return vec_l2_norm
    
    def encode_in_chunks(self, negatives_input, chunk_size):
        ids = negatives_input["input_ids"]
        attn = negatives_input["attention_mask"]
        all_vecs = []
        for start in range(0, ids.size(0), chunk_size):
            end = start + chunk_size
            out = self.encoder(input_ids=ids[start:end], attention_mask=attn[start:end])
            cls = out.last_hidden_state[:,0,:]
            proj = self.projector(cls)
            all_vecs.append(torch.nn.functional.normalize(proj, p=2, dim=-1))
        return torch.cat(all_vecs, dim=0)

    def forward(self,
        anchor_input: Dict[str, torch.Tensor],
        positive_input: Dict[str, torch.Tensor],
        negative_input: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        print(f"forward...")
        temperature = self.log_temperature.exp().clamp(self.temperature_min, self.temperature_max)
        bsz = anchor_input["input_ids"].size(0)
        neg_count = negative_input["input_ids"].size(0) // bsz

        merged_batch = {
            "input_ids": torch.cat([
                anchor_input["input_ids"], positive_input["input_ids"], negative_input["input_ids"]
            ], dim=0),
            "attention_mask": torch.cat([
                anchor_input["attention_mask"], positive_input["attention_mask"], negative_input["attention_mask"]
            ], dim=0),
        }
        vec_all = self.encode(merged_batch)
        anchor_vec   = vec_all[:bsz]
        positive_vec = vec_all[bsz:bsz*2]
        neg_flat     = vec_all[bsz*2:]
        negative_vec = neg_flat.view(bsz, neg_count, -1)

        self.print_gpu_status("all encoded in one pass")
        pos_sim = torch.cosine_similarity(anchor_vec, positive_vec, dim=-1).unsqueeze(1)
        neg_sim = torch.cosine_similarity(anchor_vec.unsqueeze(1), negative_vec, dim=-1)
        self.print_gpu_status("cosine similarity computed")

        logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature
        loss = nn.CrossEntropyLoss()(logits, labels)
        print(f"labels: {labels} │ temperature(now)={temperature.item()}")
        return {"loss": loss, "logits": logits}
    
    def print_gpu_status(self, tag=""):
        util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
        print(f"🟩 [{tag}] GPU 使用率: {util.gpu}% │ 記憶體: {mem.used / 1024**2:.0f} MB / {mem.total / 1024**2:.0f} MB")

# ----------- 自訂 Trainer：讓 temperature 有獨立 LR -----------
class TempLRTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        args = self.args
        model = self.model

        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        decay_params, nodecay_params, temp_params = [], [], []

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "log_temperature" in n:
                temp_params.append(p)
            elif any(nd in n for nd in no_decay):
                nodecay_params.append(p)
            else:
                decay_params.append(p)

        optimizer_grouped_parameters = [
            {"params": decay_params,   "weight_decay": args.weight_decay, "lr": args.learning_rate},
            {"params": nodecay_params, "weight_decay": 0.0,               "lr": args.learning_rate},
            {"params": temp_params,    "weight_decay": 0.0,               "lr": getattr(args, "temperature_lr", args.learning_rate)},
        ]

        # 根據 TrainingArguments 選擇 AdamW；若要求 fused，嘗試啟用
        adamw_kwargs = dict(
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
        from torch.optim import AdamW
        try:
            if getattr(args, "optim", "") == "adamw_torch_fused":
                # torch>=2.0 支援 fused，若無支援會拋例外
                self.optimizer = AdamW(optimizer_grouped_parameters, **adamw_kwargs, fused=True)
            else:
                self.optimizer = AdamW(optimizer_grouped_parameters, **adamw_kwargs)
        except TypeError:
            # 沒有 fused 參數就退回一般 AdamW
            self.optimizer = AdamW(optimizer_grouped_parameters, **adamw_kwargs)

        return self.optimizer

# （可選）在 optimizer.step() 之後印出溫度
from transformers import TrainerCallback
class TempWatch(TrainerCallback):
    def on_optimizer_step(self, args, state, control, **kwargs):
        m = kwargs.get("model", None)
        if hasattr(m, "log_temperature"):
            try:
                print(f"[after step {state.global_step}] T = {m.log_temperature.exp().item():.8f}")
            except Exception:
                pass

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
    train_json_path = "./coliee_dataset/task1/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_train.json"
    valid_json_path = "./coliee_dataset/task1/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_valid.json"
    doc_folder = "./coliee_dataset/task1/processed"

    train_dataset = ContrastiveDataset(json_path=train_json_path, doc_folder=doc_folder)
    valid_dataset = ContrastiveDataset(json_path=valid_json_path, doc_folder=doc_folder)
    print(f"train_dataset: {len(train_dataset)}")

    random.shuffle(train_dataset.data)
    random.shuffle(valid_dataset.data)

    # 4. 設定 TrainingArguments
    args = TrainingArguments(
        output_dir="./modernBERT_contrastive_0912",
        dataloader_num_workers=8,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        fp16=True,
        learning_rate=5e-6,           # 給 encoder / projector
        num_train_epochs=20,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_torch_fused",
        logging_strategy="steps",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to="none",
        include_for_metrics=["loss"],
        prediction_loss_only=False,
        # 可選：指定 TB 目錄（自訂 Callback 會用 output_dir/tb/extras）
        logging_dir="./modernBERT_contrastive_0912/tb",
        # weight_decay 預設 0.0；如需可加上 weight_decay=0.01
    )
    # ✅ 給 temperature 的專屬 LR（自行調整）
    args.temperature_lr = 5e-4

    # 5. 建立 Trainer（改用 TempLRTrainer）
    trainer = TempLRTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=ContrastiveCollator(tokenizer),
        compute_metrics=acc1_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5), TempWatch(), TensorBoardExtras()],
    )

    print("🔹 Trainer 設定完成，開始訓練並驗證...\n")

    # （可選）檢查各參數組 LR
    for i, g in enumerate(trainer.create_optimizer().param_groups):
        sz = sum(p.numel() for p in g["params"])
        print(f"group {i}: lr={g['lr']}  weight_decay={g['weight_decay']}  #params={sz}")

    trainer.train()
    print(f"\n✅ 訓練與驗證完成！模型已儲存於：{args.output_dir}")

if __name__ == "__main__":
    main()
