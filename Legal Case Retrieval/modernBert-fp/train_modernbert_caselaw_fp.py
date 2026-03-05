import logging
import os
import random
from statistics import median
from typing import Dict, Iterable, List, Tuple

import torch
from datasets import IterableDataset as HFIterableDataset, load_from_disk
from torch.utils.data import IterableDataset as TorchIterableDataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


LOGGER = logging.getLogger(__name__)
DATASET_LOCAL_DIR = os.path.join("data", "caselaw_access_project_filtered")
MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_SEQ_LENGTH = 4096


class IterableDatasetWithLength(TorchIterableDataset):
    def __init__(self, dataset: HFIterableDataset, length: int):
        if length <= 0:
            raise ValueError("Length for IterableDatasetWithLength must be positive.")
        self.dataset = dataset
        self.length = length

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return self.length


def _check_cuda() -> None:
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is required but not available.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class PackedIterableDataset(TorchIterableDataset):
    """Pack consecutive texts into fixed-length token sequences for MLM."""

    def __init__(
        self,
        dataset: Iterable,
        tokenizer,
        max_length: int,
        min_tokens_to_yield: int = 128,
        drop_remainder: bool = True,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.body_length = max_length - 2  # reserve CLS + SEP
        self.min_tokens_to_yield = min_tokens_to_yield
        self.drop_remainder = drop_remainder

        if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None or tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define cls_token_id, sep_token_id, and pad_token_id for packing.")

    def __iter__(self):
        tokenizer = self.tokenizer
        buffer: List[int] = []
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id

        for sample in self.dataset:
            text = sample["text"]
            tokenized: List[int] = tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
                max_length=None,  # disable internal length check; we pack/truncate manually
            )
            if not tokenized:
                continue
            buffer.extend(tokenized)

            while len(buffer) >= self.body_length:
                chunk = buffer[: self.body_length]
                buffer = buffer[self.body_length :]
                yield self._build_example(chunk, cls_id, sep_id, pad_id)

        if not self.drop_remainder and len(buffer) >= self.min_tokens_to_yield:
            chunk = buffer[: self.body_length]
            yield self._build_example(chunk, cls_id, sep_id, pad_id)

    def _build_example(self, chunk: List[int], cls_id: int, sep_id: int, pad_id: int):
        body = chunk[: self.body_length]
        input_ids = [cls_id] + body + [sep_id]
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids.extend([pad_id] * pad_len)

        attn_len = len(body) + 2
        attention_mask = [1] * attn_len + [0] * pad_len
        special_tokens_mask = [1] + [0] * len(body) + [1] + [1] * pad_len
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
        }


class TruncatingDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """Ensures input sequences are at most max_length before collating."""

    def __init__(self, *args, max_length: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_length = max_length

    def __call__(self, features):
        truncated = []
        for f in features:
            if len(f["input_ids"]) > self.max_length:
                f = {
                    "input_ids": f["input_ids"][: self.max_length],
                    "attention_mask": f["attention_mask"][: self.max_length],
                    "special_tokens_mask": f["special_tokens_mask"][: self.max_length],
                }
            truncated.append(f)
        return super().__call__(truncated)


def _ensure_local_dataset() -> None:
    if not os.path.isdir(DATASET_LOCAL_DIR):
        raise FileNotFoundError(
            f"找不到本地資料夾 {DATASET_LOCAL_DIR}，請先執行 download_caselaw_dataset.py 下載資料。"
        )


def load_local_splits(
    eval_fraction: float,
    num_shards: int | None = None,
) -> Tuple[HFIterableDataset, HFIterableDataset, List[str], object, object]:
    _ensure_local_dataset()
    dataset = load_from_disk(DATASET_LOCAL_DIR)
    column_names = dataset.column_names
    if "text" not in column_names:
        raise ValueError(f"資料集中找不到 'text' 欄位：{column_names}")

    shuffled = dataset.shuffle(seed=42)
    split = shuffled.train_test_split(test_size=eval_fraction, seed=42)
    train_split = split["train"]
    eval_split = split["test"]

    if num_shards is None or num_shards <= 0:
        train_iter = train_split.to_iterable_dataset()
        eval_iter = eval_split.to_iterable_dataset()
    else:
        # 關鍵：把單一 Dataset 切成 num_shards 份，供多個 dataloader worker 各讀一份
        train_iter = train_split.to_iterable_dataset(num_shards=num_shards)
        eval_iter = eval_split.to_iterable_dataset(num_shards=num_shards)

    return train_iter, eval_iter, column_names, train_split, eval_split


def estimate_packed_length(
    dataset,
    tokenizer,
    *,
    max_length: int,
    sample_size: int,
    min_tokens_to_yield: int,
    drop_remainder: bool,
) -> Tuple[int, Dict[str, float | int]]:
    total_docs = len(dataset)
    if total_docs == 0:
        raise ValueError("本地資料集為空，無法訓練。")
    sample_size = min(sample_size, total_docs)
    indices = list(range(total_docs))
    random.Random(42).shuffle(indices)
    indices = indices[:sample_size]

    body_length = max_length - 2
    token_counts: List[int] = []
    for idx in indices:
        text = dataset[idx]["text"]
        tokenized_len = len(tokenizer.encode(text, add_special_tokens=False, truncation=False, max_length=None))
        if tokenized_len == 0:
            continue
        token_counts.append(tokenized_len)
    if not token_counts:
        raise ValueError("樣本中的文本長度皆為 0，無法估算序列數。")

    avg_tokens = sum(token_counts) / len(token_counts)
    total_tokens_est = avg_tokens * total_docs
    full_sequences = int(total_tokens_est // body_length)
    remainder_tokens = total_tokens_est - (full_sequences * body_length)

    if drop_remainder:
        estimated_sequences = max(full_sequences, 1)
    else:
        estimated_sequences = max(full_sequences + (1 if remainder_tokens >= min_tokens_to_yield else 0), 1)
    stats: Dict[str, float | int] = {
        "sampled": len(token_counts),
        "avg_tokens": avg_tokens,
        "median_tokens": median(token_counts),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "estimated_total_tokens": total_tokens_est,
        "remainder_tokens": remainder_tokens,
    }
    return estimated_sequences, stats


def build_trainer() -> Tuple[Trainer, AutoTokenizer]:
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        model_max_length=MAX_SEQ_LENGTH,
        use_fast=True,
        trust_remote_code=True,
    )
    # Avoid tokenizer warnings on long raw docs; we enforce 4096 downstream.
    tokenizer.model_max_length = 1_000_000_000

    eval_fraction = 0.001  # 使用 0.1% 做 eval，兼顧速度與代表性
    training_args = TrainingArguments(
        output_dir="checkpoints/modernbert-caselaw-accsteps",
        overwrite_output_dir=False,
        logging_dir="runs/modernbert-caselaw-accsteps",
        report_to=["tensorboard"],
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        weight_decay=1e-5,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.02,
        num_train_epochs=3,
        eval_strategy="steps",
        # eval_steps=1_000,
        eval_steps=500,
        save_strategy="steps",
        # save_steps=1_000,
        save_steps=500,
        save_total_limit=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=3,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        per_device_eval_batch_size=1,
        dataloader_num_workers=8,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        optim="adamw_torch_fused",
        tf32=True,
    )

    train_iter, eval_iter, _, train_split, eval_split = load_local_splits(
        eval_fraction=eval_fraction,
        num_shards=training_args.dataloader_num_workers,
    )
    train_ds = PackedIterableDataset(
        dataset=train_iter,
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LENGTH,
        min_tokens_to_yield=MAX_SEQ_LENGTH - 2,
        drop_remainder=True,
    )
    eval_ds = PackedIterableDataset(
        dataset=eval_iter,
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LENGTH,
        min_tokens_to_yield=512,
        drop_remainder=False,
    )

    train_length_estimate, train_token_stats = estimate_packed_length(
        train_split,
        tokenizer,
        max_length=MAX_SEQ_LENGTH,
        sample_size=1024,
        min_tokens_to_yield=MAX_SEQ_LENGTH - 2,
        drop_remainder=True,
    )
    eval_length_estimate, eval_token_stats = estimate_packed_length(
        eval_split,
        tokenizer,
        max_length=MAX_SEQ_LENGTH,
        sample_size=256,
        min_tokens_to_yield=512,
        drop_remainder=False,
    )
    LOGGER.info(
        (
            "Estimated packed lengths -> train: %s (sample=%s, avg_tokens=%.0f, median=%.0f, total_tokens≈%.2fB), "
            "eval: %s (sample=%s, avg_tokens=%.0f, median=%.0f, total_tokens≈%.2fB)"
        ),
        train_length_estimate,
        train_token_stats["sampled"],
        train_token_stats["avg_tokens"],
        train_token_stats["median_tokens"],
        train_token_stats["estimated_total_tokens"] / 1e9,
        eval_length_estimate,
        eval_token_stats["sampled"],
        eval_token_stats["avg_tokens"],
        eval_token_stats["median_tokens"],
        eval_token_stats["estimated_total_tokens"] / 1e9,
    )
    train_ds = IterableDatasetWithLength(train_ds, length=train_length_estimate)
    eval_ds = IterableDatasetWithLength(eval_ds, length=eval_length_estimate)

    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda"},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    data_collator = TruncatingDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.30,
        pad_to_multiple_of=8,
        max_length=MAX_SEQ_LENGTH,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4, early_stopping_threshold=0.0)],
    )
    return trainer, tokenizer

def print_precision_and_loader_info(trainer: Trainer) -> None:
    """在訓練開始前印出 dtype / 精度 / dataloader 設定到 terminal。"""
    args = trainer.args
    model = trainer.model

    # 取一個 sample parameter 來看 dtype & device
    try:
        sample_param = next(model.parameters())
        param_dtype = sample_param.dtype
        param_device = sample_param.device
    except StopIteration:
        param_dtype = None
        param_device = None

    buffers = list(model.buffers())
    buffer_dtype = buffers[0].dtype if buffers else None

    lines = []
    lines.append("=" * 80)
    lines.append("🧪 Precision / Dtype 檢查")
    lines.append(f"- 模型參數 dtype:          {param_dtype}")
    lines.append(f"- 模型 buffer dtype:       {buffer_dtype}")
    lines.append(f"- 模型所在裝置:            {param_device}")
    lines.append(f"- Trainer bf16:            {args.bf16}")
    lines.append(f"- Trainer fp16:            {args.fp16}")
    lines.append(f"- Trainer tf32:            {args.tf32}")
    lines.append(f"- torch.get_default_dtype(): {torch.get_default_dtype()}")
    lines.append("")
    lines.append("🧵 DataLoader / 訓練相關設定")
    lines.append(f"- per_device_train_batch_size: {args.per_device_train_batch_size}")
    lines.append(f"- gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    lines.append(
        f"  ➜ 有效 batch (seq / step，單 GPU): "
        f"{args.per_device_train_batch_size * args.gradient_accumulation_steps}"
    )
    lines.append(f"- dataloader_num_workers:      {args.dataloader_num_workers}")
    lines.append(f"- dataloader_persistent_workers: {args.dataloader_persistent_workers}")
    lines.append("")
    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(device_idx)
        lines.append("🖥 GPU 資訊")
        lines.append(f"- GPU 名稱:   {prop.name}")
        lines.append(f"- GPU 顯存:   {prop.total_memory / 1024**3:.2f} GB")
        lines.append(f"- Compute Capability: {prop.major}.{prop.minor}")
    lines.append("=" * 80)

    print("\n".join(lines))



def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _check_cuda()

    trainer, tokenizer = build_trainer()
    print_precision_and_loader_info(trainer) # 印出 dtype / 精度 / dataloader 設定
    last_checkpoint = None
    if os.path.isdir(trainer.args.output_dir):
        last_checkpoint = get_last_checkpoint(trainer.args.output_dir)
        if last_checkpoint:
            LOGGER.info("Resuming from checkpoint: %s", last_checkpoint)

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    LOGGER.info("Training completed. Metrics: %s", train_result.metrics)

    trainer.save_model(trainer.args.output_dir)
    tokenizer.save_pretrained(trainer.args.output_dir)


if __name__ == "__main__":
    main()
