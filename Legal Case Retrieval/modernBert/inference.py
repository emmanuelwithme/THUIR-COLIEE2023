from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root (contains the lcr package) is importable when running from repo root.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import torch
from transformers import AutoTokenizer

from lcr.data import EmbeddingsData
from find_best_model import find_best_checkpoint

QUICK_TEST = False

# Shared utilities package (contains reusable helpers for retrieval pipelines)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.device import get_device
from lcr.embeddings import process_directory_to_embeddings

# 注意：下面两行要根据你實際存放 modernbert_contrastive_model.py 的路徑來 import
sys.path.append(os.path.join(os.path.dirname(__file__), "fine_tune"))
from modernbert_contrastive_model import ModernBERTContrastive, ContrastiveConfig

import logging
def enable_my_patch(enabled: bool = True):
    """傳入True會印出自己分類為"my_debug"的DEBUG level以上等級訊息，False只會印出WARNING level以上等級的訊息"""
    level = logging.DEBUG if enabled else logging.WARNING
    logging.getLogger("my_debug").setLevel(level)

# 打開自己的Debug訊息
enable_my_patch(True)

# 檢查 GPU 是否可用
device = get_device()

# 找出最佳模型checkpoint
dir_suffix = "_test" if QUICK_TEST else ""
model_root_dir = f"./modernBERT_contrastive_adaptive{dir_suffix}"
best_loss_ckpt = find_best_checkpoint(model_root_dir, "eval_loss", mode="min")
print("最佳 eval_loss checkpoint:", best_loss_ckpt)
best_acc1_ckpt = find_best_checkpoint(model_root_dir, "eval_acc1", mode="max")
print("最佳 eval_acc1 checkpoint:", best_acc1_ckpt)
best_acc5_ckpt = find_best_checkpoint(model_root_dir, "eval_acc5", mode="max")
print("最佳 eval_acc5 checkpoint:", best_acc5_ckpt)
best_f1_ckpt = find_best_checkpoint(model_root_dir, "eval_global_f1", mode="max")
print("最佳 eval_global_f1 checkpoint:", best_f1_ckpt)
# 取出路徑(路徑, 分數)
best_loss_path,  _ = best_loss_ckpt
best_acc1_path,  _ = best_acc1_ckpt
best_acc5_path,  _ = best_acc5_ckpt
best_f1_path,  _ = best_f1_ckpt

# 載入 tokenizer + 載入模型權重
tokenizer = AutoTokenizer.from_pretrained(best_f1_path)
model = ModernBERTContrastive.from_pretrained(best_f1_path, encoder_kwargs={"device_map": device, "torch_dtype": torch.float16, "attn_implementation": "flash_attention_2"})
model = model.to(device)
model = model.half() #把projector的精度也轉成torch.float16(ModernBert backbone在from_pretrained()就指定載入是float16)
model = model.eval()

def encode_batch(batch_inputs):
    return model.encode(batch_inputs)

# Path to the processed documents
# ppp是測試的資料夾，之後正式版可以刪除
# processed_new資料夾底下的文檔是前處理只取引用前後句。在原論文中用做query。processed在原論文中用做candidate。
model_name = "modernBert"
print(f"------Using {model_name} to encode documents------\n")
candidate_dataset_path = "./coliee_dataset/task1/processed"
query_dataset_path = "./coliee_dataset/task1/processed_new"
suffix = "_test" if QUICK_TEST else ""
candidate_output_path = f"./coliee_dataset/task1/processed/processed_document_{model_name}_embeddings{suffix}.pkl"
query_output_path = f"./coliee_dataset/task1/processed_new/processed_new_document_{model_name}_embeddings{suffix}.pkl"
if QUICK_TEST:
    print("⚙️  QUICK_TEST 模式啟用：使用測試模型與輸出路徑")

# -------------------------------
# Candidate 資料集處理
# -------------------------------
print("--------------------------")
print(f"\n🔹 Encoding candidate documents located at {candidate_dataset_path} ...")
candidate_data = process_directory_to_embeddings(
    candidate_dataset_path,
    candidate_output_path,
    tokenizer,
    encode_batch=encode_batch,
    batch_size=1,
    max_length=4096,
    device=device,
    show_progress=True,
)
print(f"💾 Candidate embeddings saved to {candidate_output_path} ({len(candidate_data)} documents)")


# -------------------------------
# Query 資料集處理
# -------------------------------
print("--------------------------")
print(f"\n🔹 Encoding query documents located at {query_dataset_path} ...")
query_data = process_directory_to_embeddings(
    query_dataset_path,
    query_output_path,
    tokenizer,
    encode_batch=encode_batch,
    batch_size=1,
    max_length=4096,
    device=device,
    show_progress=True,
)
print(f"💾 Query embeddings saved to {query_output_path} ({len(query_data)} documents)")

print("\n✅ All embeddings saved successfully.")
