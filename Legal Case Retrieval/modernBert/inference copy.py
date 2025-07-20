import os, sys
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import pickle
from tqdm import tqdm
from embeddings_data import EmbeddingsData
from find_best_model import find_best_checkpoint
# 注意：下面两行要根据你实际存放 modernbert_contrastive_model.py 的路径来 import
sys.path.append(os.path.join(os.path.dirname(__file__), 'fine_tune'))
from modernbert_contrastive_model import ModernBERTContrastive, ContrastiveConfig

# 檢查 GPU 是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("使用 CPU")

# 找出最佳模型checkpoint
model_root_dir = "./modernBERT_contrastive"
best_loss_ckpt = find_best_checkpoint(model_root_dir, "eval_loss", mode="min")
print("最佳 eval_loss checkpoint:", best_loss_ckpt)
best_acc1_ckpt = find_best_checkpoint(model_root_dir, "eval_acc1", mode="max")
print("最佳 eval_acc1 checkpoint:", best_acc1_ckpt)
best_acc5_ckpt = find_best_checkpoint(model_root_dir, "eval_acc5", mode="max")
print("最佳 eval_acc5 checkpoint:", best_acc5_ckpt)
# 载入 tokenizer + ContrastiveConfig + 载入整个 ModernBERTContrastive
config = ContrastiveConfig.from_pretrained(best_loss_ckpt)
tokenizer = AutoTokenizer.from_pretrained(best_loss_ckpt)
model = ModernBERTContrastive.from_pretrained(best_loss_ckpt, device_map=device, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
model.eval()

def get_embeddings(texts, batch_size=8):
    """Generate embeddings for a list of texts in batches"""
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device) # 將輸入移動到設備
        with torch.no_grad():
            outputs = model(**inputs)
            # Get CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings.cpu()) # 將 embedding 移回 CPU 儲存 (可選)
            
    return torch.cat(all_embeddings, dim=0)

# Path to the processed documents
# ppp是測試的資料夾，之後正式版可以刪除
# processed_new資料夾底下的文檔是前處理只取引用前後句。在原論文中用做query。processed在原論文中用做candidate。
model_name = "modernBert"
print(f"------Using {model_name} to encode documents------\n")
candidate_dataset_path = "./coliee_dataset/task1/processed"
query_dataset_path = "./coliee_dataset/task1/processed_new"
candidate_output_path = f"./coliee_dataset/task1/processed/processed_document_{model_name}_embeddings.pkl"
query_output_path = f"./coliee_dataset/task1/processed_new/processed_new_document_{model_name}_embeddings.pkl"

# -------------------------------
# Candidate 資料集處理
# -------------------------------
print("--------------------------")
print(f"\n🔹 Reading candidate documents from {candidate_dataset_path}...")
candidate_ids = []
candidate_texts = []

for filename in tqdm(os.listdir(candidate_dataset_path)):
    if filename.endswith(".txt"):
        doc_id = filename.replace(".txt", "")
        file_path = os.path.join(candidate_dataset_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            candidate_ids.append(doc_id)
            candidate_texts.append(f.read().strip())

print(f"🔹 Generating candidate embeddings for {len(candidate_texts)} documents...")
candidate_embeddings = get_embeddings(candidate_texts)

# 儲存 candidate 向量
candidate_data = EmbeddingsData(candidate_ids, candidate_embeddings)
print(f"💾 Saving candidate embeddings to {candidate_output_path}...")
candidate_data.save(candidate_output_path)


# -------------------------------
# Query 資料集處理
# -------------------------------
print("--------------------------")
print(f"\n🔹 Reading query documents from {query_dataset_path}...")
query_ids = []
query_texts = []

for filename in tqdm(os.listdir(query_dataset_path)):
    if filename.endswith(".txt"):
        doc_id = filename.replace(".txt", "")
        file_path = os.path.join(query_dataset_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            query_ids.append(doc_id)
            query_texts.append(f.read().strip())

print(f"🔹 Generating query embeddings for {len(query_texts)} documents...")
query_embeddings = get_embeddings(query_texts)

# 儲存 query 向量
query_data = EmbeddingsData(query_ids, query_embeddings)
print(f"💾 Saving query embeddings to {query_output_path}...")
query_data.save(query_output_path)

print("\n✅ All embeddings saved successfully.")