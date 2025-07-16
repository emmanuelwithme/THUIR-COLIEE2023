from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity

# 檢查 GPU 是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("使用 CPU")

# 初始化 ModernBERT
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", device_map=device, torch_dtype=torch.float16, attn_implementation="flash_attention_2")

# 準備輸入句子
sentences = [
    "The defendant stole a vehicle and drove away.",
    "The Court of Appeal held that the defendant was guilty of theft.",
    "The defendant was found guilty of theft by the Court of Appeal.",
    "The Court of Appeal found the defendant guilty of theft.",
    "The defendant was found guilty of theft by the Court of Appeal.",
    "The Court of Appeal found the defendant guilty of theft."
]

def get_embeddings(texts, batch_size=8):
    """Generate embeddings for a list of texts in batches"""
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(device) # 將輸入移動到設備
        with torch.no_grad():
            outputs = model(**inputs)
            # Get CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings.cpu()) # 將 embedding 移回 CPU 儲存 (可選)
            
    return torch.cat(all_embeddings, dim=0)


print(f"Generating embeddings for {len(sentences)} sentences...")
sent_embeddings = get_embeddings(sentences)
print(f"embeddings shape: {sent_embeddings.shape}")
# 計算並比較每對句子的相似度
print("Similarity Comparison:")
for i in range(len(sent_embeddings)):
    for j in range(i + 1, len(sent_embeddings)):
        # 計算餘弦相似度
        similarity = cosine_similarity(sent_embeddings[i].unsqueeze(0), sent_embeddings[j].unsqueeze(0))
        print(f"Sentence {i} vs Sentence {j}:")
        print(f"{sentences[i]}")
        print(f"{sentences[j]}")
        print(f"Cosine Similarity: {similarity.item():.4f}")
        # 通常0.8以上的相似度表示語義非常相似
        if similarity > 0.8:
            print("Result: Highly Similar")
        elif similarity > 0.6:
            print("Result: Moderately Similar")
        else:
            print("Result: Not Very Similar")
        print()
