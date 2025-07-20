from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer

# 初始化模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("CSHaitao/SAILER_en_finetune")
model = AutoModel.from_pretrained("CSHaitao/SAILER_en_finetune")

# 準備輸入句子
sentences = [
    "The defendant stole a vehicle and drove away.",
    "The Court of Appeal held that the defendant was guilty of theft.",
    "The defendant was found guilty of theft by the Court of Appeal.",
    "The Court of Appeal found the defendant guilty of theft.",
    "The defendant was found guilty of theft by the Court of Appeal.",
    "The Court of Appeal found the defendant guilty of theft."
]

# 將所有句子轉為模型輸入格式
inputs_arr = [tokenizer(sent, return_tensors="pt", truncation=True, 
                       padding="max_length", max_length=512) 
             for sent in sentences]

# 儲存所有句子的CLS embeddings
embeddings = []

# forward pass
with torch.no_grad():
    for i, inputs in enumerate(inputs_arr):
        outputs = model(**inputs)
        # 獲取CLS token的embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: [1, 768]
        embeddings.append(cls_embedding)
        print(f"Input {i}:")
        print(f"Sentence: {sentences[i]}")
        print(f"Shape: {outputs.last_hidden_state.shape}")
        print(f"CLS embedding shape: {cls_embedding.shape}\n")

# 計算並比較每對句子的相似度
print("Similarity Comparison:")
for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        # 計算餘弦相似度
        similarity = cosine_similarity(embeddings[i], embeddings[j])
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