import json
import os

# 文件路徑
label_path = r"C:\THUIR-COLIEE2023\coliee_dataset\task1\task1_train_labels_2025.json"

# 檢查文件是否存在
if not os.path.exists(label_path):
    print(f"錯誤: 文件 {label_path} 不存在!")
    exit(1)

try:
    # 讀取標籤文件
    with open(label_path, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    
    # 計算query和documents數量
    query_count = len(label_data)
    
    # 計算所有相關文檔的總數
    total_docs = 0
    for query, docs in label_data.items():
        total_docs += len(docs)
    
    # 計算每個query平均有多少相關文檔
    avg_docs = total_docs / query_count if query_count > 0 else 0
    
    # 找出最多和最少相關文檔的query
    max_docs = 0
    min_docs = float('inf')
    max_query = ""
    min_query = ""
    
    for query, docs in label_data.items():
        if len(docs) > max_docs:
            max_docs = len(docs)
            max_query = query
        if len(docs) < min_docs:
            min_docs = len(docs)
            min_query = query
    
    # 輸出結果
    print(f"標籤文件: {label_path}")
    print(f"查詢(Query)總數: {query_count}")
    print(f"相關文檔總數: {total_docs}")
    print(f"每個查詢平均相關文檔數: {avg_docs:.2f}")
    print(f"最多相關文檔的查詢: {max_query} (共 {max_docs} 個文檔)")
    print(f"最少相關文檔的查詢: {min_query} (共 {min_docs} 個文檔)")
    
    # 分析相關文檔數量分佈
    distribution = {}
    for query, docs in label_data.items():
        doc_count = len(docs)
        if doc_count not in distribution:
            distribution[doc_count] = 0
        distribution[doc_count] += 1
    
    print("\n相關文檔數量分佈:")
    for doc_count in sorted(distribution.keys()):
        print(f"{doc_count}個相關文檔的查詢數量: {distribution[doc_count]}")
    
except Exception as e:
    print(f"處理文件時出錯: {e}") 