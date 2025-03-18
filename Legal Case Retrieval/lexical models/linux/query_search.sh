# linux用此腳本進行查詢

# 處理驗證集查詢
echo "運行驗證集BM25檢索..."
python -m pyserini.search.lucene \
  --index ./coliee_dataset/task1/lht_process/BM25/index \
  --topics ./coliee_dataset/task1/lht_process/BM25/query_valid.tsv \
  --output ./coliee_dataset/task1/lht_process/BM25/output_bm25_valid.tsv \
  --bm25 \
  --k1 3 \
  --b 1 \
  --hits 4451 \
  --threads 10 \
  --batch-size 16 

# 處理訓練集查詢
echo "運行訓練集BM25檢索..."
python -m pyserini.search.lucene \
  --index ./coliee_dataset/task1/lht_process/BM25/index \
  --topics ./coliee_dataset/task1/lht_process/BM25/query_train.tsv \
  --output ./coliee_dataset/task1/lht_process/BM25/output_bm25_train.tsv \
  --bm25 \
  --k1 3 \
  --b 1 \
  --hits 4451 \
  --threads 10 \
  --batch-size 16

echo "BM25檢索完成！"
echo "驗證集結果保存在: ./coliee_dataset/task1/lht_process/BM25/output_bm25_valid.tsv"
echo "訓練集結果保存在: ./coliee_dataset/task1/lht_process/BM25/output_bm25_train.tsv"