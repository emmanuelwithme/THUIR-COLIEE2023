# linux用此腳本進行查詢
python -m pyserini.search.lucene \
  --index ./coliee_dataset/task1/lht_process/BM25/index \
  --topics ./coliee_dataset/task1/lht_process/BM25/query.tsv \
  --output ./coliee_dataset/task1/lht_process/BM25/output_bm25_all.tsv \
  --bm25 \
  --k1 3 \
  --b 1 \
  --hits 4451 \
  --threads 10 \
  --batch-size 16 \