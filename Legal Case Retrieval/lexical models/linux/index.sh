# linux用此腳本建立索引
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ./coliee_dataset/task1/lht_process/BM25/corpus \
  --index ./coliee_dataset/task1/lht_process/BM25/index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 28 \
  --storePositions --storeDocvectors --storeRaw \