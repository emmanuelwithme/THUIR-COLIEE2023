# linux用此腳本進行查詢
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

# Preserve caller-provided env values so `.env` won't override them.
_CALLER_COLIEE_TASK1_YEAR="${COLIEE_TASK1_YEAR:-}"
_CALLER_COLIEE_TASK1_ROOT="${COLIEE_TASK1_ROOT:-}"
_CALLER_COLIEE_TASK1_DIR="${COLIEE_TASK1_DIR:-}"

if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  . "${REPO_ROOT}/.env"
  set +a
fi

if [ -n "${_CALLER_COLIEE_TASK1_YEAR}" ]; then
  COLIEE_TASK1_YEAR="${_CALLER_COLIEE_TASK1_YEAR}"
fi
if [ -n "${_CALLER_COLIEE_TASK1_ROOT}" ]; then
  COLIEE_TASK1_ROOT="${_CALLER_COLIEE_TASK1_ROOT}"
fi
if [ -n "${_CALLER_COLIEE_TASK1_DIR}" ]; then
  COLIEE_TASK1_DIR="${_CALLER_COLIEE_TASK1_DIR}"
fi

COLIEE_TASK1_YEAR="${COLIEE_TASK1_YEAR:-2025}"
COLIEE_TASK1_ROOT="${COLIEE_TASK1_ROOT:-./coliee_dataset/task1}"
TASK1_DIR="${COLIEE_TASK1_DIR:-${COLIEE_TASK1_ROOT}/${COLIEE_TASK1_YEAR}}"

# 處理驗證集查詢
echo "運行驗證集BM25檢索..."
python -m pyserini.search.lucene \
  --index "${TASK1_DIR}/lht_process/BM25/index" \
  --topics "${TASK1_DIR}/lht_process/BM25/query_valid.tsv" \
  --output "${TASK1_DIR}/lht_process/BM25/output_bm25_valid.tsv" \
  --bm25 \
  --k1 3 \
  --b 1 \
  --hits 4451 \
  --threads 10 \
  --batch-size 16 

# 處理訓練集查詢
echo "運行訓練集BM25檢索..."
python -m pyserini.search.lucene \
  --index "${TASK1_DIR}/lht_process/BM25/index" \
  --topics "${TASK1_DIR}/lht_process/BM25/query_train.tsv" \
  --output "${TASK1_DIR}/lht_process/BM25/output_bm25_train.tsv" \
  --bm25 \
  --k1 3 \
  --b 1 \
  --hits 4451 \
  --threads 10 \
  --batch-size 16

echo "BM25檢索完成！"
echo "驗證集結果保存在: ${TASK1_DIR}/lht_process/BM25/output_bm25_valid.tsv"
echo "訓練集結果保存在: ${TASK1_DIR}/lht_process/BM25/output_bm25_train.tsv"
