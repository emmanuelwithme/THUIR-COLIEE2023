#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

run_step() {
  local title="$1"
  shift
  echo
  echo "============================================================"
  echo "[STEP] ${title}"
  echo "============================================================"
  "$@"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Required file not found: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Required directory not found: ${path}" >&2
    exit 1
  fi
}

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.env"
  set +a
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-THUIR-COLIEE2023-WSL}"
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda.sh not found under ~/miniconda3 or ~/anaconda3" >&2
  exit 1
fi
conda activate "${CONDA_ENV_NAME}"

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] Conda env: ${CONDA_DEFAULT_ENV:-<unknown>}"
echo "[INFO] COLIEE_TASK2_YEAR=${COLIEE_TASK2_YEAR:-2026}"
echo "[INFO] TASK2_MODE=${TASK2_MODE:-full}"

TASK2_YEAR="${COLIEE_TASK2_YEAR:-2026}"
TASK2_ROOT="${COLIEE_TASK2_ROOT:-./coliee_dataset/task2}"
TASK2_DIR="${COLIEE_TASK2_DIR:-${TASK2_ROOT}/task2_train_files_${TASK2_YEAR}}"
TASK2_PREPARED_DIR="${COLIEE_TASK2_PREPARED_DIR:-./Legal Case Entailment by Mou/data/task2_${TASK2_YEAR}_prepared}"
TASK2_SKIP_STATS="${TASK2_SKIP_STATS:-0}"

require_dir "${TASK2_DIR}/cases"
require_file "${TASK2_DIR}/task2_train_labels_${TASK2_YEAR}.json"

run_step "Prepare task2 paragraph data" \
  python "Legal Case Entailment by Mou/prepare_task2_paragraph_data.py"

require_dir "${TASK2_PREPARED_DIR}/processed_queries"
require_dir "${TASK2_PREPARED_DIR}/processed_candidates"
require_file "${TASK2_PREPARED_DIR}/query_candidates_map.json"
require_file "${TASK2_PREPARED_DIR}/train_qid.tsv"
require_file "${TASK2_PREPARED_DIR}/valid_qid.tsv"
require_file "${TASK2_PREPARED_DIR}/finetune_data/contrastive_task2_random15_valid.json"

if [[ "${TASK2_SKIP_STATS}" == "1" ]]; then
  echo "[INFO] Skip statistics step (TASK2_SKIP_STATS=1)"
else
  run_step "Generate task2 statistics" \
    python "Legal Case Entailment by Mou/analyze_task2_stats.py"

  require_file "${TASK2_PREPARED_DIR}/stats/summary.json"
  require_file "${TASK2_PREPARED_DIR}/stats/relevant_count_distribution.csv"
  require_file "${TASK2_PREPARED_DIR}/stats/query_token_length_hist.png"
  require_file "${TASK2_PREPARED_DIR}/stats/candidate_token_length_hist.png"
fi

run_step "Train task2 paragraph encoder" \
  python "Legal Case Entailment by Mou/fine_tune_task2.py"

echo
echo "[DONE] Task2 preprocess + fine-tune finished."
