from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.task1_paths import get_task1_dir, get_task1_year

TASK1_DIR = get_task1_dir()
TASK1_YEAR = get_task1_year()

from lcr.data import EmbeddingsData, load_query_ids
from lcr.similarity import compute_similarity_and_save

if __name__ == "__main__":
    # 設定路徑
    model_name = "SAILER"
    # candidate判決書
    processed_doc_embedding_path = f"{TASK1_DIR}/processed/processed_document_{model_name}_embeddings.pkl"
    # query判決書
    processed_new_doc_embedding_path = f"{TASK1_DIR}/processed_new/processed_new_document_{model_name}_embeddings.pkl"
    valid_qid_path = f"{TASK1_DIR}/valid_qid.tsv"
    train_qid_path = f"{TASK1_DIR}/train_qid.tsv"
    output_dot_train_path = f"{TASK1_DIR}/lht_process/{model_name}/output_{model_name}_dot_train.tsv"
    output_dot_valid_path = f"{TASK1_DIR}/lht_process/{model_name}/output_{model_name}_dot_valid.tsv"
    output_cos_valid_path = f"{TASK1_DIR}/lht_process/{model_name}/output_{model_name}_cos_valid.tsv"
    output_cos_train_path = f"{TASK1_DIR}/lht_process/{model_name}/output_{model_name}_cos_train.tsv"
    model_scope_path = Path(f"{TASK1_DIR}/lht_process/{model_name}/query_candidate_scope.json")
    shared_scope_path = Path(f"{TASK1_DIR}/lht_process/modernBert/query_candidate_scope.json")
    if model_scope_path.exists():
        query_candidate_scope_path = model_scope_path
        print(f"🔹 使用 query candidate scope: {query_candidate_scope_path}")
    elif shared_scope_path.exists():
        query_candidate_scope_path = shared_scope_path
        print(f"🔹 未找到 {model_scope_path}，改用共用 scope: {query_candidate_scope_path}")
    else:
        print(f"⚠️ 未找到 {model_scope_path}，也未找到 {shared_scope_path}。")
        print("⚠️ 將對全部 candidates 計分。")
        query_candidate_scope_path = None

    # 載入 embeddings
    processed_doc_data = EmbeddingsData.load(processed_doc_embedding_path)
    processed_new_doc_data = EmbeddingsData.load(processed_doc_embedding_path)

    # 載入查詢 ID
    valid_qids = load_query_ids(valid_qid_path)
    train_qids = load_query_ids(train_qid_path)

    # 執行 similarity 計算與排序輸出
    for split_name, qids in [("valid", valid_qids), ("train", train_qids)]:
        for metric in ["dot", "cos"]:
            output_path = {
                ("valid", "dot"): output_dot_valid_path,
                ("train", "dot"): output_dot_train_path,
                ("valid", "cos"): output_cos_valid_path,
                ("train", "cos"): output_cos_train_path,
            }[(split_name, metric)]
            missing = compute_similarity_and_save(
                qids,
                processed_new_doc_data,
                processed_doc_data,
                output_path,
                metric=metric,
                run_tag=f"{model_name}_{metric}",
                query_candidate_scope_path=query_candidate_scope_path,
            )
            if missing:
                print(f"⚠️ {split_name} split 缺少 {len(missing)} 個查詢向量：{missing}")
            print(f"✅ 已輸出 {split_name} split / {metric} 相似度至 {output_path}")
