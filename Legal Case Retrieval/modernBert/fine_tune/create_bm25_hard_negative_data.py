import json
import sys
from typing import Dict, List, Set
from collections import defaultdict
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.task1_paths import get_task1_dir, get_task1_year

TASK1_DIR = get_task1_dir()
TASK1_YEAR = get_task1_year()

def read_bm25_output_trec(tsv_path: str, top_k: int = 100) -> Dict[str, List[str]]:
    """讀取 TREC 格式 BM25 檢索結果，補齊 query_id 和 doc_id 至 6 位數"""
    bm25_results: Dict[str, List[str]] = defaultdict(list)
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            query_id_raw, _, doc_id_raw, rank_str, score, _ = parts
            rank = int(rank_str)
            if rank <= top_k:
                query_id = query_id_raw.zfill(6)
                doc_id = doc_id_raw.zfill(6)
                bm25_results[query_id].append(doc_id)
    return bm25_results

def read_positive_pairs_from_json(json_path: str) -> Dict[str, Set[str]]:
    """從 JSON 讀取正樣本對映表，並去除 .txt 副檔名"""
    positives: Dict[str, Set[str]] = defaultdict(set)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for q_txt, pos_list in data.items():
        qid = q_txt.replace(".txt", "")
        for doc_txt in pos_list:
            doc_id = doc_txt.replace(".txt", "")
            positives[qid].add(doc_id)
    return positives

def generate_contrastive_json(
    bm25_path: str,
    json_positive_path: str,
    output_path: str,
    top_k: int = 100,
    max_negatives: int = 15
) -> None:
    """
    產生 contrastive learning 格式的 JSON 檔，包含 query_id、positive_id、negative_ids
    僅保留那些擁有至少 max_negatives 筆負樣本的樣本
    """
    bm25_results: Dict[str, List[str]] = read_bm25_output_trec(bm25_path, top_k=top_k)
    positives: Dict[str, Set[str]] = read_positive_pairs_from_json(json_positive_path)

    dataset: List[Dict[str, object]] = []
    skipped_queries = 0

    for qid, pos_set in positives.items():
        if qid not in bm25_results:
            continue
        bm25_docs = bm25_results[qid]
        for pos_id in pos_set:
            # 過濾出不是正樣本的 BM25 文件作為負樣本候選
            negatives = [doc_id for doc_id in bm25_docs if doc_id not in pos_set]
            if len(negatives) >= max_negatives:
                dataset.append({
                    "query_id": qid,
                    "positive_id": pos_id,
                    "negative_ids": negatives[:max_negatives]
                })
            else:
                skipped_queries += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"✅ 共產生 {len(dataset)} 筆對比學習資料，已儲存至 {output_path}")
    print(f"⚠️ 有 {skipped_queries} 筆樣本因負樣本不足 {max_negatives} 而被略過")

# ==== 🚀 產生 Train Set 的 JSON ====
bm25_train_path: str = f"{TASK1_DIR}/lht_process/BM25/output_bm25_train.tsv"
positive_train_json_path: str = f"{TASK1_DIR}/task1_train_labels_{TASK1_YEAR}_train.json"
output_train_json_path: str = f"{TASK1_DIR}/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_train.json"

generate_contrastive_json(
    bm25_path=bm25_train_path,
    json_positive_path=positive_train_json_path,
    output_path=output_train_json_path,
    top_k=100,
    max_negatives=15
)

# ==== 🚀 產生 Valid Set 的 JSON ====
bm25_valid_path: str = f"{TASK1_DIR}/lht_process/BM25/output_bm25_valid.tsv"
positive_valid_json_path: str = f"{TASK1_DIR}/task1_train_labels_{TASK1_YEAR}_valid.json"
output_valid_json_path: str = f"{TASK1_DIR}/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_valid.json"

generate_contrastive_json(
    bm25_path=bm25_valid_path,
    json_positive_path=positive_valid_json_path,
    output_path=output_valid_json_path,
    top_k=100,
    max_negatives=15
)
