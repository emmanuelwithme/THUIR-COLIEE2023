import json
from typing import Dict, List, Set
from collections import defaultdict
import random

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
    max_negatives: int = 15,
    random_seed: int = None
) -> None:
    """
    產生 contrastive learning 格式的 JSON 檔，包含 query_id、positive_id、negative_ids
    此版本會從 BM25 top_k 中去除正樣本後，隨機抽出 max_negatives 個負樣本。

    - bm25_path: BM25 TREC 格式結果檔路徑
    - json_positive_path: 正樣本 JSON 檔（key、value 都帶 .txt）
    - output_path: 將產出的訓練/驗證資料寫到這個 JSON 路徑
    - top_k: 從 BM25 前 top_k 名中挑選負樣本候選
    - max_negatives: 從候選負樣本中要隨機抽幾個
    - random_seed: 如果指定，會在抽取負樣本時設置隨機種子以保證可重現
    """
    if random_seed is not None:
        random.seed(random_seed)

    bm25_results: Dict[str, List[str]] = read_bm25_output_trec(bm25_path, top_k=top_k)
    positives: Dict[str, Set[str]] = read_positive_pairs_from_json(json_positive_path)

    dataset: List[Dict[str, object]] = []
    skipped_pairs = 0

    for qid, pos_set in positives.items():
        # 如果 query 不在 BM25 結果中就跳過
        if qid not in bm25_results:
            continue
        bm25_docs = bm25_results[qid]  # 這是 BM25 排序後的 doc_id 列表

        for pos_id in pos_set:
            # 候選負樣本 = BM25 top_k 裡，但不在正樣本集合中的所有 doc_id
            all_neg_candidates = [doc_id for doc_id in bm25_docs if doc_id not in pos_set]
            # 確保候選負樣本至少要有 max_negatives 個
            if len(all_neg_candidates) >= max_negatives:
                # 隨機抽 max_negatives 個
                neg_sample = random.sample(all_neg_candidates, max_negatives)
                dataset.append({
                    "query_id": qid,
                    "positive_id": pos_id,
                    "negative_ids": neg_sample
                })
            else:
                skipped_pairs += 1

    # 將結果寫到 output_path
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"✅ 共產生 {len(dataset)} 筆對比學習資料，已儲存至 {output_path}")
    print(f"⚠️ 有 {skipped_pairs} 筆 (query_id, positive_id) 因候選負樣本不足 {max_negatives} 而被略過")


if __name__ == "__main__":
    # ==== 🚀 產生 Train Set 的 JSON ====
    bm25_train_path = './coliee_dataset/task1/lht_process/BM25/output_bm25_train.tsv'
    positive_train_json_path = './coliee_dataset/task1/task1_train_labels_2025_train.json'
    # 將原本的檔名改為你想要的名字：
    output_train_json_path = './coliee_dataset/task1/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_train.json'

    generate_contrastive_json(
        bm25_path=bm25_train_path,
        json_positive_path=positive_train_json_path,
        output_path=output_train_json_path,
        top_k=100,
        max_negatives=15,
        random_seed=289
    )

    # ==== 🚀 產生 Valid Set 的 JSON ====
    bm25_valid_path = './coliee_dataset/task1/lht_process/BM25/output_bm25_valid.tsv'
    positive_valid_json_path = './coliee_dataset/task1/task1_train_labels_2025_valid.json'
    # 同樣把驗證集檔名改為你想要的名字：
    output_valid_json_path = './coliee_dataset/task1/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_valid.json'

    generate_contrastive_json(
        bm25_path=bm25_valid_path,
        json_positive_path=positive_valid_json_path,
        output_path=output_valid_json_path,
        top_k=100,
        max_negatives=15,
        random_seed=289
    )
