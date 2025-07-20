from typing import List
import torch
import torch.nn.functional as F
from tqdm import tqdm
from embeddings_data import EmbeddingsData  # 類別檔案：embeddings_data.py

def load_query_ids(path: str) -> List[str]:
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def compute_similarity_and_save(query_ids: List[str], query_data: EmbeddingsData, candidate_data: EmbeddingsData, output_path: str, model_name: str, similarity_type='dot') -> None:
    """
    query 向量與 candidate 向量計算相似度並輸出成 TREC 格式檔案。
    Args:
        query_ids (List[str]): 查詢文件的ID list。
        query_data (EmbeddingsData): query判決書的文件庫，在這裡是processed_new資料夾底下的所有文檔。
        candidate_data (EmbeddingsData): candidate判決書的文件庫，在這裡是processed資料夾底下的所有文檔。
        output_path (str): 輸出trec file的 .tsv 檔案路徑。
        similarity_type (str): 'dot' 或 'cos'計算相似度。
        model_name (str): 'modernBert' 或 'SAILER'
    """
    lines = []

    # 查詢文件的id及embedding (從EmbeddingsData中獲取並用valid_qids或train_qids篩選)
    actual_qids = []
    query_vecs = []

    # 篩選query cases embeddings
    for qid in query_ids:
        if qid in query_data.id2vec: #只處理出現在 valid_qids or train_qids 裡的查詢 ID
            actual_qids.append(qid)
            query_vecs.append(query_data.id2vec[qid])
    print(f"查詢總數：{len(query_ids)}，實際找到向量的查詢數：{len(actual_qids)}")
    missing_qids = [qid for qid in query_ids if qid not in query_data.id2vec]
    if missing_qids:
        print(f"⚠️ 以下查詢 ID 找不到向量，共 {len(missing_qids)} 筆：")
        print(missing_qids)

    query_vecs = torch.stack(query_vecs)

    for i, qid in enumerate(tqdm(actual_qids, desc=f"處理 {output_path}")):
        qvec = query_vecs[i].unsqueeze(0) # shape = (768,) -> (1, 768) 方便下一行與 doc_data.embeddings shape = (N_docs, 768) 計算相似度
        if similarity_type == 'dot':
            sims = torch.matmul(qvec, candidate_data.embeddings.T).squeeze(0).tolist()
        elif similarity_type == 'cos':
            sims = F.cosine_similarity(qvec, candidate_data.embeddings).tolist()
        ranked = sorted(zip(candidate_data.ids, sims), key=lambda x: x[1], reverse=True)
        for rank, (docid, score) in enumerate(ranked):
            lines.append(f"{qid} Q0 {docid} {rank+1} {score} {model_name}_{similarity_type}")

    with open(output_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    print(f"✅ 輸出完成：{output_path}")

if __name__ == "__main__":
    # 設定路徑
    model_name = "SAILER"
    # candidate判決書
    processed_doc_embedding_path = f"./coliee_dataset/task1/processed/processed_document_{model_name}_embeddings.pkl"
    # query判決書
    processed_new_doc_embedding_path = f"./coliee_dataset/task1/processed_new/processed_new_document_{model_name}_embeddings.pkl"
    valid_qid_path = f"./coliee_dataset/task1/valid_qid.tsv"
    train_qid_path = f"./coliee_dataset/task1/train_qid.tsv"
    output_dot_train_path = f"./coliee_dataset/task1/lht_process/{model_name}/output_{model_name}_dot_train.tsv"
    output_dot_valid_path = f"./coliee_dataset/task1/lht_process/{model_name}/output_{model_name}_dot_valid.tsv"
    output_cos_valid_path = f"./coliee_dataset/task1/lht_process/{model_name}/output_{model_name}_cos_valid.tsv"
    output_cos_train_path = f"./coliee_dataset/task1/lht_process/{model_name}/output_{model_name}_cos_train.tsv"

    # 載入 embeddings
    processed_doc_data = EmbeddingsData.load(processed_doc_embedding_path)
    processed_new_doc_data = EmbeddingsData.load(processed_doc_embedding_path) #改這裡

    # 載入查詢 ID
    valid_qids = load_query_ids(valid_qid_path)
    train_qids = load_query_ids(train_qid_path)

    # 執行 similarity 計算與排序輸出
    compute_similarity_and_save(valid_qids, processed_new_doc_data, processed_doc_data, output_dot_valid_path, model_name, similarity_type='dot')
    compute_similarity_and_save(train_qids, processed_new_doc_data, processed_doc_data, output_dot_train_path, model_name, similarity_type='dot')
    compute_similarity_and_save(valid_qids, processed_new_doc_data, processed_doc_data, output_cos_valid_path, model_name, similarity_type='cos')
    compute_similarity_and_save(train_qids, processed_new_doc_data, processed_doc_data, output_cos_train_path, model_name, similarity_type='cos')
