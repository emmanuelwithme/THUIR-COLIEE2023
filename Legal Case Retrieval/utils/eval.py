
import json
# from sklearn.metrics import (accuracy_score, f1_score, classification_report)
import os
import random
import pandas as pd




def my_classification_report(list_label_ohe, list_answer_ohe):
    """
    Calculate F1, Precision and Recall

    Args:
      list_label_ohe: list of one hot encodings of the labels(正確答案)
      list_answer_ohe: list of one hot encodings of the answers(預測答案)

    Returns:
      F1, Precision, Recall
    """  
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for list_label, list_ohe in zip(list_label_ohe, list_answer_ohe):
      for label in list_label:
        if label in list_ohe:
          true_positive += 1
        else:
          false_negative += 1
      for answer in list_ohe:
        if answer not in list_label:
          false_positive += 1

    precision = true_positive/(true_positive+false_positive) if (true_positive + false_positive)!=0 else 0.0
    recall = true_positive/(true_positive+false_negative) if (true_positive + false_negative)!=0 else 0.0
    f1 = 2*((precision*recall)/(precision + recall)) if (precision + recall)!=0 else 0.0

    return f1, precision, recall


def trec_file_convert(trec_path, topk):
  """
  Convert the 預測資料 TREC file: (<QueryID> Q0 <DocID> <Rank> <Score> <RunTag>) to a dict(trec_dict)
  篩選前topk個排名的相關判決書，並且不包含query id和document id相同的情況

  Args:
    trec_path: path to the TREC file
    topk: number of top documents to keep for each query
  Returns:
    trec_dict: qid: [topk document ids]
  """
  trec_file = open(trec_path,'r')
  trec_dict = {}
  for line in trec_file:
    line = line.strip().split(' ')
    qid = int(line[0]) # query id
    pid = line[2] # document id

    # if pid not in all_dict[str(qid)]:
    #   continue
    if int(pid) == int(qid):
      continue
    if qid not in trec_dict:
      trec_dict[qid] = []
    if len(trec_dict[qid]) < topk:
      trec_dict[qid].append(pid)
  
  # print(trec_dict)
  return trec_dict
    
def rel_file_convert(rel_path, valid_path):
  """
  Convert 相關判決書答案資料的json檔案，並且篩選出在驗證資料裡的query id， to a dict(rel_dict)
  Arg:
    rel_path: path to the 相關判決書答案資料的json檔案(主辦單位給的)
    valid_path: path to the 驗證資料的query id檔案(自己建的)
  Returns:
    rel_dict: qid: [relevant document ids]
  """

  valid_list = []
  valid_file = open(valid_path,'r')
  for line in valid_file.readlines():
      line = line.strip().split(' ')
      qid = line[0]
      valid_list.append(int(qid))

  rel_file = open(rel_path,'r')
  label_dict = json.load(rel_file)
  rel_dcit = {}
  for qid in label_dict.keys():
    # print(qid)
    label_list = label_dict[qid]
    qid = int(qid.split('.')[0])
    if qid not in valid_list:
      continue
    if qid not in rel_dcit:
      rel_dcit[qid] = []
    label_list = list(set(label_list))
    for label in label_list:
      pid = label.split('.')[0]
      rel_dcit[qid].append(pid)

  return rel_dcit




def random_guess_baseline(rel_dict, topk=5, seed=42):
    """
    隨機從所有候選文件中亂猜 topk 答案，不包含 query 本身。
    Args:
        rel_dict: dict of {query_id: [正確答案們]}
        topk: 每個 query 隨機猜 topk 篇文件
        seed: 隨機種子，確保結果可重現
    Returns:
        precision, recall, f1 score
    """
    random.seed(seed)

    # 收集所有出現過的 doc_id
    all_doc_ids = list({pid for pids in rel_dict.values() for pid in pids})
    list_answer_ohe, list_label_ohe = [], []

    for qid, gold in rel_dict.items():
        pool = [pid for pid in all_doc_ids if pid != qid]
        guess = random.sample(pool, topk)
        list_answer_ohe.append(guess)
        list_label_ohe.append([int(pid) for pid in gold])

    return my_classification_report(list_label_ohe, list_answer_ohe)

def record_result(
    model_name: str,
    topk: int,
    trec_file: str,
    f1: float,
    precision: float,
    recall: float,
    notes: str = "",
    csv_path: str = "./Legal Case Retrieval/results/experiment_results.csv"
):
    """
    將單次實驗結果記錄進 CSV 檔案中（若檔案不存在則自動建立）

    Args:
        model_name: 模型名稱
        topk: 預測前幾名
        trec_file: 相似度分數檔案路徑
        f1, precision, recall: 評估指標
        notes: 備註
        csv_path: 結果儲存的檔案路徑（預設為 'experiment_results.csv'）
    """
    # 如果檔案不存在，就建立一個新的 DataFrame
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=["model_name", "topk", "trec_file", "f1", "precision", "recall", "notes"])
    else:
        df = pd.read_csv(csv_path)

    # 新增紀錄
    new_row = {
        "model_name": model_name,
        "topk": topk,
        "trec_file": trec_file,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "notes": notes
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"✅ 已記錄結果到 {csv_path}")


if __name__ == '__main__':
  # === 標準答案設定 ===
  rel_path = 'coliee_dataset/task1/task1_train_labels_2025.json'
  valid_path = 'coliee_dataset/task1/valid_qid.tsv'
  train_path = 'coliee_dataset/task1/train_qid.tsv'
  topk = 5 # topk
  
  # === 讀取正確答案（只取 valid 查詢）===
  valid_rel_dict = rel_file_convert(rel_path, valid_path)
  train_rel_dict = rel_file_convert(rel_path, train_path)
  combine_rel_dict = {"valid": valid_rel_dict, "train": train_rel_dict}

  # === 預測答案設定(需在value相似度分數與排名路徑中有valid或train，如果沒有會跳過) ===
  models = {
    "BM25_valid": 'coliee_dataset/task1/lht_process/BM25/output_bm25_valid.tsv',
    "SAILER_en_finetune_by_CSHaitao_Dot_valid": 'coliee_dataset/task1/lht_process/SAILER/output_SAILER_dot_valid.tsv',
    "SAILER_en_finetune_by_CSHaitao_Cos_valid": 'coliee_dataset/task1/lht_process/SAILER/output_SAILER_cos_valid.tsv',
    "SAILER_en_finetune_by_CSHaitao_Dot_train": 'coliee_dataset/task1/lht_process/SAILER/output_SAILER_dot_train.tsv',
    "SAILER_en_finetune_by_CSHaitao_Cos_train": 'coliee_dataset/task1/lht_process/SAILER/output_SAILER_cos_train.tsv',
    "moderBert_dot_valid": 'coliee_dataset/task1/lht_process/modernBert/output_modernBert_dot_valid.tsv',
    "moderBert_cos_valid": 'coliee_dataset/task1/lht_process/modernBert/output_modernBert_cos_valid.tsv',
    "moderBert_dot_train": 'coliee_dataset/task1/lht_process/modernBert/output_modernBert_dot_train.tsv',
    "moderBert_cos_train": 'coliee_dataset/task1/lht_process/modernBert/output_modernBert_cos_train.tsv'
  }

  for split, rel_dict in combine_rel_dict.items():
    print(f"現在在比對{split}正確答案...")
    for model_name, trec_path in models.items():
      # 如果trec_path沒有valid或train就跳過
      if split not in trec_path:
         print(f"現在在比對{split}正確答案...")
         print(f"{model_name}: {trec_path} 這個路徑中沒有{split}，跳過!")
         continue
      answer_dict = trec_file_convert(trec_path, topk)
      
      list_answer_ohe = [] #預測答案
      list_label_ohe = [] #真實答案
      
      for qid in rel_dict.keys(): #遍歷在驗證答案裡的每個query id
        one_answer = answer_dict[qid] #預測 e.g., ['000123', '000456', '000789']
        one_rel = rel_dict[qid] #真實
        one_answer = [int(pid) for pid in one_answer] #轉型 e.g., [123, 456, 789]
        one_rel = [int(pid) for pid in one_rel] #轉型
        list_answer_ohe.append(one_answer) #list of list e.g., [[123, 456, 789](對應qid=100), [234, 567, 890](對應qid=101), ...]
        list_label_ohe.append(one_rel)
        
      f1, precision, recall = my_classification_report(list_label_ohe, list_answer_ohe)
      print(f"=== {model_name} ===")
      print('Precision',precision)
      print('Recall',recall)
      print('F1_Score',f1)

      # record to csv
      record_result(
          model_name=model_name,
          topk=topk,
          trec_file=trec_path,
          f1=f1,
          precision=precision,
          recall=recall,
          notes="",
          csv_path="./Legal Case Retrieval/results/experiment_results.csv"
      )
  for split, rel_dict in combine_rel_dict.items():
    random_f1, random_precision, random_recall = random_guess_baseline(rel_dict, topk=topk)
    print(f"=== Random Guess ({split})===")
    print('Precision',random_precision)
    print('Recall',random_recall)
    print('F1_Score',random_f1)
    # record to csv
    record_result(
        model_name=f"Random Guess ({split})",
        topk=topk,
        trec_file="",
        f1=random_f1,
        precision=random_precision,
        recall=random_recall,
        notes="",
        csv_path="./Legal Case Retrieval/results/experiment_results.csv"
    )