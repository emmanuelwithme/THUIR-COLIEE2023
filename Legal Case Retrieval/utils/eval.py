from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.metrics import (
    classification_report as my_classification_report,
    random_guess_baseline,
    rel_file_to_dict as rel_file_convert,
    trec_file_to_dict as trec_file_convert,
)
from lcr.results import record_result


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
