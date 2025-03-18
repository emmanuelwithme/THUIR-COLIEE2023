import json
import stat
from tqdm import tqdm
import jieba
import re
import os
import numpy as np
from collections import defaultdict

# 輸入輸出路徑定義
raw_path = './coliee_dataset/task1/processed'
output_dir = './coliee_dataset/task1/lht_process/BM25'
query_valid_file = os.path.join(output_dir, 'query_valid.tsv')
query_train_file = os.path.join(output_dir, 'query_train.tsv')
label_path = './coliee_dataset/task1/task1_train_labels_2025.json'
valid_dir = './coliee_dataset/task1'
valid_path = os.path.join(valid_dir, 'valid_qid.tsv')
train_path = os.path.join(valid_dir, 'train_qid.tsv')

# 確保所有目錄存在
os.makedirs(raw_path, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(os.path.dirname(label_path), exist_ok=True)

# 確保所有用戶都有讀寫權限
os.chmod(raw_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
os.chmod(output_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
os.chmod(valid_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
os.chmod(os.path.dirname(label_path), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

# 開始處理
file_dir = os.listdir(raw_path)

# 讀取標籤文件
train_file = []
try:
    label_file = open(label_path, 'r', encoding='utf-8')
    label_dict = json.load(label_file)
    for key in label_dict.keys():
        pid = key.split('.')[0]
        train_file.append(pid)
    label_file.close()
except Exception as e:
    print(f"讀取標籤文件時出錯: {e}")
    label_dict = {}

# 讀取驗證集ID
valid_list = []
try:
    valid_file = open(valid_path, 'r', encoding='utf-8')
    for line in valid_file.readlines():
        line = line.strip().split(' ')
        qid = line[0]
        valid_list.append(qid)
    valid_file.close()
except Exception as e:
    print(f"讀取驗證集ID文件時出錯: {e}")
    # 如果出錯，使用所有文件作為驗證集
    valid_list = [f.split('.')[0] for f in file_dir]

# 創建訓練集ID列表 (所有不在驗證集中的ID)
train_list = [pid for pid in train_file if pid not in valid_list]

# 將訓練集ID保存到文件
with open(train_path, 'w', encoding='utf-8') as f:
    for qid in train_list:
        f.write(f"{qid}\n")
print(f"已成功創建訓練集ID文件: {train_path}")

# 處理驗證集查詢
outfile_valid = open(query_valid_file, 'w', encoding='utf-8')
max_len_valid = 0
print("處理驗證集查詢...")
for a_file in tqdm(file_dir):
    pid = a_file.split('.')[0]
    if pid not in valid_list:
        continue

    path = os.path.join(raw_path, a_file)
    text_ = ""
    
    try:
        with open(path, encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.replace("\n", " ").replace("\t"," ").strip()
                if len(line) != 0:
                    text_ = text_ + line
        
        if len(text_) > max_len_valid:
            max_len_valid = len(text_)
        if len(text_) > 30000:
            text_ = text_[:10000]
        outfile_valid.write(pid + '\t' + text_ + '\n')
    except Exception as e:
        print(f"處理文件 {a_file} 時出錯: {e}")

outfile_valid.close()
print(f"最大驗證集文本長度: {max_len_valid}")
print(f"已成功創建驗證集查詢文件: {query_valid_file}")

# 處理訓練集查詢
outfile_train = open(query_train_file, 'w', encoding='utf-8')
max_len_train = 0
print("處理訓練集查詢...")
for a_file in tqdm(file_dir):
    pid = a_file.split('.')[0]
    if pid not in train_list:
        continue

    path = os.path.join(raw_path, a_file)
    text_ = ""
    
    try:
        with open(path, encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.replace("\n", " ").replace("\t"," ").strip()
                if len(line) != 0:
                    text_ = text_ + line
        
        if len(text_) > max_len_train:
            max_len_train = len(text_)
        if len(text_) > 30000:
            text_ = text_[:10000]
        outfile_train.write(pid + '\t' + text_ + '\n')
    except Exception as e:
        print(f"處理文件 {a_file} 時出錯: {e}")

outfile_train.close()
print(f"最大訓練集文本長度: {max_len_train}")
print(f"已成功創建訓練集查詢文件: {query_train_file}")
print(f"訓練集查詢文件包含 {len(train_list)} 條記錄，驗證集查詢文件包含 {len(valid_list)} 條記錄")