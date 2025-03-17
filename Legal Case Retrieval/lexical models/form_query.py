import json
import stat
from tqdm import tqdm
import jieba
import re
import os
import numpy as np
from collections import defaultdict

# 2輸入(task1_train_labels_2025.json, valid_qid.tsv)，1輸出(query_valid.tsv)
raw_path = './coliee_dataset/task1/processed'
output_dir = './coliee_dataset/task1/lht_process/BM25'
query_file = os.path.join(output_dir, 'query_valid.tsv')
label_path = './coliee_dataset/task1/task1_train_labels_2025.json'
valid_dir = './coliee_dataset/task1'
valid_path = os.path.join(valid_dir, 'valid_qid.tsv')

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
outfile = open(query_file, 'w', encoding='utf-8')  # 明確指定使用UTF-8編碼

# 讀取標籤文件
train_file = []
try:
    label_file = open(label_path, 'r', encoding='utf-8')  # 明確指定使用UTF-8編碼
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
    valid_file = open(valid_path, 'r', encoding='utf-8')  # 明確指定使用UTF-8編碼
    for line in valid_file.readlines():
        line = line.strip().split(' ')
        qid = line[0]
        valid_list.append(qid)
    valid_file.close()
except Exception as e:
    print(f"讀取驗證集ID文件時出錯: {e}")
    # 如果出錯，使用所有文件作為驗證集
    valid_list = [f.split('.')[0] for f in file_dir]

# print(train_file)
# 處理每個文件
max_len = 0
for a_file in file_dir:
    pid = a_file.split('.')[0]
    if pid not in valid_list:
        continue

    path = os.path.join(raw_path, a_file)
    text_ = ""
    
    try:
        with open(path, encoding='utf-8') as fin:  # 明確指定使用UTF-8編碼
            lines = fin.readlines()
            for line in lines:
                line = line.replace("\n", " ").replace("\t"," ").strip()
                if len(line) != 0:
                    text_ = text_ + line
        
        if len(text_) > max_len:
            max_len = len(text_)
            print(f"目前發現的最大文本長度: {max_len} 字元")
        if len(text_) > 30000:
            text_ = text_[:10000]
        outfile.write(pid + '\t' + text_ + '\n')
    except Exception as e:
        print(f"處理文件 {a_file} 時出錯: {e}")

outfile.close()
print(f"最大文本長度: {max_len}")
print(f"已成功創建查詢文件: {query_file}")