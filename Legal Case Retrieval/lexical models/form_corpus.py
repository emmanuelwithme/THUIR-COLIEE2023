import json
import os
import re
from tqdm import tqdm
import stat

# 定義路徑
raw_path = './coliee_dataset/task1/processed'
file_dir = os.listdir(raw_path)

# 定義目標資料夾
output_dir = './coliee_dataset/task1/lht_process/BM25/corpus'

# 確保目標資料夾存在，如果不存在則自動創建
os.makedirs(output_dir, exist_ok=True)
# 設置資料夾的權限，確保所有用戶都有讀寫權限
os.chmod(output_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
print(f"✅ 目標資料夾 {output_dir} 已建立並設定適當權限")


outfile = open(f'{output_dir}/corpus.json','w', encoding='utf-8')  

# for a_file in file_dir:
#     pid = a_file.split('.')[0]
#     path = f'{raw_path}/{a_file}'
#     text_ = ''
    
#     with open(path, encoding='utf-8') as fin:
#         lines = fin.readlines()
#         for line in lines:
#             line = line.replace("\n", "")
#             text_ = text_ + line
  
#     save_dict = {}
#     save_dict['id'] = pid
#     save_dict['contents'] = text_
#     # print(save_dict)
#     outline = json.dumps(save_dict,ensure_ascii=False)+'\n'
#     outfile.write(outline)
#     # break
save_dict = {}
for a_file in tqdm(file_dir):
    pid = a_file.split('.')[0]
    path = f'{raw_path}/{a_file}'
    text_ = ''
    
    with open(path, encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.replace("\n", "")
            text_ = text_ + line

    save_dict = {}
    save_dict['id'] = pid
    save_dict['contents'] = text_
    # print(save_dict)
    outline = json.dumps(save_dict,ensure_ascii=False)+'\n'
    outfile.write(outline)

    # save_dict[str(pid)] = text_
    # save_dict['contents'] = text_
    # print(save_dict)
    # outline = json.dumps(save_dict,ensure_ascii=False)+'\n'
    # outfile.write(outline)
    # # break
# with open('./coliee_dataset/task1/corpus_all.json','w',encoding='utf-8') as fp:
#     json.dump(save_dict,fp,ensure_ascii=False)
print(f"✅ JSON 檔案已成功寫入至 {outfile.name}")