import json
import random
import os
import argparse

def split_dataset(input_file, train_ratio=0.8, seed=42, output_dir=None):
    """
    將訓練資料分割成訓練集和驗證集
    
    Args:
        input_file: 輸入的標籤檔案路徑，JSON格式
        train_ratio: 訓練集比例，預設0.8
        seed: 隨機種子，預設42
        output_dir: 輸出目錄，預設為input_file所在目錄
    
    Returns:
        train_file: 訓練集檔案路徑
        valid_file: 驗證集檔案路徑
        valid_qid_file: 驗證集qid檔案路徑
    """
    # 設定隨機種子，確保結果可重現
    random.seed(seed)
    
    # 讀取輸入檔案
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 獲取所有qid
    qids = list(data.keys())
    
    # 隨機打亂qid
    random.shuffle(qids)
    
    # 計算訓練集大小
    train_size = int(len(qids) * train_ratio)
    
    # 分割qid列表
    train_qids = qids[:train_size]
    valid_qids = qids[train_size:]
    
    # 根據qid列表創建訓練集和驗證集
    train_data = {qid: data[qid] for qid in train_qids}
    valid_data = {qid: data[qid] for qid in valid_qids}
    
    # 設定輸出路徑
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    
    # 獲取檔案名稱（不含副檔名）
    file_name = os.path.basename(input_file).split('.')[0]
    
    # 設定輸出檔案路徑
    train_file = os.path.join(output_dir, f"{file_name}_train.json")
    valid_file = os.path.join(output_dir, f"{file_name}_valid.json")
    valid_qid_file = os.path.join(output_dir, "valid_qid.tsv")
    
    # 保存訓練集和驗證集
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)
    
    with open(valid_file, 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, indent=4)
    
    # 保存驗證集qid列表
    with open(valid_qid_file, 'w', encoding='utf-8') as f:
        for qid in valid_qids:
            # 移除.txt後綴，只保留數字部分
            qid_num = qid.split('.')[0]
            f.write(f"{qid_num}\n")
    
    print(f"訓練集大小: {len(train_qids)}")
    print(f"驗證集大小: {len(valid_qids)}")
    print(f"訓練集已保存至: {train_file}")
    print(f"驗證集已保存至: {valid_file}")
    print(f"驗證集qid列表已保存至: {valid_qid_file}")
    
    return train_file, valid_file, valid_qid_file

if __name__ == "__main__":
    # 需要修改的參數
    input_file = r"./coliee_dataset/task1/task1_train_labels_2025.json"  # 輸入的標籤檔案路徑
    train_ratio = 0.8  # 訓練集比例
    seed = 42  # 隨機種子
    output_dir = r"./coliee_dataset/task1"  # 輸出目錄，如果為None則使用輸入檔案所在目錄
    
    # 執行分割
    split_dataset(input_file, train_ratio, seed, output_dir) 