import os
import re
from tqdm import tqdm
import multiprocessing
from functools import partial

def process_file(name, input_dir, output_dir):
    with open(f"{input_dir}/{name}", "r", encoding="utf-8") as f:
        txt = f.read()
        if "Summary:" in txt and "no summary" not in txt and "for this document are in preparation." not in txt:
            idx = txt.find("Summary:")
            end = txt.find("- Topic", idx) #從idx位置之後正面數來第一次出現"- Topic"字串的位置
            end2 = txt.rfind("\n", idx, end) #從最後面數來第一次出現換行的位置
            summ=txt[idx+8:end2].strip()
            if summ.count("\n") > 20:
                print(f"摘要換行大於20個: {name}")
                return False
            with open(f"{output_dir}/{name}", "w+", encoding="utf-8") as fp:
                if summ == "":
                    print(f"摘要為空: {name}")
                    return False
                fp.write(summ)
            return True
        else:
            # print(f"沒摘要: {name}")
            return False


if __name__ == "__main__":
    input_dir = "./coliee_dataset/task1/task1_train_files_2025"
    output_dir = "./coliee_dataset/task1/summary"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    names = os.listdir(input_dir)
    
    # 创建进程池，使用所有可用的CPU核心
    num_cores = multiprocessing.cpu_count()
    print(f"使用 {num_cores} 個CPU核心進行並行處理")
    
    # 创建偏函数，固定输入和输出目录参数
    process_func = partial(process_file, input_dir=input_dir, output_dir=output_dir)
    
    # 使用进程池并行处理所有文件
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(process_func, names), total=len(names), desc="处理文件"))

    total_files = len(names)
    summaries_written = sum(results)

    print(f"\n總共處理 {total_files} 篇檔案")
    print(f"成功記錄下來摘要的有 {summaries_written} 篇")
    print(f"沒有記錄摘要的有 {total_files - summaries_written} 篇")
