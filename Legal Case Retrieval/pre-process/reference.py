import os
import re
from langdetect import detect
from langdetect import detect_langs
from langdetect import DetectorFactory
import nltk
from tqdm import tqdm
import multiprocessing
from functools import partial
# nltk.download('punkt') # 使用 pickle 格式來存儲 tokenizer 模型，舊版，不安全，已棄用
nltk.download('punkt_tab') # nltk 3.8.2 或更新
DetectorFactory.seed = 0

def is_sentence(s):
    return s == "" or s.strip().endswith(('.', ':', ';'))

def remove(match):
    result = match.group()
    return result.replace("[", "").replace("]", "").replace(" ", "")

def remove2(match):
    result = match.group()
    return result.replace("[", "").replace("]", "")

def rep(match):
    result = match.group()
    return result.replace("[", "{").replace("]", "}")

def rep2(match):
    result = match.group()
    return result.replace("{}", "[").replace("}", "]")

def process_file(name, input_dir, processed_dir, output_dir):
    last_lang = "en"
    
    with open(f"{input_dir}/{name}", "r", encoding="utf-8") as f:
        t = f.read()
        idx_ = t.find("[1]")
        if idx_ != -1:
            t = t[idx_:]
        t = nltk.sent_tokenize(t)
        result = []
        for i, sentence in enumerate(t):
            if ("FRAGMENT_SUPPRESSED" in sentence or 'REFERENCE_SUPPRESSED' in sentence or 'CITATION_SUPPRESSED' in sentence):
                position = sentence.find("FRAGMENT_SUPPRESSED")
                if position == -1:
                    position = sentence.find("REFERENCE_SUPPRESSED")
                if position == -1:
                    position = sentence.find("CITATION_SUPPRESSED")
                length = len(sentence)
                if position < length / 2:
                    if i > 0 and t[i-1] not in result:
                        result.append(t[i-1])
                    if sentence not in result:
                        result.append(sentence)
                else:
                    if sentence not in result:
                        result.append(sentence)
                    if i < len(t) - 1 and t[i+1] not in result:
                        result.append(t[i+1])
        t = "".join(result)
        lines = t.splitlines()
        lines = [line.strip() for line in lines]
        sentence_list = []
        flag = True
        
        for l in lines:
            l1 = l.replace("<FRAGMENT_SUPPRESSED>", "").replace("FRAGMENT_SUPPRESSED", "").strip()
            l2 = re.sub(r'\[\d{1,3}\]', "", l1).strip()
            if (
                (len(l2) == 1 or
                    (
                        l2 != ""
                        and l2[0] != "("
                        and len(l2) > 1
                        and l2[1] != ")"
                        and not l2[0].isdigit()
                    ))
                and sentence_list
                and not is_sentence(sentence_list[-1])
            ):
                sentence_list[-1] += f" {l2}"
            else:
                sentence_list.append(l2)
    txt = "\n".join(sentence_list)

    txt = re.sub(r"\. *(\. *)+", "", txt)
    txt = re.sub(r"[A-Z]*_SUPPRESSED", "", txt)
    
    need_to_removed = ["[translation]", "[Translation]", "[sic]", "[ sic ]", "[Emphasis added.]",
                       "[emphasis added]", 
                       "[End of document]", "*", "[  ]", "[]", "[ ]",
                        "[DATE_SUPPRESSED]", "[TRANSLATION]", 
                       "[English language version follows French language version]", 
                       "[La version anglaise vient à la suite de la version française]", 
                       "[Diagram omitted - see printed version]", 
                       "[French language version follows English language version]",
                       "[La version française vient à la suite de la version anglaise]", 
                       "[Traduction]"]
    for token in need_to_removed:
        txt = txt.replace(token, "")


    txt = re.sub(r"\[[A-Z][A-Z]+\]", rep, txt)
    txt = re.sub(r"[^a-zA-Z]\[[b-zB-Z]\] ", remove, txt)
    txt = re.sub(r"\[[a-zA-Z][a-zA-Z \.']*\]", remove2, txt)
    txt = re.sub(r"\{[A-Z][A-Z]+\}", rep2, txt)
    txt = re.sub(r"\n\n+", "\n\n", txt)
    txt = re.sub(r"\.\.+", ".", txt)
    txt = re.sub(r"\n\.\n", "\n\n", txt)
    
    new_lines = txt.split("\n")
    for i in range(len(new_lines)):
        if len(new_lines[i]) > 0:
            try:
                lang = detect(new_lines[i])
            except:
                lang = last_lang  # 在異常情況下為 lang 賦值
                if last_lang == "fr":
                    new_lines[i] = ""
                   
            if lang == "fr":
                last_lang = "fr"
                new_lines[i] = ""
            elif lang != "en":
                if last_lang == "fr":
                    new_lines[i] = ""
            else:
                last_lang = "en"
    
    txt = "\n".join(new_lines)     
    txt = re.sub(r"\n\n+", "\n\n", txt)
    words = nltk.word_tokenize(txt)
    now_len = len(words)
    if len(words) < 512:
        try:
            with open(f"{processed_dir}/{name}", "r", encoding="utf-8") as fd:
                new_txt = fd.read()
                slt = nltk.sent_tokenize(new_txt)
                ii = 0  # 初始化 ii 變數
                for ii, sl in enumerate(slt):
                    new_len_words = len(sl.split(' '))
                    now_len += new_len_words
                    if now_len > 512:
                        break
                txt = "".join(slt[:ii]) + txt
        except FileNotFoundError:
            print(f"警告：找不到文件 {processed_dir}/{name}")
    
    with open(f"{output_dir}/{name}", "w+", encoding="utf-8") as f:
        f.write(txt)

if __name__ == "__main__":
    input_dir = "./coliee_dataset/task1/task1_train_files_2025"
    processed_dir = "./coliee_dataset/task1/processed"
    output_dir = "./coliee_dataset/task1/processed_new"
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    names = os.listdir(input_dir)
    
    # 創建進程池，使用所有可用的CPU核心
    num_cores = multiprocessing.cpu_count()
    print(f"使用 {num_cores} 個CPU核心進行並行處理")
    
    # 創建偏函數，固定輸入和輸出目錄參數
    process_func = partial(process_file, 
                          input_dir=input_dir, 
                          processed_dir=processed_dir, 
                          output_dir=output_dir)
    
    # 使用進程池並行處理所有文件
    with multiprocessing.Pool(processes=num_cores) as pool:
        list(tqdm(pool.imap(process_func, names), total=len(names), desc="處理文件"))