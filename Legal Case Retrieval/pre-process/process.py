import os
import re
from langdetect import detect
from langdetect import detect_langs
from langdetect import DetectorFactory
from tqdm import tqdm
import multiprocessing
from functools import partial
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

def process_file(name, input_dir, summary_dir, output_dir, have_sum):
    last_lang = "en"
    
    with open(f"{input_dir}/{name}", "r", encoding="utf-8") as f:
        t = f.read()
        idx_ = t.find("[1]")
        if idx_ != -1:
            t = t[idx_:]
        lines = t.splitlines()
        lines = [line.strip() for line in lines]
        sentence_list = []
        flag = True
        
        for l in lines:
            if flag and (
                "<FRAGMENT_SUPPRESSED>" in l
                or " FRAGMENT_SUPPRESSED" in l
                or l == ""
            ):
                continue
            flag = False
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
    
    if "Summary:" not in txt and name in have_sum:
        with open(f"{summary_dir}/{name}", "r", encoding="utf-8") as f:
            sum_ = f.read()
            txt = f"Summary:\n{sum_}\n{txt}"
    with open(f"{output_dir}/{name}", "w+", encoding="utf-8") as f:
        f.write(txt)

if __name__ == "__main__":
    input_dir = "./coliee_dataset/task1/task1_train_files_2025"
    summary_dir = "./coliee_dataset/task1/summary"
    output_dir = "./coliee_dataset/task1/processed"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    names = os.listdir(input_dir)
    have_sum = os.listdir(summary_dir)
    
    # 创建进程池，使用所有可用的CPU核心
    num_cores = multiprocessing.cpu_count()
    print(f"使用 {num_cores} 个CPU核心进行并行处理")
    
    # 创建偏函数，固定输入和输出目录参数
    process_func = partial(process_file, 
                          input_dir=input_dir, 
                          summary_dir=summary_dir, 
                          output_dir=output_dir, 
                          have_sum=have_sum)
    
    # 使用进程池并行处理所有文件
    with multiprocessing.Pool(processes=num_cores) as pool:
        list(tqdm(pool.imap(process_func, names), total=len(names), desc="处理文件"))