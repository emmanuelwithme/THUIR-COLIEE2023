import argparse
import json
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.task1_paths import get_task1_dir, get_task1_root, get_task1_year

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count label statistics for COLIEE Task1."
    )
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        help="Task1 year (e.g., 2025 or 2026). Default uses COLIEE_TASK1_YEAR/.env.",
    )
    parser.add_argument(
        "--task1-root",
        type=str,
        default=None,
        help="Task1 root directory. Default uses COLIEE_TASK1_ROOT/.env.",
    )
    return parser.parse_args()


def resolve_task1_dir(year: str | None, task1_root: str | None) -> tuple[Path, str]:
    resolved_year = (year or get_task1_year()).strip()
    if task1_root:
        root = Path(task1_root).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        resolved_dir = (root / resolved_year).resolve()
    else:
        if year:
            root = Path(get_task1_root())
            resolved_dir = (root / resolved_year).resolve()
        else:
            resolved_dir = Path(get_task1_dir())
    return resolved_dir, resolved_year


args = parse_args()
TASK1_DIR, TASK1_YEAR = resolve_task1_dir(args.year, args.task1_root)

# 文件路徑
label_path = TASK1_DIR / f"task1_train_labels_{TASK1_YEAR}.json"

# 檢查文件是否存在
if not os.path.exists(label_path):
    print(f"錯誤: 文件 {label_path} 不存在!")
    exit(1)

try:
    # 讀取標籤文件
    with open(label_path, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    
    # 計算query和documents數量
    query_count = len(label_data)
    
    # 計算所有相關文檔的總數
    total_docs = 0
    for query, docs in label_data.items():
        total_docs += len(docs)
    
    # 計算每個query平均有多少相關文檔
    avg_docs = total_docs / query_count if query_count > 0 else 0
    
    # 找出最多和最少相關文檔的query
    max_docs = 0
    min_docs = float('inf')
    max_query = ""
    min_query = ""
    
    for query, docs in label_data.items():
        if len(docs) > max_docs:
            max_docs = len(docs)
            max_query = query
        if len(docs) < min_docs:
            min_docs = len(docs)
            min_query = query
    
    # 輸出結果
    print(f"Task1 年份: {TASK1_YEAR}")
    print(f"標籤文件: {label_path}")
    print(f"查詢(Query)總數: {query_count}")
    print(f"相關文檔總數: {total_docs}")
    print(f"每個查詢平均相關文檔數: {avg_docs:.2f}")
    print(f"最多相關文檔的查詢: {max_query} (共 {max_docs} 個文檔)")
    print(f"最少相關文檔的查詢: {min_query} (共 {min_docs} 個文檔)")
    
    # 分析相關文檔數量分佈
    distribution = {}
    for query, docs in label_data.items():
        doc_count = len(docs)
        if doc_count not in distribution:
            distribution[doc_count] = 0
        distribution[doc_count] += 1
    
    print("\n相關文檔數量分佈:")
    for doc_count in sorted(distribution.keys()):
        print(f"{doc_count}個相關文檔的查詢數量: {distribution[doc_count]}")
    
except Exception as e:
    print(f"處理文件時出錯: {e}") 
