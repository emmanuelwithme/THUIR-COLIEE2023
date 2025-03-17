@REM 使用 Windows 進行檢索查詢
@echo off
chcp 65001 >nul  & REM 設定 CMD 為 UTF-8，避免亂碼

@REM 切換到 Miniconda 環境
call C:\ProgramData\Miniconda3\condabin\conda_hook.bat
call conda activate pyserini

@REM 取得 CPU 執行緒數量
for /f "tokens=2 delims==" %%T in ('wmic cpu get NumberOfLogicalProcessors /value ^| findstr NumberOfLogicalProcessors') do set threads=%%T

echo 正在使用 %threads% 個執行緒進行檢索...

@REM 進行檢索查詢
python -m pyserini.search.lucene ^
  --index "C:\THUIR-COLIEE2023\coliee_dataset\task1\lht_process\BM25\index" ^
  --topics "C:\THUIR-COLIEE2023\coliee_dataset\task1\lht_process\BM25\query_valid.tsv" ^
  --output "C:\THUIR-COLIEE2023\coliee_dataset\task1\lht_process\BM25\output_bm25_valid.tsv" ^
  --bm25 ^
  --k1 3 ^
  --b 1 ^
  --hits 4451 ^
  --threads %threads% ^
  --batch-size 16

pause
