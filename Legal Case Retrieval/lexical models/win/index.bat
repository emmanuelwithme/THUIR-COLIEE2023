@REM 使用 Windows 建立索引
@echo off
chcp 65001 >nul  & REM 設定 CMD 為 UTF-8，避免亂碼

@REM 切換到 Miniconda 環境
call C:\ProgramData\Miniconda3\condabin\conda_hook.bat
call conda activate pyserini

@REM 取得 CPU 執行緒數量
for /f "tokens=2 delims==" %%T in ('wmic cpu get NumberOfLogicalProcessors /value ^| findstr NumberOfLogicalProcessors') do set threads=%%T

echo 正在使用 %threads% 個執行緒進行索引建置...

@REM 建立索引
python -m pyserini.index.lucene ^
  --collection JsonCollection ^
  --input "C:\THUIR-COLIEE2023\coliee_dataset\task1\lht_process\BM25\corpus" ^
  --index "C:\THUIR-COLIEE2023\coliee_dataset\task1\lht_process\BM25\index" ^
  --generator DefaultLuceneDocumentGenerator ^
  --threads %threads% ^
  --storePositions --storeDocvectors --storeRaw

pause
