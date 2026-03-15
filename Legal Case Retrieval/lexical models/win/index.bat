@REM 使用 Windows 建立索引
@echo off
chcp 65001 >nul  & REM 設定 CMD 為 UTF-8，避免亂碼

set "REPO_ROOT=%~dp0..\..\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"
set "COLIEE_TASK1_YEAR=2025"

if exist "%REPO_ROOT%\.env" (
  for /f "usebackq tokens=1,* delims==" %%A in ("%REPO_ROOT%\.env") do (
    if /I "%%A"=="COLIEE_TASK1_YEAR" set "COLIEE_TASK1_YEAR=%%B"
  )
)

set "TASK1_DIR=%REPO_ROOT%\coliee_dataset\task1\%COLIEE_TASK1_YEAR%"

@REM 切換到 Miniconda 環境
call C:\ProgramData\Miniconda3\condabin\conda_hook.bat
call conda activate pyserini

@REM 取得 CPU 執行緒數量
for /f "tokens=2 delims==" %%T in ('wmic cpu get NumberOfLogicalProcessors /value ^| findstr NumberOfLogicalProcessors') do set threads=%%T

echo 正在使用 %threads% 個執行緒進行索引建置...

@REM 建立索引
python -m pyserini.index.lucene ^
  --collection JsonCollection ^
  --input "%TASK1_DIR%\lht_process\BM25\corpus" ^
  --index "%TASK1_DIR%\lht_process\BM25\index" ^
  --generator DefaultLuceneDocumentGenerator ^
  --threads %threads% ^
  --storePositions --storeDocvectors --storeRaw

pause
