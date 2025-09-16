# ModernBERT 對比式訓練與動態負樣本抽樣（最新流程）

本說明整合並更新了原先的兩份文件：
- ADAPTIVE_SAMPLING_README.md（動態負樣本抽樣）
- EVALUATION_SUMMARY.md（評估整合與指標）

以目前的訓練腳本 `Legal Case Retrieval/modernBert/fine_tune/fine_tune1.py` 為準。

## 目標
- 訓練 ModernBERT 對比式檢索模型，並在訓練過程中使用「模型自己算出的相似度分數」動態挑選最具迷惑性的負樣本。
- 每個 epoch 結束執行「全候選語料」的檢索評估，並以驗證集 Top‑5 F1 作為最佳模型挑選與早停依據，同步寫入 TensorBoard。

## 整體流程
1) Epoch 0 開始前（由 Trainer 內部 `_inner_training_loop` 觸發）：
- 以當前模型在整個 candidate corpus 上計算查詢的相似度分數。
- 將分數寫成 TREC 檔 `similarity_scores_epoch0.tsv`。
- 依 softmax(score / T) 機率，對每個 (query, positive) 抽取固定數量的 negatives（不足時允許 with‑replacement），寫出 `adaptive_negative_epoch0_train.json`，並更新 `train_dataset`。

2) 進入每個 epoch 開始（由 Callback 觸發）：
- 已註冊 `AdaptiveNegativeSamplingCallback`，在每個 epoch 開始時依 `update_frequency` 決定是否更新。
- 若需更新：重算相似度 → 輸出 `similarity_scores_epoch{E}.tsv` → 依分數抽樣 → 輸出 `adaptive_negative_epoch{E}_train.json` → 更新 `train_dataset`，供本輪訓練使用。

3) 每個 epoch 結束（eval loop）：
- `compute_metrics` 執行「全候選語料」的檢索評估，對 train/valid 分別輸出：
  - `similarity_scores_epoch{E}_eval_train.tsv`
  - `similarity_scores_epoch{E}_eval_valid.tsv`
- 以驗證集 Top‑5 F1 作為 `eval_global_f1` 回傳給 Trainer：
  - 最佳模型挑選依據：`metric_for_best_model="eval_global_f1"`、`greater_is_better=True`
  - 早停監控指標：`eval_global_f1`
- 另外把 Top‑5 F1 / Precision / Recall（train 與 valid 共 6 個值）寫入 TensorBoard（見下）。

## 輸出檔命名與用途
- 訓練用（epoch 開始前）：
  - `similarity_scores_epoch{E}.tsv`：本輪抽樣機率的依據（模型算出的相似度與排名）。
  - `adaptive_negative_epoch{E}_train.json`：本輪訓練使用的對比式資料（query、positive、固定數量的 negatives）。
- 評估用（epoch 結束）：
  - `similarity_scores_epoch{E}_eval_train.tsv`：評估 train 查詢的全語料檢索分數。
  - `similarity_scores_epoch{E}_eval_valid.tsv`：評估 valid 查詢的全語料檢索分數（用於 `eval_global_f1`）。
- 預設輸出路徑：`./coliee_dataset/task1/lht_process/modernBert/finetune_data`

## 負樣本抽樣細節（Adaptive Negative Sampling）
- 相似度來源：模型輸出的 L2 正規化向量之內積（≈ cosine），對每個 query 與全 corpus 計算。
- 機率化：對「非正樣本且非查詢自身」的候選，以 `softmax(score / temperature)` 產生機率分佈：
  - `temperature`（預設 1.0）越小 → 更聚焦高分（更難）的負樣本；越大 → 分佈更平坦（更多樣性）。
- 固定負樣本數：若候選數 < 需求數，改用 with‑replacement 抽樣，確保每筆樣本 negatives 數量一致。
- 自我排除：不會將查詢自身（qid）加入負樣本。

## 檢索評估與最佳模型/早停
- `compute_metrics` 於每個 eval epoch 同時計算 train 與 valid 在「全候選語料」的 Top‑5 指標：
  - F1 / Precision / Recall（Top‑5）
  - 以 valid F1 作為 `eval_global_f1`
- Trainer 設定：
  - `metric_for_best_model="eval_global_f1"`、`greater_is_better=True`、`load_best_model_at_end=True`
  - `EarlyStoppingCallback(early_stopping_patience=5)`（或依需求調整）
- TensorBoard 指標（每個 eval epoch 寫一次）：
  - `retrieval/train_top5_f1`
  - `retrieval/train_top5_precision`
  - `retrieval/train_top5_recall`
  - `retrieval/valid_top5_f1`
  - `retrieval/valid_top5_precision`
  - `retrieval/valid_top5_recall`

## QUICK_TEST（快速測試模式）
- 切換旗標：`QUICK_TEST = True/False`
- 控制規模的環境變數（僅在 QUICK_TEST=True 時生效）：
  - `QT_CAND_K`：候選檔案上限（預設 10）
  - `QT_QUERY_K`：query 數量上限（預設 5）
- 影響範圍：
  - 訓練前/抽樣/評估所用的候選與查詢集合會被子集化；eval loop 也只跑部分 `eval_dataset`（以縮短每個 epoch 的驗證時間）。
- 正式訓練：將 `QUICK_TEST=False` 即不受此模式影響。

## 重要實作與穩定性
- 只保留「單一責任源」更新負樣本：
  - Epoch 0：由 Trainer 內 `_inner_training_loop` 先更新一次。
  - 後續 epoch：僅由 `AdaptiveNegativeSamplingCallback.on_epoch_begin()` 依 `update_frequency` 觸發，避免重複更新。
- ID 處理：
  - `load_query_ids` 以字串保留前導零；讀檔時同時嘗試原字串與 `zfill(6)`。
  - 相似度 map 的 qid 也使用字串，與標註 JSON 的鍵一致。

## 執行重點
1. 準備資料夾與標註（預設路徑在 `coliee_dataset/task1/...`）。
2. 直接執行 `fine_tune1.py` 進行訓練。
3. 於 `./modernBERT_contrastive_adaptive/tb` 查看 TensorBoard：
   ```bash
   tensorboard --logdir ./modernBERT_contrastive_adaptive/tb
   ```

## 常見檔案清單（示例）
```
coliee_dataset/task1/lht_process/modernBert/finetune_data/
├── similarity_scores_epoch0.tsv
├── adaptive_negative_epoch0_train.json
├── similarity_scores_epoch1.tsv
├── adaptive_negative_epoch1_train.json
├── similarity_scores_epoch1_eval_train.tsv
├── similarity_scores_epoch1_eval_valid.tsv
├── similarity_scores_epoch2.tsv
├── adaptive_negative_epoch2_train.json
├── similarity_scores_epoch2_eval_train.tsv
├── similarity_scores_epoch2_eval_valid.tsv
└── ...
```

## 小貼士
- 大規模運算成本高：若要加速測試，使用 QUICK_TEST 並酌量提高 `QT_CAND_K` / `QT_QUERY_K`。
- 若遇到某輪抽不到樣本：請增大候選或調整 `temperature` / `QT_*`，目前程式會依模型分數抽樣且固定落檔（不再回退 BM25）。
- 若要每 N 個 epoch 才更新一次：設定 `update_frequency=N`。

如需更多細節，請直接參閱 `Legal Case Retrieval/modernBert/fine_tune/fine_tune1.py` 之實作。

## 檔案產出時序（舉例）
- epoch 0 開始前 → `similarity_scores_epoch0.tsv`、`adaptive_negative_epoch0_train.json`
- epoch 0 結束後（eval）→ `similarity_scores_epoch1_eval_{train/valid}.tsv`
- epoch 1 開始前 → `similarity_scores_epoch1.tsv`、`adaptive_negative_epoch1_train.json`
- epoch 1 結束後（eval）→ `similarity_scores_epoch2_eval_{train/valid}.tsv`
- epoch 2 開始前 → `similarity_scores_epoch2.tsv`、`adaptive_negative_epoch2_train.json`
- epoch 2 結束後（eval）→ `similarity_scores_epoch3_eval_{train/valid}.tsv`
- 若訓練未開始 epoch 3，則不會有 `similarity_scores_epoch3.tsv` 與 `adaptive_negative_epoch3_train.json`。
