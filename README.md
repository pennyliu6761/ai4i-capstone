# 基於故障可預測性分析之銑床兩階段預測性維護決策支援系統

**A Two-Stage Predictive Maintenance Decision Support System for CNC Milling Machines Based on Failure Diagnosability Analysis**

---

| 項目 | 內容 |
|------|------|
| 資料集 | AI4I 2020 Predictive Maintenance Dataset（UCI Machine Learning Repository） |
| 資料規模 | 10,000 筆製程記錄 · 5 種故障類型 · 14 個原始特徵 |
| 研究方法 | EDA 可診斷性分析 → 兩階段決策設計 → ML/DL 模型比較 → SHAP 可解釋性驗證 |
| 開發工具 | Python 3 · XGBoost · LightGBM · TabNet · scikit-learn · SHAP · Streamlit |
| 展示平台 | Streamlit Community Cloud 互動式決策支援系統 |
| 系所 | 工業工程與管理學系 畢業專題 |

---

## 摘要

本研究以 AI4I 2020 合成銑床資料集（10,000 筆）為對象，提出一套「兩階段預測性維護決策支援系統」。研究核心在於：**不同本質的故障應採用不同的處理策略**，而非直接將所有故障混入機器學習模型。

**第一階段**透過探索性資料分析（EDA）量化五種故障的可診斷性。發現 RNF（純隨機故障）的 ROC-AUC 僅 0.6629 且 Recall = 0，無任何可預測訊號；TWF（刀具磨耗故障）在磨耗 200 分鐘後進入條件隨機區，各區間發生率 4–16% 且不穩定。因此，Stage 1 排除 RNF 並保留 TWF 觀察；Stage 2 進一步排除 TWF，僅對三種規則型故障建模。

**第二階段**針對三種具有明確物理邊界的規則型故障（HDF 散熱不良、PWF 功率異常、OSF 過負荷），比較 10 個模型。排除 TWF 和 RNF 後，各模型 MCC 平均提升 **+0.16**，最佳 LightGBM 達到 MCC = 0.9291。SHAP 可解釋性分析確認模型學習到的決策特徵與各故障的物理觸發條件 3/3 完全吻合。

**關鍵字**：預測性維護、故障可診斷性、兩階段決策、SHAP 可解釋性、類別不平衡

---

## 一、研究背景與動機

### 1.1 預測性維護的重要性

在智慧製造環境中，非計畫性設備停機是影響生產效率與成本的主要因素。預測性維護（Predictive Maintenance, PdM）透過即時感測器資料與機器學習，能在故障發生前提供預警，是工業 4.0 的核心議題。

### 1.2 現有研究的不足

回顧以 AI4I 2020 資料集為對象的現有研究，多數工作直接將五種故障混合成一個多分類問題，忽略了一個更根本的問題：

> **並非所有故障類型都適合用機器學習預測。**
> 若不先分析故障的物理本質，直接套用模型，結果中包含「無法消除的隨機誤差（Bayes Error）」，導致評估指標失真，也無法為工廠提供正確的維護策略。

### 1.3 研究目標

1. 透過 EDA 量化五種故障的可診斷性，建立分層決策依據
2. 設計兩階段決策框架，對不同本質故障採用最適合的處理策略
3. 比較 10 個 ML/DL 模型在排除不可預測故障後的效能，並量化改善幅度
4. 透過 SHAP 與 TabNet Sparse Attention 驗證模型決策的物理合理性
5. 開發 Streamlit 互動式系統作為成果展示平台

---

## 二、整體研究架構

![兩階段架構圖](fig_two_stage.png)
*圖 2-1　兩階段預測性維護決策支援系統架構圖*

**完整研究流程：**

```
資料讀取與基本理解（10,000 筆，UCI 官方版本）
       ↓
特徵工程（衍生 4 個領域知識特徵）
       ↓
EDA：TWF & RNF 可預測性分析（Cell 2b、2c）
       ↓
實驗A：故障可診斷性量化（1-vs-Rest XGBoost 5-Fold CV）
       ↓
實驗B：特徵工程消融實驗（Ablation Study）
       ↓
Stage 1：排除 RNF → 5 類分類（9 個模型）
       ↓
Stage 2：排除 RNF + TWF → 4 類分類（10 個模型）
       ↓
實驗C：SHAP 可解釋性分析（XGBoost + TabNet）
       ↓
Streamlit UI 系統整合（7 頁互動展示）
```

---

## 三、資料集介紹

### 3.1 AI4I 2020 概述

AI4I 2020 Predictive Maintenance Dataset 由 Stephan Matzka 發布，收錄於 UCI Machine Learning Repository，模擬 CNC 銑床製程感測器記錄，共 10,000 筆、14 個欄位。

### 3.2 五種故障的物理觸發條件與可診斷性

| 代碼 | 故障名稱 | 觸發條件 | 樣本數 | 本質類型 | AUC（實驗A） |
|------|---------|---------|-------|---------|------------|
| HDF | 散熱不良故障 | 溫差 < 8.6K **且** 轉速 < 1,380 rpm | 115 | 規則型 | **1.0000** ✅ |
| PWF | 功率異常故障 | 功率 < 3,500W **或** > 9,000W | 95 | 規則型 | **0.9992** ✅ |
| OSF | 過負荷故障 | 扭矩×磨耗 > 門檻（L:11,000 / M:12,000 / H:13,000 Nm·min） | 98 | 規則型 | **0.9998** ✅ |
| TWF | 刀具磨耗故障 | 磨耗達 200 min 後條件隨機觸發（4–16%，不穩定） | 46 | 半隨機型 | 0.9637 ⚠️ |
| RNF | 隨機故障 | 每次製程有 0.1% 純隨機故障機率 | 19 | 純隨機型 | 0.6629 ❌ |

---

## 四、EDA：TWF & RNF 可預測性分析

### 4.1 TWF 分析

![TWF 分析](fig_twf_analysis.png)
*圖 4-1　TWF 可預測性分析：Tool wear 分布對比（左）與各磨耗區間發生率（右）*

TWF 故障幾乎全部集中在 Tool wear > 200 min 的區間，但各區間發生率僅 4–16% 且無單調趨勢，屬於條件隨機事件。正確策略：**磨耗 > 200 min 直接發出換刀警示，不需要 ML 預測。**

### 4.2 RNF 分析

![RNF 分析](fig_rnf_analysis.png)
*圖 4-2　RNF 可預測性分析：轉速 vs 扭矩散佈圖（左）與各特徵 violin 分布（右）*

19 筆 RNF 完全隨機散落在正常樣本雲中，各特徵 violin 圖顯示 RNF 與正常分布完全重疊，無任何可預測訊號。**Stage 1 起即排除。**

---

## 五、三項核心實驗

![三實驗完整報告](experiments_ABC_report.png)
*圖 5-1　實驗A（可診斷性）· 實驗B（消融實驗）· 實驗C（SHAP 可解釋性）完整報告*

### 5.1 實驗A：故障可診斷性量化

對每種故障分別進行 1-vs-Rest 二元分類（XGBoost + 5-Fold StratifiedKFold）：

| 故障 | 本質類型 | 樣本數 | AUC | AP | Recall | MCC |
|------|---------|-------|-----|----|--------|-----|
| HDF | 規則型 | 115 | 1.0000 | 0.9966 | 0.9652 | 0.9570 |
| PWF | 規則型 | 95 | 0.9992 | 0.9092 | 0.9684 | 0.8788 |
| OSF | 規則型 | 98 | 0.9998 | 0.9853 | 0.9795 | 0.9256 |
| TWF | 半隨機型 | 46 | 0.9637 | 0.1088 | 0.1733 | 0.1153 |
| RNF | 純隨機型 | 19 | 0.6629 | 0.0051 | 0.0000 | −0.0018 |

> TWF 的 AUC 雖達 0.9637，但 Recall 僅 0.17、MCC 僅 0.12，說明模型幾乎無法抓到 TWF 樣本——這正是「半隨機型」的特徵。

### 5.2 實驗B：特徵工程消融實驗

基準（完整 11 特徵）：MCC = 0.8076

| 移除特徵 | ΔMCC | 結論 |
|---------|------|------|
| 功率（Power） | **−0.0193** | 最重要衍生特徵，對應 PWF |
| 溫差（Temperature difference） | −0.0109 | 對應 HDF |
| 功率×磨耗（Power wear） | −0.0107 | 對應 OSF |
| 溫差/功率（Temperature power） | −0.0005 | 貢獻極小 |
| 轉速（Rotational speed） | −0.0630 | 最重要原始特徵 |
| 只用原始特徵（7個） | −0.0883 | 衍生特徵整體貢獻顯著 |

### 5.3 實驗C：SHAP 可解釋性分析

![SHAP vs TabNet](fig4_shap_tabnet.png)
*圖 5-2　XGBoost SHAP 各故障類別（左）與 XGBoost vs TabNet 全局特徵重要性對比（右）*

| 故障類別 | 物理觸發條件 | SHAP Top-1 特徵 | SHAP 值 | 物理一致性 |
|---------|------------|----------------|--------|----------|
| HDF 散熱不良 | 溫差 < 8.6K + 轉速 < 1,380 rpm | 溫差 | 4.141 | ✅ 一致 |
| PWF 功率異常 | 功率超出 [3,500, 9,000] W | 功率 | 4.011 | ✅ 一致 |
| OSF 過負荷 | 扭矩×磨耗超型別門檻 | 功率×磨耗 | 4.257 | ✅ 一致 |

---

## 六、兩階段模型訓練結果

### 6.1 Stage 1：排除 RNF（5 類分類，9 個模型）

資料：9,981 筆 · 標籤：0=正常 / 1=TWF / 2=HDF / 3=PWF / 4=OSF · SMOTE 平衡後各類 7,736 筆

| 模型 | MCC | F1w | AUC | 訓練時間 |
|------|-----|-----|-----|---------|
| Stacking (XGB+LGBM→RF) | **0.8716** | 0.9923 | 0.9418 | 17.6s |
| LightGBM | 0.8044 | 0.9879 | 0.9953 | 1.7s |
| Random Forest | 0.7837 | 0.9867 | 0.9813 | 27.6s |
| Decision Tree | 0.7716 | 0.9858 | 0.9162 | 0.8s |
| XGBoost | 0.7406 | 0.9833 | 0.9943 | 3.0s |
| Gradient Boosting | 0.7112 | 0.9807 | 0.9944 | 145.3s |
| MLP (upgraded) | 0.6911 | 0.9793 | 0.9923 | 109.8s |
| MLP (original) | 0.5765 | 0.9599 | 0.9913 | 34.5s |
| KNN | 0.5179 | 0.9613 | 0.8760 | 0.1s |

### 6.2 Stage 2：排除 RNF + TWF（4 類分類，10 個模型）

資料：9,936 筆 · 標籤：0=正常 / 1=HDF / 2=PWF / 3=OSF · SMOTE 平衡後各類 7,721 筆

![模型效能比較](fig1_model_comparison.png)
*圖 6-1　Stage 2 十個模型效能比較（MCC / ROC-AUC / F1 加權）*

| 排名 | 模型 | 類別 | MCC | ROC-AUC | F1w | Recall(macro) | 訓練時間 |
|------|------|------|-----|---------|-----|--------------|---------|
| 🥇 1 | LightGBM | ML | **0.9291** | 0.9997 | 0.9960 | 0.9450 | 1.3s |
| 🥈 2 | Gradient Boosting | ML | 0.9247 | 0.9995 | 0.9957 | **0.9725** | 69.9s |
| 🥉 3 | Stacking (XGB+LGBM→RF) | 集成 | 0.9163 | 0.9761 | 0.9953 | 0.8794 | 12.3s |
| 4 | XGBoost | ML | 0.9106 | **0.9998** | 0.9951 | 0.9184 | 3.8s |
| 5 | Decision Tree | ML | 0.8982 | 0.9260 | 0.9944 | 0.8832 | 0.4s |
| 6 | Random Forest | ML | 0.8910 | 0.9992 | 0.9940 | 0.8949 | 7.9s |
| 7 | MLP (upgraded) | DL | 0.7958 | 0.9977 | 0.9871 | 0.9381 | 43.7s |
| 8 | TabNet | DL | 0.7842 | 0.9980 | 0.9873 | 0.8729 | 69.1s |
| 9 | MLP (original) | DL | 0.7500 | 0.9968 | 0.9840 | 0.9108 | 24.2s |
| 10 | KNN | ML | 0.5361 | 0.8584 | 0.9695 | 0.7238 | 0.1s |

### 6.3 兩階段 MCC 提升效果

![深度分析](fig_deep_analysis.png)
*圖 6-2　Stage 1 vs Stage 2 MCC 對比（左）與故障可預測性天花板分析（右）*

| 模型 | Stage 1 MCC | Stage 2 MCC | 提升幅度 |
|------|------------|------------|---------|
| KNN | 0.5179 | 0.5361 | +0.018 |
| Decision Tree | 0.7716 | 0.8982 | +0.127 |
| Random Forest | 0.7837 | 0.8910 | +0.107 |
| Gradient Boosting | 0.7112 | 0.9247 | **+0.214** |
| XGBoost | 0.7406 | 0.9106 | +0.170 |
| LightGBM | 0.8044 | 0.9291 | +0.125 |
| MLP (original) | 0.5765 | 0.7500 | +0.174 |
| MLP (upgraded) | 0.6911 | 0.7958 | +0.105 |
| Stacking | 0.8716 | 0.9163 | +0.045 |
| **平均** | **0.718** | **0.838** | **+0.120** |

### 6.4 混淆矩陣分析

![混淆矩陣](fig3_confusion.png)
*圖 6-3　LightGBM（MCC=0.9291，左）與 MLP upgraded（MCC=0.7958，右）混淆矩陣*

### 6.5 效率 vs 效能

![效率效能](fig2_efficiency.png)
*圖 6-4　訓練效率 vs 預測效能氣泡圖（左）與 Recall(macro) 比較（右）*

**LightGBM 推薦作為部署模型**：MCC 最高（0.9291）、AUC 第二（0.9997）、訓練僅需 1.3 秒，CP 值最高。

---

## 七、深度學習 vs 機器學習：差距成因分析

深度學習（MLP、TabNet）效能低於傳統 ML 樹狀模型，原因由資料特性決定：

| 原因 | 說明 |
|------|------|
| **資料量不足** | Stage 2 共 9,936 筆，DL 通常需要 100,000 筆以上才能充分發揮 |
| **特徵維度低** | 僅 11 個特徵，邊界清晰，樹狀模型天生擅長低維空間切割 |
| **規則型邊界** | HDF/PWF/OSF 的觸發條件本質上是數學規則，XGBoost 幾乎可以直接復現 |

> **重要發現**：排除 TWF 和 RNF 後，DL 的 MCC 提升幅度（平均 +0.15）與 ML 相當，說明**問題定義的正確性，比模型架構選擇更關鍵**。

---

## 八、研究結論

![研究結論摘要](fig5_summary.png)
*圖 8-1　完整研究架構、模型排名與四項核心結論*

### 四項核心研究結論

**C1：故障本質決定預測天花板**
- 規則型故障（HDF/PWF/OSF）：AUC > 0.999，已接近理論極限
- 半隨機型故障（TWF）：AUC = 0.9637，但 Recall 僅 0.17，不可靠
- 純隨機型故障（RNF）：AUC = 0.6629，Recall = 0，無法預測

**C2：排除雜訊後效能顯著提升**
- 兩階段架構使各模型 MCC 平均提升 **+0.12**（Stage 1 → Stage 2）
- ML 樹狀模型在低維規則邊界問題上優於深度學習

**C3：衍生特徵有實質貢獻**
- 「功率」為最重要特徵（ΔMCC = −0.019），對應 PWF 物理觸發條件
- 完整特徵（MCC = 0.808）顯著優於僅原始特徵（MCC = 0.719，差距 −0.088）

**C4：模型學到了領域知識**
- SHAP Top-1 與物理規則 **3/3 一致**（HDF→溫差、PWF→功率、OSF→功率×磨耗）
- XGBoost 與 TabNet 全局特徵重要性排序高度一致，跨架構驗證

---

## 九、Streamlit 互動式系統

**部署網址**：[ai4i-capstone.streamlit.app]([https://ai4i-capstone.streamlit.app](https://ai4i-capstone-5kqvmrpxcozvcmccvvb4ut.streamlit.app/))

系統共 7 個頁面：

| 頁面 | 功能 |
|------|------|
| 🏠 總覽儀表板 | KPI 卡片、兩階段 MCC 進展圖、SHAP 一致性摘要 |
| 🔮 即時預測器 | 輸入感測器值 → Stage 1 規則檢查 → Stage 2 十模型預測 + 投票共識 |
| 📊 模型比較分析 | 7 個分頁：指標表、雷達圖、ROC、PR 曲線、混淆矩陣、特徵重要性、一對一比較 |
| 🔬 資料探索 | SMOTE 前後類別分布、特徵 violin 分布、相關係數熱圖、故障說明 |
| 📡 即時監控看板 | 模擬銑床動態過程（正常→HDF→PWF→OSF），即時預測可視化 |
| 📂 批次預測上傳 | 上傳 CSV → 規則檢查 + ML 預測 → 高風險清單 + 結果下載 |
| 📖 研究方法論 | 完整研究流程、踩坑紀錄、程式碼片段、套件需求 |

---

## 十、專案檔案說明

```
ai4i_complete_code/
├── AI4I_Predictive_Maintenance_Complete.ipynb  # 完整研究 notebook（16 個 Cell）
├── ai4i2020.csv                                # 原始資料集（UCI 官方版本）
│
├── 實驗結果 CSV
│   ├── expA_diagnostics.csv    # 實驗A：5 種故障可診斷性（AUC/AP/Recall/MCC）
│   ├── expB_ablation.csv       # 實驗B：特徵工程消融實驗（ΔMCC）
│   ├── expC_shap.csv           # 實驗C：SHAP Top-3 特徵與物理一致性
│   ├── results_stage1.csv      # Stage 1 九個模型完整指標
│   ├── results_stage2.csv      # Stage 2 九個模型完整指標
│   └── final_results.csv       # Stage 2 含 TabNet 最終排行
│
├── 模型 PKL 檔案
│   ├── state_stage1.pkl        # Stage 1 資料前處理狀態（scaler/SMOTE）
│   ├── state_stage2.pkl        # Stage 2 資料前處理狀態
│   ├── results_stage1.pkl      # Stage 1 模型物件 + 評估結果
│   ├── results_stage2.pkl      # Stage 2 模型物件 + SHAP 結果
│   └── app_bundle.pkl          # Streamlit 專用 bundle（模型+資料+SHAP）
│
└── 圖表 PNG
    ├── fig_twf_analysis.png        # TWF 可預測性分析（分布 + 各區間發生率）
    ├── fig_rnf_analysis.png        # RNF 可預測性分析（散佈圖 + violin）
    ├── experiments_ABC_report.png  # 三實驗完整報告圖
    ├── fig1_model_comparison.png   # Stage 2 模型效能比較（橫條圖）
    ├── fig2_efficiency.png         # 效率 vs 效能氣泡圖
    ├── fig3_confusion.png          # LightGBM vs MLP 混淆矩陣
    ├── fig4_shap_tabnet.png        # SHAP + XGBoost vs TabNet 特徵重要性
    ├── fig5_summary.png            # 研究架構與核心結論摘要
    ├── fig_deep_analysis.png       # 兩階段 MCC 對比 + 天花板分析
    └── fig_two_stage.png           # 兩階段決策流程架構圖
```

---

## 十一、環境設定與執行

### 11.1 套件需求

```bash
pip install xgboost lightgbm imbalanced-learn pytorch-tabnet shap
```

### 11.2 執行 Notebook

在 Google Colab 執行，依序執行 Cell 0 → Cell 12。

| Cell | 內容 |
|------|------|
| 0 | 安裝套件、掛載 Google Drive |
| 1 | 資料準備與特徵工程 |
| 2 | 實驗A：故障可診斷性量化 |
| 2b | TWF 可預測性深度分析（圖表） |
| 2c | RNF 可預測性分析（圖表） |
| 3 | 實驗B：特徵工程消融實驗 |
| 4a/4b | Stage 1 前處理 + 模型訓練 |
| 5a/5b | Stage 2 前處理 + 模型訓練 |
| 6 | 實驗C：SHAP 可解釋性分析 |
| 7 | TabNet 深度學習模型 |
| 8 | 圖表：完整模型比較（5 張） |
| 9 | 圖表：深度分析（兩階段對比 + 天花板） |
| 10 | 圖表：三實驗完整報告圖 |
| 11 | 圖表：兩階段架構流程圖 |
| 12 | 產生 app_bundle.pkl（供 Streamlit 使用） |

### 11.3 啟動 Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 十二、常見問題與注意事項

| 問題 | 說明 | 修正方式 |
|------|------|---------|
| Scaler 在全資料 fit | 測試集資訊洩漏 | 只在訓練集 `fit_transform`，測試集只 `transform` |
| SMOTE 在測試集執行 | 評估結果不真實 | SMOTE 只在訓練集執行 |
| F1 未指定 average | 多分類計算錯誤 | 加上 `average='weighted'` |
| XGBoost 特徵名含括號 | `[` `]` 造成錯誤 | 傳入 numpy array 或重新命名（FEAT_COLS_SAFE） |
| Temperature power 有 inf | Power=0 時除法產生 inf | `np.where(Power!=0, Temp_diff/Power, 0.0)` |
| Stacking 用 LR 當 meta | 不平衡下崩潰 | 改用 `RandomForestClassifier(class_weight='balanced')` |

---

## 參考文獻

1. Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance Applications. *Third International Conference on Artificial Intelligence for Industries (AI4I)*. IEEE.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, pp. 785–794.
3. Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
4. Arik, S. O., & Pfister, T. (2021). TabNet: Attentive interpretable tabular learning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(8), pp. 6679–6687.
5. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
6. Chawla, N. V. et al. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, pp. 321–357.
7. Matthews, B. W. (1975). Comparison of the predicted and observed secondary structure of T4 phage lysozyme. *Biochimica et Biophysica Acta*, 405(2), pp. 442–451.
