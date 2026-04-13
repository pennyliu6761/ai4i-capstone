# ⚙️ AI4I 兩階段預測性維護決策支援系統

**基於故障可預測性分析之銑床兩階段預測性維護決策支援系統**

> A Two-Stage Predictive Maintenance Decision Support System for CNC Milling Machines Based on Failure Diagnosability Analysis

---

## 📁 專案結構

```
streamlit_app/
├── app.py                  # 主入口 + 7 頁導覽
├── requirements.txt        # 套件需求
├── README.md
├── models/
│   ├── app_bundle.pkl      # 6 個精簡模型 + scaler + 測試資料 + SHAP 結果
│   ├── results.csv         # 第二階段各模型評估指標
│   ├── simulation_data.csv # 200 步監控模擬腳本（含 4 種故障情境）
│   └── batch_test.csv      # 批次預測測試資料（200 筆）
└── pages/
    ├── __init__.py
    ├── loader.py           # 共用資料載入（快取）
    ├── p1_overview.py      # 🏠 總覽儀表板
    ├── p2_predictor.py     # 🔮 即時預測器（整合兩階段邏輯）
    ├── p3_comparison.py    # 📊 模型比較分析（7 分頁）
    ├── p4_explorer.py      # 🔬 資料探索
    ├── p5_methodology.py   # 📖 研究方法論
    ├── p6_monitor.py       # 📡 即時監控看板
    └── p7_batch.py         # 📂 批次預測上傳
```

---

## 🔬 兩階段決策框架

| 階段 | 故障類型 | 處理方式 | 說明 |
|------|---------|---------|------|
| **第一階段** | RNF 隨機故障 | 排除建模 | AUC = 0.503，純隨機不可預測 |
| **第一階段** | TWF 刀具磨耗 | 規則警示 | 磨耗 > 200 min → 直接發出換刀提醒 |
| **第二階段** | HDF 散熱不良 | ML / DL 分類 | AUC = 1.000，規則型故障 |
| **第二階段** | PWF 功率異常 | ML / DL 分類 | AUC = 1.000，規則型故障 |
| **第二階段** | OSF 過負荷 | ML / DL 分類 | AUC = 0.999，規則型故障 |

---

## 📊 七個頁面說明

| 頁面 | 說明 |
|------|------|
| 🏠 **總覽儀表板** | KPI 卡片、三階段 MCC 進展比較長條圖、兩階段架構說明卡、SHAP 結論摘要 |
| 🔮 **即時預測器** | 感測器滑桿 → Stage 1 規則檢查（TWF/HDF/PWF/OSF 即時判斷）→ Stage 2 全模型同時預測 + 信心度 + 投票共識 |
| 📊 **模型比較分析** | 指標比較表、雷達圖、ROC-AUC 曲線、PR 曲線、混淆矩陣、特徵重要性（SHAP vs TabNet）、一對一比較（7 分頁） |
| 🔬 **資料探索** | 類別分佈（SMOTE 前後）、各故障特徵 violin 圖、相關係數熱圖、故障類型說明卡 |
| 📡 **即時監控看板** | 播放 200 步模擬腳本（正常→HDF→PWF→OSF），動態折線圖 + 故障警示橫幅 + 異常事件日誌 |
| 📂 **批次預測上傳** | 上傳 CSV → Stage 1 規則旗標 + Stage 2 全模型預測 → 高風險清單 + 下載結果 |
| 📖 **研究方法論** | 完整 7 步驟流程說明、踩坑紀錄（6 個常見錯誤）、程式碼範例 |

---

## 🏆 第二階段最佳成績（排除 TWF & RNF 後）

| 排名 | 模型 | 類別 | MCC | ROC-AUC | 訓練時間 |
|------|------|------|-----|---------|---------|
| 🥇 | Random Forest | 傳統 ML | **0.9736** | 0.9998 | 7s |
| 🥈 | XGBoost（調參） | 傳統 ML | 0.9683 | **0.9999** | 1s |
| 🥉 | Stacking (XGB+LGBM) | 集成 | 0.9680 | 0.9874 | 11s |
| 4 | MLP（升級版） | 深度學習 | 0.9505 | 0.9992 | 17s |
| 5 | LightGBM | 傳統 ML | 0.9476 | 0.9997 | 2s |
| 6 | TabNet | 深度學習 | ~0.94–0.95* | 0.9819* | 112s |

*\* TabNet Stage 2 趨勢推估值*

> 📈 **原始設定（含 TWF & RNF）平均 MCC ≈ 0.695 → 兩階段設計後 ≈ 0.963，提升 +0.268（+28%）**

---

## 🚀 本地啟動

```bash
# 1. Clone repo
git clone https://github.com/<your-team>/<repo-name>.git
cd <repo-name>/streamlit_app

# 2. 建立虛擬環境
python -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows

# 3. 安裝套件
pip install -r requirements.txt

# 4. 啟動
streamlit run app.py
```

---

## ☁️ 部署到 Streamlit Community Cloud（免費）

1. 將 `streamlit_app/` 推送到 **public** GitHub repo
2. 前往 [share.streamlit.io](https://share.streamlit.io)
3. **New app** → 選擇 repo
4. **Main file path** 設為 `streamlit_app/app.py`
5. **Deploy** → 約 2 分鐘上線

> ⚠️ `app_bundle.pkl` 約 200 MB，需使用 **Git LFS**：
> ```bash
> git lfs install
> git lfs track "*.pkl"
> git add .gitattributes
> git commit -m "track pkl with LFS"
> git push
> ```

---

## 📦 套件需求（requirements.txt）

```
streamlit>=1.32
scikit-learn>=1.4
xgboost>=2.0
lightgbm>=4.0
imbalanced-learn>=0.12
pytorch-tabnet>=4.0
torch>=2.0
shap>=0.44
plotly>=5.18
pandas>=2.0
numpy>=1.26
```

---

## 📋 批次上傳 CSV 格式

| 欄位 | 說明 | 範例 |
|------|------|------|
| `Type` | 機台類型 L / M / H | M |
| `Air temperature [K]` | 空氣溫度 | 298.5 |
| `Process temperature [K]` | 製程溫度 | 309.2 |
| `Rotational speed [rpm]` | 轉速 | 1450 |
| `Torque [Nm]` | 扭矩 | 42.3 |
| `Tool wear [min]` | 刀具磨耗時間 | 120 |

系統自動計算衍生特徵（Power、Temperature difference、Power wear 等），並先執行 Stage 1 規則檢查再進行 ML 預測。

---

## 🔑 資料集

**AI4I 2020 Predictive Maintenance Dataset**
- 來源：[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- 規模：10,000 筆 · 14 個欄位 · 5 種故障類型
- 引用：Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance Applications. *AI4I Conference*. IEEE.

---

## 🏫 系所資訊

工業工程與管理學系 · 畢業專題
