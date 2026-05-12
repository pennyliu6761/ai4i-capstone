# AI4I 兩階段預測性維護系統（v6）

工業工程與管理學系畢業專題

## 系統架構

**兩階段決策框架：**
- Stage 1：排除 RNF，保留 TWF，5 類分類（9 個模型），最佳 Stacking MCC=0.8716
- Stage 2：再排除 TWF，4 類分類（10 個模型），最佳 LightGBM MCC=0.9291

**三項核心實驗：**
- 實驗A：故障可診斷性量化（1-vs-Rest XGBoost）
- 實驗B：特徵工程消融實驗
- 實驗C：SHAP 可解釋性分析

## 啟動方式

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 檔案說明

```
streamlit_app/
├── app.py                  # 主程式（側欄導覽）
├── requirements.txt
├── models/
│   ├── app_bundle.pkl      # 模型 + 資料（從 Colab notebook 產生）
│   ├── simulation_data.csv # 監控看板模擬資料
│   └── batch_test.csv      # 批次預測測試資料
└── pages/
    ├── loader.py           # 共用常數與資料載入
    ├── p1_overview.py      # 總覽儀表板
    ├── p2_predictor.py     # 即時預測器
    ├── p3_comparison.py    # 模型比較分析
    ├── p4_explorer.py      # 資料探索
    ├── p5_methodology.py   # 研究方法論
    ├── p6_monitor.py       # 即時監控看板
    └── p7_batch.py         # 批次預測上傳
```

## 注意事項

`app_bundle.pkl` 需從 Colab notebook 的 Cell 12 重新產生後，放置於 `models/` 目錄下。
