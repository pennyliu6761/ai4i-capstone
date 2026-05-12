"""p5_methodology.py — 研究方法論（v6）"""
import streamlit as st
from pages.loader import sec

def show():
    st.markdown("# 📖 研究方法論")
    st.markdown("<p style='color:#8888aa;margin-top:-.4rem'>完整研究流程 · 關鍵設計決策 · 踩坑與修正紀錄</p>",
                unsafe_allow_html=True)

    sec("🗓️ 完整研究流程")

    steps = [
        ("01", "資料理解與 EDA", "#7bf7c8",
         "讀入 AI4I 2020 CSV（UCI 官方版本），確認 10,000 筆、14 欄位、無缺值。\n"
         "各故障樣本數：TWF=46、HDF=115、PWF=95、OSF=98、RNF=19，正常佔 96.61%，故障類別嚴重不平衡。\n"
         "使用相關係數熱圖、PairPlot 散點圖進行初步探索。",
         None),
        ("02", "特徵工程", "#7bb4f7",
         "根據各故障的物理觸發條件，衍生 4 個領域知識特徵：\n"
         "Power = 轉速×扭矩（對應 PWF）\n"
         "Power wear = Power×磨耗（對應 OSF）\n"
         "Temperature difference = 製程溫度−空氣溫度（對應 HDF）\n"
         "Temperature power = 溫差/Power（Power=0 時設為 0，避免 inf）",
         """df['Power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']
df['Power wear'] = df['Power'] * df['Tool wear [min]']
df['Temperature difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['Temperature power'] = np.where(
    df['Power'] != 0,
    df['Temperature difference'] / df['Power'], 0.0)  # 避免 inf"""),
        ("03", "故障可診斷性量化（實驗A，1-vs-Rest）", "#f7c47b",
         "對每種故障分別做 1-vs-Rest 二元分類（XGBoost + 5-Fold CV）。\n"
         "實際結果（UCI 版本）：\n"
         "HDF AUC=1.0000（規則型）\n"
         "PWF AUC=0.9992（規則型）\n"
         "OSF AUC=0.9998（規則型）\n"
         "TWF AUC=0.9637（半隨機型，Recall 僅 0.17）\n"
         "RNF AUC=0.6629（純隨機型，Recall=0）\n"
         "→ 確立研究設計依據：RNF 排除、TWF Stage 1 保留 Stage 2 排除、HDF/PWF/OSF 建模。",
         None),
        ("04", "TWF & RNF 可預測性分析", "#f87171",
         "TWF：故障幾乎全部集中在 Tool wear > 200 min 的區間，"
         "但各區間發生率僅 4-16% 且不穩定，屬於條件隨機事件。\n"
         "RNF：在轉速 vs 扭矩散佈圖中，19 筆 RNF 完全隨機散落在正常樣本雲中，"
         "各特徵的 violin 圖顯示 RNF 與正常分布完全重疊，無任何可預測訊號。\n"
         "正確策略：磨耗 > 200 min 發出換刀警示；RNF 從 Stage 1 起排除。",
         """# TWF 各區間發生率
for lo in range(200, 260, 10):
    sub = df[(df['Tool wear [min]'] >= lo) & (df['Tool wear [min]'] < lo+10)]
    print(f"{lo}-{lo+10} min: {sub['TWF'].sum()}/{len(sub)} ({sub['TWF'].mean():.1%})")
# 200-210: 17/407 (4.2%)  210-220: 12/253 (4.7%)
# 220-230:  9/96  (9.4%)  230-240:  5/31 (16.1%)"""),
        ("05", "資料前處理（兩階段）", "#c47bf7",
         "Stage 1：排除 RNF（19 筆），保留 TWF，共 9,981 筆，5 類標籤（正常/TWF/HDF/PWF/OSF）。\n"
         "Stage 2：再排除 TWF（46 筆），共 9,936 筆，4 類標籤（正常/HDF/PWF/OSF）。\n"
         "80/20 分層切割（stratify=y），MinMaxScaler 僅在訓練集 fit。\n"
         "SMOTE（k=3）只在訓練集執行，Stage 2 四類各平衡至 7,721 筆。",
         """# 正確前處理順序（避免 Data Leakage）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)
scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)   # 只在訓練集 fit
X_test_sc  = scaler.transform(X_test)        # 測試集只 transform
sm = SMOTE(random_state=0, k_neighbors=3)
X_tr_sm, y_tr_sm = sm.fit_resample(X_train_sc, y_train)  # 只在訓練集"""),
        ("06", "10 個模型訓練與評估（Stage 2）", "#7bf7c8",
         "傳統 ML：KNN、Decision Tree、Random Forest、Gradient Boosting、XGBoost、LightGBM\n"
         "深度學習：MLP (original)、MLP (upgraded)（128-64-32）、TabNet（Sparse Attention）\n"
         "集成策略：Stacking（XGB+LGBM → RF）\n"
         "主指標：MCC（最適合類別不平衡問題）；輔助：AUC、F1(weighted)、Recall(macro)\n"
         "Stage 2 最佳：LightGBM MCC=0.9291，各模型平均較 Stage 1 提升 +0.16",
         None),
        ("07", "特徵工程消融實驗（實驗B）", "#f97316",
         "基準（完整 11 特徵）：MCC=0.8076\n"
         "移除功率：ΔMCC=−0.019（最重要衍生特徵，對應 PWF）\n"
         "移除溫差：ΔMCC=−0.011（對應 HDF）\n"
         "移除功率×磨耗：ΔMCC=−0.011（對應 OSF）\n"
         "僅用原始特徵（7個）：MCC=0.719（−0.088）\n"
         "→ 衍生特徵有實質貢獻，完整特徵組合最佳",
         None),
        ("08", "SHAP 可解釋性分析（實驗C）", "#5b9fd4",
         "對 Stage 2 XGBoost 計算各故障類別的 SHAP 值。\n"
         "結果：3/3 故障類別的 SHAP Top-1 符合物理觸發條件：\n"
         "HDF → 溫差（SHAP=4.141）\n"
         "PWF → 功率（SHAP=4.011）\n"
         "OSF → 功率×磨耗（SHAP=4.257）\n"
         "TabNet 全局特徵重要性排序與 XGBoost 高度一致。\n"
         "→ 模型學到了工程師的領域知識",
         None),
    ]

    for num, title, color, desc, code in steps:
        with st.expander(f"**步驟 {num} — {title}**",
                         expanded=(num in ['01', '03', '04', '05'])):
            st.markdown(f"""
            <div style="border-left:3px solid {color};padding-left:1rem;margin-bottom:.5rem">
                <div style="color:#b0b0cc;font-size:.87rem;line-height:1.9">
                    {desc.replace(chr(10), '<br>')}
                </div>
            </div>""", unsafe_allow_html=True)
            if code:
                st.code(code, language='python')

    # ── 關鍵設計決策 ─────────────────────────────────────────────────
    sec("💡 關鍵設計決策與踩坑紀錄")

    bugs = [
        ("🐛 Temperature/Power 出現 inf", "#f87171",
         "Power=0 時 Temperature/Power=inf，導致 MinMaxScaler 報錯。\n"
         "修正：np.where(Power!=0, Temp_diff/Power, 0.0)"),
        ("⚠️ F1-Score 必須指定 average", "#f7c47b",
         "f1_score() 預設 binary，多分類必須加 average='weighted'，否則計算完全錯誤。\n"
         "所有基準實驗都必須注意此問題。"),
        ("🚫 MLP sample_weight 崩潰", "#f97316",
         "balanced sample_weight 使少數類別權重過高，MLP 訓練完全偏移。\n"
         "修正：改用 SMOTE 取代 sample_weight。"),
        ("🔀 Stacking meta-learner 崩潰", "#a78bfa",
         "LogisticRegression 在多類別不平衡問題下 Accuracy 崩潰至 30%。\n"
         "修正：改用 RandomForestClassifier(class_weight='balanced') 作為 meta-learner。"),
        ("📊 MCC 才是正確主指標", "#7bf7c8",
         "Accuracy 在類別不平衡時虛高（全預測正常仍有 96.6%）。\n"
         "MCC 同時考慮 TP/TN/FP/FN，對類別不平衡最穩健，應作為主要評估指標。"),
        ("🔧 XGBoost 特徵名稱含括號", "#7bb4f7",
         "特徵名稱如 'Air temperature [K]' 中的 [] 會造成 XGBoost 錯誤。\n"
         "修正：訓練前將欄位名稱去除括號（FEAT_COLS_SAFE）。"),
    ]

    cl, cr = st.columns(2)
    for i, (title_b, color, text) in enumerate(bugs):
        col = cl if i % 2 == 0 else cr
        col.markdown(f"""
        <div class="kpi-card" style="border-left:4px solid {color};padding:1rem 1.2rem;margin-bottom:.8rem">
            <div style="font-weight:700;color:{color};font-size:.88rem;margin-bottom:.3rem">
                {title_b}
            </div>
            <div style="font-size:.81rem;color:#b0b0cc;line-height:1.7">
                {text.replace(chr(10), '<br>')}
            </div>
        </div>""", unsafe_allow_html=True)

    sec("📦 套件需求（requirements.txt）")
    st.code("""streamlit>=1.32
scikit-learn>=1.4
xgboost>=2.0
lightgbm>=4.0
imbalanced-learn>=0.12
pytorch-tabnet>=4.0
torch>=2.0
shap>=0.44
plotly>=5.18
pandas>=2.0
numpy>=1.26""", language='text')

    sec("🚀 本地啟動")
    st.code("""# 安裝套件
pip install -r requirements.txt

# 啟動應用
streamlit run app.py""", language='bash')
