"""p5_methodology.py — 研究方法論"""
import streamlit as st
from pages.loader import sec

def show():
    st.markdown("# 📖 研究方法論")
    st.markdown("<p style='color:#8888aa;margin-top:-.4rem'>完整研究流程 · 關鍵設計決策 · 踩坑與修正紀錄</p>",
                unsafe_allow_html=True)

    # ── 研究流程 ─────────────────────────────────────────────────────
    sec("🗓️ 完整研究流程")

    steps = [
        ("01","資料理解與 EDA","#7bf7c8",
         "讀入 AI4I 2020 CSV，確認 10,000 筆、14 欄位、無缺值。"
         "計算各故障樣本數與比例（正常佔 81.8%，故障類別嚴重不平衡）。"
         "使用相關係數熱圖、ProfileReport 自動報告、PairPlot 散點圖進行初步探索。",
         None),
        ("02","特徵工程","#7bb4f7",
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
        ("03","故障可診斷性量化（1-vs-Rest）","#f7c47b",
         "對每種故障分別做 1-vs-Rest 二元分類（XGBoost + 5-Fold CV）。\n"
         "結果：HDF/PWF/OSF AUC>0.999（規則型）；TWF AUC=0.935（半隨機）；RNF AUC=0.503（純隨機）。\n"
         "確立研究設計依據：RNF 排除、TWF 改用規則警示、僅對 HDF/PWF/OSF 建模。",
         None),
        ("04","TWF 貝葉斯誤差分析","#f87171",
         "磨耗 200–250 min 區間：770 筆 TWF + 976 筆正常，感測器特徵幾乎完全相同。\n"
         "此為不可約誤差（Irreducible Error），任何模型均無法克服。\n"
         "正確策略：磨耗>200 min 直接發出換刀警示，不需要 ML。",
         """# 驗證貝葉斯誤差
mask_zone = (df['Tool wear [min]'] >= 200) & (df['Tool wear [min]'] <= 250)
n_twf  = df[mask_zone & (df['TWF']==1)].shape[0]   # 770
n_norm = df[mask_zone & (df['Machine failure']==0)].shape[0]  # 976
print(f"TWF發生率：{n_twf/(n_twf+n_norm):.1%}")  # ~44%"""),
        ("05","資料前處理（Stage 2）","#c47bf7",
         "移除 TWF=1 和 RNF=1 的樣本（10,000 → 9,221 筆）。\n"
         "建立四元標籤：0=正常、1=HDF、2=PWF、3=OSF（重疊樣本以 HDF 優先）。\n"
         "80/20 分層切割（stratify=y），MinMaxScaler 僅在訓練集 fit。\n"
         "SMOTE（k=3）只在訓練集執行，四類各平衡至 ~6,546 筆。",
         """# 正確前處理順序（避免 Data Leakage）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)
scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)   # 只在訓練集 fit
X_test_sc  = scaler.transform(X_test)        # 測試集只 transform
sm = SMOTE(random_state=0, k_neighbors=3)
X_tr_sm, y_tr_sm = sm.fit_resample(X_train_sc, y_train)  # 只在訓練集"""),
        ("06","6 個模型訓練與評估","#7bf7c8",
         "傳統 ML：Random Forest、XGBoost（調參版）、LightGBM\n"
         "深度學習：MLP 升級版（128-64-32）、TabNet（Sparse Attention）\n"
         "集成策略：Stacking（XGB+LGBM → RF）\n"
         "主指標：MCC（最適合類別不平衡問題）；輔助：AUC、F1(weighted)、Recall(macro)",
         """# XGBoost 調參（以 MCC 為目標）
search = RandomizedSearchCV(
    XGBClassifier(eval_metric='mlogloss'),
    param_distributions={'max_depth':[4,5,6,7],'learning_rate':[.05,.1,.15],
                         'n_estimators':[100,150,200],'subsample':[.8,.9,1.]},
    scoring=make_scorer(matthews_corrcoef), n_iter=15, cv=3, random_state=0)
search.fit(X_tr_sm, y_tr_sm)"""),
        ("07","SHAP 可解釋性分析","#5b9fd4",
         "對 XGBoost 計算各故障類別的 SHAP 值，驗證 Top-1 特徵與物理規則的一致性。\n"
         "對比 TabNet 的 Sparse Attention 全局特徵重要性，確認跨架構的排序一致性。\n"
         "結果：3/3 故障類別的 SHAP Top-1 符合物理觸發條件，兩模型全局排序高度一致。",
         None),
    ]

    for num,title,color,desc,code in steps:
        with st.expander(f"**步驟 {num} — {title}**", expanded=(num in ['01','03','04','05'])):
            st.markdown(f"""
            <div style="border-left:3px solid {color};padding-left:1rem;margin-bottom:.5rem">
                <div style="color:#b0b0cc;font-size:.87rem;line-height:1.9">
                    {desc.replace(chr(10),'<br>')}
                </div>
            </div>""", unsafe_allow_html=True)
            if code:
                st.code(code, language='python')

    # ── 關鍵設計決策 ─────────────────────────────────────────────────
    sec("💡 關鍵設計決策與踩坑紀錄")

    bugs = [
        ("🐛 Temperature/Power 出現 inf","#f87171",
         "Power=0 時 Temperature/Power=inf，導致 MinMaxScaler 報錯。\n"
         "修正：np.where(Power!=0, Temp_diff/Power, 0.0)"),
        ("⚠️ F1-Score 必須指定 average","#f7c47b",
         "f1_score() 預設 binary，多分類必須加 average='weighted'，否則計算完全錯誤。\n"
         "所有基準實驗都必須注意此問題。"),
        ("🚫 MLP sample_weight 崩潰","#f97316",
         "balanced sample_weight 使 RNF 權重達 190×，MLP 訓練完全偏移（MCC 0.71→0.55）。\n"
         "修正：改用 SMOTE 取代 sample_weight，或限制最大權重倍率。"),
        ("🔀 Stacking meta-learner 崩潰","#a78bfa",
         "LogisticRegression 在多類別不平衡問題下 Accuracy 崩潰至 30%。\n"
         "修正：改用 RandomForestClassifier(class_weight='balanced') 作為 meta-learner。"),
        ("📊 MCC 才是正確主指標","#7bf7c8",
         "Accuracy 在類別不平衡時虛高（全預測正常仍有 88.7%）。\n"
         "MCC 同時考慮 TP/TN/FP/FN，對類別不平衡最穩健，應作為主要評估指標。"),
        ("🔧 XGBoost 特徵名稱含括號","#7bb4f7",
         "特徵名稱如 'Air temperature [K]' 中的 [] 會造成 XGBoost 錯誤。\n"
         "修正：訓練前將 DataFrame 轉為 numpy array，或重新命名特徵。"),
    ]

    cl,cr = st.columns(2)
    for i,(title_b,color,text) in enumerate(bugs):
        col = cl if i%2==0 else cr
        col.markdown(f"""
        <div class="kpi-card" style="border-left:4px solid {color};padding:1rem 1.2rem;margin-bottom:.8rem">
            <div style="font-weight:700;color:{color};font-size:.88rem;margin-bottom:.3rem">
                {title_b}
            </div>
            <div style="font-size:.81rem;color:#b0b0cc;line-height:1.7">
                {text.replace(chr(10),'<br>')}
            </div>
        </div>""", unsafe_allow_html=True)

    # ── 套件需求 ─────────────────────────────────────────────────────
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
