"""p7_batch.py — 批次預測上傳器"""
import warnings; warnings.filterwarnings('ignore')
import streamlit as st
import plotly.graph_objects as go
import pandas as pd, numpy as np, io
from pages.loader import (load_bundle, FAILURE_LABELS, FAILURE_LONG,
                           FAILURE_SHORT, FAIL_COLORS, sec, plotly_base)

FEAT_COLS = ['Air temperature [K]','Process temperature [K]',
             'Rotational speed [rpm]','Torque [Nm]','Tool wear [min]',
             'Power','Power wear','Temperature difference',
             'Temperature power','Type_L','Type_M']
INPUT_COLS = ['Type','Air temperature [K]','Process temperature [K]',
              'Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']

def preprocess(df_raw):
    df = df_raw.copy()
    df['Power']                  = df['Rotational speed [rpm]'] * df['Torque [Nm]']
    df['Power wear']             = df['Power'] * df['Tool wear [min]']
    df['Temperature difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['Temperature power']      = np.where(df['Power']!=0,
        df['Temperature difference']/df['Power'], 0.0)
    df['Type_L'] = (df['Type'].str.upper()=='L').astype(int)
    df['Type_M'] = (df['Type'].str.upper()=='M').astype(int)
    return df

def stage1_flag(row):
    """第一階段規則旗標（正確物理公式）"""
    import math
    flags = []
    wear  = float(row.get('Tool wear [min]',0))
    rpm   = float(row.get('Rotational speed [rpm]',0))
    torque= float(row.get('Torque [Nm]',0))
    td    = float(row.get('Temperature difference',0))
    t     = str(row.get('Type','M')).upper()
    power_w     = rpm * 2 * math.pi / 60 * torque   # 正確功率（W）
    torque_wear = torque * wear                       # Nm·min
    thresh      = {'L':11000,'M':12000,'H':13000}.get(t, 12000)
    if wear >= 200:                          flags.append('TWF⚠️ 換刀警示')
    if td < 8.6 and rpm < 1380:             flags.append('HDF規則')
    if power_w < 3500 or power_w > 9000:    flags.append('PWF規則')
    if torque_wear > thresh:                 flags.append('OSF規則')
    return '、'.join(flags) if flags else '—'

def show():
    b       = load_bundle()
    scaler  = b['scaler']
    models  = b['models']

    st.markdown("# 📂 批次預測上傳器")
    st.markdown("""<p style='color:#8888aa;margin-top:-.4rem'>
    上傳 CSV → 第一階段規則檢查 → 第二階段 ML/DL 預測 → 下載結果</p>""",
    unsafe_allow_html=True)

    # ── 格式說明 + 測試檔 ─────────────────────────────────────────
    with st.expander("📋 CSV 格式說明 & 測試資料下載", expanded=True):
        st.markdown("""
| 欄位名稱 | 說明 | 範例 |
|---|---|---|
| `Type` | 機台類型 L / M / H | M |
| `Air temperature [K]` | 空氣溫度 | 298.5 |
| `Process temperature [K]` | 製程溫度 | 309.2 |
| `Rotational speed [rpm]` | 轉速 | 1450 |
| `Torque [Nm]` | 扭矩 | 42.3 |
| `Tool wear [min]` | 刀具磨耗時間 | 120 |

系統自動計算衍生特徵，並先執行 Stage 1 規則檢查。
        """)
        import os
        bat_path = os.path.join(os.path.dirname(__file__),'..','models','batch_test.csv')
        if os.path.exists(bat_path):
            with open(bat_path,'rb') as f: bat_bytes=f.read()
            st.download_button("⬇️ 下載測試資料（200 筆，含多種故障情境）",
                data=bat_bytes, file_name='batch_test.csv', mime='text/csv')

    # ── 上傳 ─────────────────────────────────────────────────────────
    sec("📤 上傳感測器資料")
    uploaded = st.file_uploader("選擇 CSV 檔案", type=['csv'])

    sel_models = st.multiselect("選擇預測模型",list(models.keys()),
        default=list(models.keys())[:3])

    if not uploaded:
        st.info("👆 請上傳 CSV 檔案，或先下載測試資料。")
        return

    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"CSV 讀取失敗：{e}"); return

    missing = [c for c in INPUT_COLS if c not in df_raw.columns]
    if missing:
        st.error(f"缺少必要欄位：{missing}"); return

    st.success(f"✅ 成功讀取 **{len(df_raw)}** 筆資料")
    with st.expander("🔍 原始資料預覽（前 5 筆）"):
        st.dataframe(df_raw.head(5), use_container_width=True)

    # ── 前處理 & 預測 ────────────────────────────────────────────────
    df_feat = preprocess(df_raw)
    X_df    = pd.DataFrame(scaler.transform(df_feat[FEAT_COLS]), columns=FEAT_COLS)

    if not sel_models:
        st.warning("請至少選擇一個模型。"); return

    result_df = df_raw[INPUT_COLS].copy()

    # Stage 1 規則旗標
    result_df['Stage1 規則旗標'] = df_feat.apply(stage1_flag, axis=1)

    prog = st.progress(0, text="計算中…")
    prob_store = {}
    for mi,mname in enumerate(sel_models):
        mdl  = models[mname]
        yp   = mdl.predict(X_df).astype(int)
        rp   = mdl.predict_proba(X_df)
        if rp.shape[1]<4:
            p4=np.zeros((len(yp),4))
            for i,c in enumerate(mdl.classes_): p4[:,int(c)]=rp[:,i]
            rp=p4
        conf = rp[np.arange(len(yp)),yp]
        cname = mname.replace(' ','_').replace('(','').replace(')','').replace('+','')
        result_df[f'{cname}_預測']  = [FAILURE_LONG[c] for c in yp]
        result_df[f'{cname}_信心度'] = (conf*100).round(1)
        prob_store[mname] = (yp,rp)
        prog.progress((mi+1)/len(sel_models), text=f"完成 {mname}")
    prog.empty()

    # 主模型
    primary = sel_models[0]
    yp_main,probs_main = prob_store[primary]
    conf_main = probs_main[np.arange(len(yp_main)),yp_main]*100
    result_df.insert(len(INPUT_COLS)+1,'主要預測',    [FAILURE_LONG[c] for c in yp_main])
    result_df.insert(len(INPUT_COLS)+2,'信心度(%)',   conf_main.round(1))
    result_df.insert(len(INPUT_COLS)+3,'需要維護',
        ['⚠️ 是' if c!=0 else '✅ 否' for c in yp_main])

    # ── 統計摘要 ─────────────────────────────────────────────────────
    sec("📊 預測摘要統計")
    total = len(result_df)
    n_fail= (yp_main!=0).sum()
    n_twf_warn = (result_df['Stage1 規則旗標'].str.contains('TWF')).sum()

    s1,s2,s3,s4,s5 = st.columns(5)
    def sc(col,val,lbl,c):
        col.markdown(f"""
        <div class="kpi-card" style="padding:.8rem 1rem;text-align:center">
            <div style="font-family:monospace;font-size:1.5rem;font-weight:700;color:{c}">{val}</div>
            <div style="font-size:.70rem;color:#8888aa">{lbl}</div>
        </div>""", unsafe_allow_html=True)
    sc(s1,total,     '總筆數',    '#c0c8ff')
    sc(s2,total-n_fail,'Stage2 正常','#4cde80')
    sc(s3,n_fail,    'Stage2 故障','#f87171')
    sc(s4,f"{n_fail/total:.1%}",'故障率','#f7c47b')
    sc(s5,n_twf_warn,'Stage1 換刀警示','#f7c47b')

    # ── 圖表 ────────────────────────────────────────────────────────
    ch1,ch2 = st.columns(2)
    with ch1:
        cnt = pd.Series(yp_main).value_counts().sort_index()
        fig = go.Figure(go.Pie(
            labels=[FAILURE_LONG[i] for i in cnt.index],
            values=cnt.values, hole=0.5,
            marker=dict(colors=[FAIL_COLORS[i] for i in cnt.index],
                        line=dict(color='#0d0d1a',width=2)),
            textinfo='label+percent',textfont=dict(size=11)))
        fig.update_layout(**plotly_base(height=300,
            title=dict(text='預測類別分布',font=dict(size=12)),showlegend=False))
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        fig2 = go.Figure(go.Histogram(
            x=conf_main, nbinsx=20,
            marker_color='#4a6cf7',opacity=0.85,
            hovertemplate='信心度 %{x:.1f}%：%{y} 筆<extra></extra>'))
        fig2.update_layout(**plotly_base(height=300,
            title=dict(text='信心度分布',font=dict(size=12)),
            xaxis=dict(title='信心度 (%)',gridcolor='#2a2a4a'),
            yaxis=dict(title='筆數',      gridcolor='#2a2a4a')))
        st.plotly_chart(fig2, use_container_width=True)

    # ── 高風險清單 ───────────────────────────────────────────────────
    sec("⚠️ 高風險清單（Stage2 故障 且 信心度 ≥ 80%）")
    hr_mask = (yp_main!=0) & (conf_main>=80)
    hr_df   = result_df[hr_mask][
        INPUT_COLS[:2]+['Stage1 規則旗標','主要預測','信心度(%)','需要維護']
    ].copy()
    hr_df.index = [f"第 {i+1} 筆" for i in hr_df.index]
    if len(hr_df)>0:
        st.dataframe(
            hr_df.style.applymap(
                lambda v: 'color:#f87171' if '是' in str(v) else '',
                subset=['需要維護'])
            .format({'信心度(%)':'{:.1f}%'}),
            use_container_width=True,
            height=min(380, len(hr_df)*42+50))
    else:
        st.success("無高信心度故障預測。")

    # ── 多模型一致性 ─────────────────────────────────────────────────
    if len(sel_models)>=2:
        sec("🔁 多模型預測一致性")
        all_same = np.all([
            prob_store[m][0]==prob_store[sel_models[0]][0]
            for m in sel_models], axis=0)
        n_agree = all_same.sum()
        st.markdown(f"""
        <div class="kpi-card" style="padding:.9rem 1.2rem">
            <div class="kpi-lbl">模型完全一致率</div>
            <div style="font-family:monospace;font-size:1.5rem;color:#7bf7c8">
                {n_agree/total:.1%}
            </div>
            <div style="font-size:.78rem;color:#8888aa">
                {n_agree}/{total} 筆所有選擇模型預測結果相同
            </div>
        </div>""", unsafe_allow_html=True)

        diff_data = [[int((prob_store[m][0]==c).sum()) for c in range(4)]
                     for m in sel_models]
        diff_df = pd.DataFrame(diff_data, index=sel_models,
                               columns=[FAILURE_SHORT[c] for c in range(4)])
        fig3 = go.Figure(go.Heatmap(
            z=diff_df.values, x=list(diff_df.columns), y=list(diff_df.index),
            colorscale=[[0,'#1a1a3a'],[.5,'rgba(74,108,247,.7)'],[1,'rgba(123,247,200,1)']],
            text=[[str(v) for v in r] for r in diff_df.values],
            texttemplate='%{text}',textfont=dict(size=11),
            colorbar=dict(tickfont=dict(color='#8888aa'),thickness=12)))
        fig3.update_layout(**plotly_base(height=220,
            title=dict(text='各模型預測各類別筆數',font=dict(size=11)),
            xaxis=dict(title='故障類別'),
            yaxis=dict(title='',tickfont=dict(size=10))))
        st.plotly_chart(fig3, use_container_width=True)

    # ── 完整結果表 & 下載 ────────────────────────────────────────────
    sec("📋 完整預測結果")
    st.dataframe(result_df, use_container_width=True, height=360)

    sec("⬇️ 下載結果")
    csv_buf = io.StringIO()
    result_df.to_csv(csv_buf, index=False, encoding='utf-8-sig')
    d1,d2 = st.columns(2)
    d1.download_button("📥 下載完整結果 CSV",
        data=csv_buf.getvalue().encode('utf-8-sig'),
        file_name='prediction_results.csv', mime='text/csv',
        use_container_width=True)
    if len(hr_df)>0:
        hr_buf = io.StringIO()
        hr_df.to_csv(hr_buf, index=True, encoding='utf-8-sig')
        d2.download_button("🚨 下載高風險清單 CSV",
            data=hr_buf.getvalue().encode('utf-8-sig'),
            file_name='high_risk.csv', mime='text/csv',
            use_container_width=True)
