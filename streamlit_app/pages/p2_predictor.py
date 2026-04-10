"""p2_predictor.py — 即時預測器（整合兩階段決策）"""
import warnings; warnings.filterwarnings('ignore')
import streamlit as st
import plotly.graph_objects as go
import numpy as np, pandas as pd
from collections import Counter
from pages.loader import (load_bundle, FAILURE_LABELS, FAILURE_LONG,
                           mcolor, sec, plotly_base, kpi_card)

FEAT_COLS = ['Air temperature [K]','Process temperature [K]',
             'Rotational speed [rpm]','Torque [Nm]','Tool wear [min]',
             'Power','Power wear','Temperature difference',
             'Temperature power','Type_L','Type_M']

PRESETS = {
    '✅ 正常運作':     (300.0, 310.0, 1500, 40.0,  50, 'M'),
    '🌡️ 散熱不良 HDF': (301.0, 307.5, 1320, 52.0,  80, 'L'),
    '⚡ 功率異常 PWF':  (299.0, 309.0, 1530, 64.0, 120, 'H'),
    '💪 過負荷 OSF':    (300.0, 310.0, 1200, 70.0, 175, 'M'),
    '🔧 刀具磨耗 TWF': (300.0, 310.0, 1450, 42.0, 215, 'L'),
}

def preprocess(air_t, proc_t, rpm, torque, wear, m_type):
    power      = rpm * torque
    power_wear = power * wear
    temp_diff  = proc_t - air_t
    temp_power = temp_diff / power if power != 0 else 0.0
    type_l = 1 if m_type == 'L' else 0
    type_m = 1 if m_type == 'M' else 0
    row = [air_t, proc_t, rpm, torque, wear,
           power, power_wear, temp_diff, temp_power, type_l, type_m]
    return pd.DataFrame([row], columns=FEAT_COLS)

def stage1_check(wear, air_t, proc_t, rpm, torque, m_type):
    """第一階段規則決策（使用正確的物理公式）"""
    import math
    alerts      = []
    power_w     = rpm * 2 * math.pi / 60 * torque   # 正確功率（W）
    temp_diff   = proc_t - air_t
    torque_wear = torque * wear                       # Nm·min（OSF 觸發條件）
    thresh      = {'L':11000,'M':12000,'H':13000}.get(m_type.upper(), 12000)

    if wear >= 200:
        alerts.append(('TWF_WARN',
            f'刀具磨耗已達 {wear:.0f} min，超過 200 min 門檻，建議安排換刀評估', '#f7c47b'))
    if temp_diff < 8.6 and rpm < 1380:
        alerts.append(('HDF_RULE',
            f'溫差={temp_diff:.2f}K（<8.6K）且轉速={rpm:.0f}rpm（<1380）→ 符合 HDF 觸發條件', '#f97316'))
    if power_w < 3500 or power_w > 9000:
        alerts.append(('PWF_RULE',
            f'功率={power_w:.0f}W 超出正常範圍 [3500, 9000]W → 符合 PWF 觸發條件', '#fbbf24'))
    if torque_wear > thresh:
        alerts.append(('OSF_RULE',
            f'扭矩×磨耗={torque_wear:.0f} Nm·min（>{thresh}，{m_type}型門檻）→ 符合 OSF 觸發條件', '#a78bfa'))
    return alerts

def show():
    b       = load_bundle()
    scaler  = b['scaler']
    models  = b['models']

    st.markdown("# 🔮 即時預測器")
    st.markdown("""
    <p style='color:#8888aa;margin-top:-.4rem'>
    整合兩階段決策邏輯：先執行第一階段規則檢查，再送入第二階段 ML/DL 模型預測
    </p>""", unsafe_allow_html=True)

    # ── 快速情境 ─────────────────────────────────────────────────────
    sec("⚡ 快速情境選擇")
    col_pre, _ = st.columns([2,3])
    with col_pre:
        preset = st.selectbox("", ['— 自訂 —'] + list(PRESETS.keys()))

    d_air,d_proc,d_rpm,d_torq,d_wear,d_type = (
        PRESETS[preset] if preset in PRESETS else (300.0,310.0,1500,40.0,100,'M')
    )

    # ── 感測器輸入 ───────────────────────────────────────────────────
    sec("🎛️ 感測器輸入")
    c1,c2,c3 = st.columns(3)
    with c1:
        air_t  = st.slider("🌡️ 空氣溫度 [K]",     292.0, 308.0, d_air,  0.1)
        proc_t = st.slider("🔥 製程溫度 [K]",     301.0, 318.0, d_proc, 0.1)
    with c2:
        rpm    = st.slider("⚙️ 轉速 [rpm]",       1168,  2886,  d_rpm,  1)
        torque = st.slider("🔩 扭矩 [Nm]",        0.0,   77.0,  d_torq, 0.1)
    with c3:
        wear   = st.slider("🪛 刀具磨耗 [min]",   0,     250,   d_wear, 1)
        m_type = st.radio("🏷️ 機台類型",['L','M','H'],
                          index=['L','M','H'].index(d_type), horizontal=True)

    # 衍生特徵展示
    power      = rpm * torque
    temp_diff  = proc_t - air_t
    d1,d2,d3   = st.columns(3)
    for col, lbl, val, unit in [
        (d1,'⚡ 功率',      power,    'W'),
        (d2,'🌡️ 溫差',      temp_diff,'K'),
        (d3,'🔩 功率×磨耗', power*wear,''),
    ]:
        col.markdown(f"""
        <div class="kpi-card" style="padding:.7rem 1rem">
            <div style="font-size:.70rem;color:#6666aa">{lbl}</div>
            <div style="font-family:monospace;font-size:1.2rem;color:#a0b4ff">
                {val:,.1f}<span style="font-size:.72rem;color:#6666aa"> {unit}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    # 第一階段：規則決策
    # ════════════════════════════════════════════════════════════════
    sec("🔍 第一階段：規則決策結果")
    alerts = stage1_check(wear, air_t, proc_t, rpm, torque, m_type)

    if not alerts:
        st.markdown("""
        <div style='background:#0a1a0f;border:1px solid #4cde80;border-radius:10px;
                    padding:.9rem 1.3rem;color:#4cde80;font-size:.9rem'>
            ✅ 第一階段規則檢查通過，無觸發條件，繼續送入第二階段 ML/DL 模型。
        </div>""", unsafe_allow_html=True)
    else:
        for code, msg, color in alerts:
            icon = '⚠️' if 'WARN' in code else '🔴'
            bg   = '#1a1400' if 'WARN' in code else '#1a0a0a'
            st.markdown(f"""
            <div style='background:{bg};border:1px solid {color};border-radius:10px;
                        padding:.85rem 1.2rem;margin-bottom:.5rem;font-size:.87rem'>
                {icon} <b style='color:{color}'>[{code}]</b>
                <span style='color:#c0c8ff;margin-left:.5rem'>{msg}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    # 第二階段：ML/DL 預測
    # ════════════════════════════════════════════════════════════════
    sec("🤖 第二階段：ML/DL 模型預測（HDF / PWF / OSF / 正常）")

    X_df     = preprocess(air_t, proc_t, rpm, torque, wear, m_type)
    X_sc_arr = scaler.transform(X_df)
    X_sc     = pd.DataFrame(X_sc_arr, columns=FEAT_COLS)

    preds_all = {}
    for name, mdl in models.items():
        try:
            yp   = int(mdl.predict(X_sc)[0])
            prob = mdl.predict_proba(X_sc)[0]
            if len(prob) < 4:
                p4 = np.zeros(4)
                for i,c in enumerate(mdl.classes_):
                    p4[int(c)] = prob[i]
                prob = p4
        except Exception:
            yp = 0; prob = np.array([1,0,0,0],dtype=float)
        preds_all[name] = (yp, prob)

    # 主結果（Random Forest）
    primary = 'Random Forest'
    p_cls, p_prob = preds_all.get(primary, (0, np.array([1,0,0,0])))
    lbl, emoji, color = FAILURE_LABELS[p_cls]

    st.markdown(f"""
    <div class="kpi-card" style="text-align:center;padding:1.4rem;margin-bottom:1.2rem;
                border-left:none;border:2px solid {color}">
        <div style="font-size:2.5rem">{emoji}</div>
        <div style="font-size:1.5rem;font-weight:700;color:#e0e0ff;margin:.2rem 0">{lbl}</div>
        <div style="color:#8888aa;font-size:.82rem">
            {primary} · 類別 {p_cls} · 信心度
        </div>
        <div style="font-family:monospace;font-size:2rem;color:{color}">
            {p_prob[p_cls]*100:.1f}%
        </div>
    </div>""", unsafe_allow_html=True)

    # 全模型網格
    cols_g = st.columns(3)
    for i,(name,(cls,prob)) in enumerate(preds_all.items()):
        fl,em,co = FAILURE_LABELS[cls]
        cf = prob[cls]*100
        mc = mcolor(name)
        tag = 'DL' if name in ['MLP (upgraded)','Stacking(XGB+LGBM)'] else 'ML'
        cols_g[i%3].markdown(f"""
        <div class="kpi-card" style="padding:.8rem 1rem;border-left:3px solid {mc}">
            <div style="display:flex;justify-content:space-between;margin-bottom:.2rem">
                <span style="font-size:.70rem;color:#8888aa">{name}</span>
                <span style="font-size:.65rem;color:{mc}">[{tag}]</span>
            </div>
            <div style="display:flex;align-items:center;gap:.5rem">
                <span style="font-size:1.3rem">{em}</span>
                <div>
                    <div style="font-weight:600;color:#e0e0ff;font-size:.85rem">{fl}</div>
                    <div style="font-family:monospace;color:{mc};font-size:.78rem">
                        {cf:.1f}% 信心
                    </div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # 機率分布圖
    sec("📊 機率分布（Random Forest）")
    cls_lbl = [f'{FAILURE_LONG[i]}（{i}）' for i in range(4)]
    bc      = [color if i==p_cls else '#4a6cf7' for i in range(4)]
    fig = go.Figure(go.Bar(
        x=cls_lbl, y=p_prob*100,
        marker_color=bc,
        text=[f'{v:.2f}%' for v in p_prob*100],
        textposition='outside', textfont=dict(size=11),
        hovertemplate='%{x}<br>%{y:.4f}%<extra></extra>',
    ))
    fig.update_layout(**plotly_base(
        height=280,
        yaxis=dict(title='機率 (%)', gridcolor='#2a2a4a'),
        xaxis=dict(tickfont=dict(size=10)),
    ))
    st.plotly_chart(fig, use_container_width=True)

    # 投票共識
    vote  = Counter(c for c,_ in preds_all.values())
    maj   = vote.most_common(1)[0][0]
    agree = vote[maj] / len(preds_all) * 100
    st.markdown(f"""
    <div class="kpi-card" style="padding:.9rem 1.2rem">
        <div class="kpi-lbl">🗳️ 模型投票共識</div>
        <div style="display:flex;align-items:center;gap:1.5rem;margin-top:.4rem">
            <div>
                <div style="font-family:monospace;font-size:1.5rem;color:#7bf7c8">{agree:.0f}%</div>
                <div style="font-size:.75rem;color:#8888aa">一致率</div>
            </div>
            <div style="color:#c0c8ff;font-size:.88rem">
                {vote[maj]}/{len(preds_all)} 個模型預測 →
                <b style="color:{FAILURE_LABELS[maj][2]}">{FAILURE_LABELS[maj][0]}</b>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
