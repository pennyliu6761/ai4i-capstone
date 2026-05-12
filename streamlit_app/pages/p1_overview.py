"""p1_overview.py — 總覽儀表板（v6）"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from pages.loader import (load_bundle, load_results, mcolor,
                           ML_MODELS, DL_MODELS, kpi_card, sec, plotly_base,
                           FAIL_COLORS, FAILURE_LONG)

def show():
    st.markdown("# 🏠 總覽儀表板")
    st.markdown("<p style='color:#8888aa;margin-top:-.4rem'>AI4I 2020 · 兩階段預測性維護決策支援系統 · 成果總覽</p>",
                unsafe_allow_html=True)

    b  = load_bundle()
    df = load_results()

    # ── KPI ──────────────────────────────────────────────────────────
    sec("🎯 第二階段模型效能（排除 TWF & RNF 後，10 個模型）")
    best = df['MCC'].idxmax()
    br   = df.loc[best]
    k1,k2,k3,k4,k5 = st.columns(5)
    kpi_card(k1, f"{br['MCC']:.4f}",      f"最佳 MCC — {best}",   "#7bf7c8")
    kpi_card(k2, f"{br['AUC']:.4f}",      f"最佳 AUC — {best}",   "#7bb4f7")
    kpi_card(k3, f"{br['Accuracy']:.2%}", f"最佳準確率 — {best}", "#f7c47b")
    kpi_card(k4, "10",                     "比較模型數",            "#c47bf7")
    kpi_card(k5, "9,936",                  "第二階段資料筆數",      "#f77b7b")

    # ── 兩階段架構說明卡 ─────────────────────────────────────────────
    sec("🗂️ 兩階段決策框架")
    c_s1, c_arr, c_s2 = st.columns([5, 1, 5])

    with c_s1:
        st.markdown("""
        <div style='background:#1a1400;border:2px solid #f7c47b;border-radius:14px;padding:1.2rem'>
            <div style='color:#f7c47b;font-size:1rem;font-weight:700;margin-bottom:.8rem'>
                第一階段：故障可診斷性篩選
            </div>
            <div style='font-size:.82rem;color:#c0c8ff;line-height:2.0'>
                🔴 <b style='color:#f87171'>RNF</b> — AUC=0.6629，純隨機，Recall=0<br>
                &nbsp;&nbsp;&nbsp;&nbsp;→ Stage 1 起排除，不納入建模<br><br>
                🟡 <b style='color:#f7c47b'>TWF</b> — AUC=0.9637，半隨機<br>
                &nbsp;&nbsp;&nbsp;&nbsp;→ Stage 1 保留，Stage 2 排除<br>
                &nbsp;&nbsp;&nbsp;&nbsp;→ 磨耗 &gt; 200 min 發出換刀警示<br><br>
                <span style='color:#8888aa;font-size:.75rem'>
                Stage 1：5 類分類（含 TWF），最佳 Stacking MCC=0.8716
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

    with c_arr:
        st.markdown("<div style='text-align:center;font-size:2rem;margin-top:3rem;color:#7bf7c8'>→</div>",
                    unsafe_allow_html=True)

    with c_s2:
        st.markdown("""
        <div style='background:#0a1a0f;border:2px solid #4cde80;border-radius:14px;padding:1.2rem'>
            <div style='color:#4cde80;font-size:1rem;font-weight:700;margin-bottom:.8rem'>
                第二階段：ML/DL 智慧分類
            </div>
            <div style='font-size:.82rem;color:#c0c8ff;line-height:2.0'>
                🟢 <b style='color:#f97316'>HDF</b> 散熱不良 — AUC=1.0000<br>
                🟢 <b style='color:#fbbf24'>PWF</b> 功率異常 — AUC=0.9992<br>
                🟢 <b style='color:#a78bfa'>OSF</b> 過負荷 &nbsp;&nbsp; — AUC=0.9998<br><br>
                <span style='color:#8888aa;font-size:.75rem'>
                10 個 ML/DL 模型 · 最佳 LightGBM MCC = 0.9291<br>
                各模型平均提升 +0.16（Stage 1 → Stage 2）
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── 兩階段 MCC 比較 ──────────────────────────────────────────────
    sec("📈 兩階段 MCC 進展（研究核心貢獻）")
    three = b.get('three_stage', {})

    # 只取兩個階段都有的模型
    models_ord = [m for m in ML_MODELS + DL_MODELS
                  if m in three and three[m].get('排除RNF') and three[m].get('排除TWF_RNF')]
    s1_mcc = [three[m]['排除RNF']     for m in models_ord]
    s2_mcc = [three[m]['排除TWF_RNF'] for m in models_ord]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Stage 1（排除 RNF）', x=models_ord, y=s1_mcc,
        marker_color='#7bb4f7', opacity=0.75,
        text=[f'{v:.4f}' for v in s1_mcc], textposition='outside', textfont=dict(size=9)))
    fig.add_trace(go.Bar(name='Stage 2（排除 TWF & RNF）', x=models_ord, y=s2_mcc,
        marker_color='#7bf7c8', opacity=0.9,
        text=[f'{v:.4f}' for v in s2_mcc], textposition='outside', textfont=dict(size=9)))

    avg_delta = np.mean([s2 - s1 for s1, s2 in zip(s1_mcc, s2_mcc)])
    fig.add_annotation(x=len(models_ord)//2, y=1.02,
        text=f"平均提升 +{avg_delta:.3f}",
        font=dict(size=10, color='#f7c47b'), showarrow=False)
    fig.update_layout(**plotly_base(
        height=380, barmode='group',
        yaxis=dict(title='MCC', gridcolor='#2a2a4a', range=[0.4, 1.05]),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10), orientation='h', y=-0.12),
        xaxis=dict(tickangle=-15),
    ))
    st.plotly_chart(fig, use_container_width=True)

    # ── MCC 排行榜 + AUC 雷達 ───────────────────────────────────────
    sec("🏆 模型排行榜 vs 多指標雷達圖")
    cl, cr = st.columns([3, 2], gap='large')

    with cl:
        df_s   = df.sort_values('MCC', ascending=True)
        colors = [mcolor(n) for n in df_s.index]
        fig2 = go.Figure(go.Bar(
            x=df_s['MCC'], y=df_s.index, orientation='h',
            marker=dict(color=colors),
            text=[f'{v:.4f}' for v in df_s['MCC']],
            textposition='outside', textfont=dict(size=10),
            hovertemplate='<b>%{y}</b><br>MCC：%{x:.4f}<extra></extra>',
        ))
        fig2.update_layout(**plotly_base(
            height=380,
            xaxis=dict(range=[0.45, 1.02], gridcolor='#2a2a4a', title='MCC'),
            bargap=0.2,
        ))
        st.plotly_chart(fig2, use_container_width=True)

    with cr:
        cats    = ['MCC', 'AUC', 'Accuracy', 'Recall_macro', 'MCC']
        cat_lbl = ['MCC', 'AUC', '準確率', '召回率(macro)', 'MCC']
        pal     = ['#7bf7c8', '#7bb4f7', '#f7c47b', '#c47bf7']
        top4    = df.nlargest(4, 'MCC')
        fig3    = go.Figure()
        for i, (nm, row) in enumerate(top4.iterrows()):
            vals = [row[c] for c in cats]
            c    = pal[i]
            r, g, bv = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
            fig3.add_trace(go.Scatterpolar(
                r=vals, theta=cat_lbl, fill='toself', name=nm,
                line=dict(color=c, width=2),
                fillcolor=f'rgba({r},{g},{bv},0.08)',
            ))
        fig3.update_layout(**plotly_base(
            height=380,
            polar=dict(
                bgcolor='rgba(20,20,40,.5)',
                radialaxis=dict(visible=True, range=[0.5, 1.0],
                    gridcolor='#2a2a4a', tickfont=dict(size=8)),
                angularaxis=dict(gridcolor='#2a2a4a'),
            ),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=8), x=1.05),
            margin=dict(l=8, r=8, t=8, b=8),
        ))
        st.plotly_chart(fig3, use_container_width=True)

    # ── SHAP 一致性摘要 ──────────────────────────────────────────────
    sec("🧠 SHAP 可解釋性：模型決策 vs 物理規則")
    shap_data = [
        ('HDF 散熱不良', '#f97316', '溫差 < 8.6K 且 轉速 < 1380 rpm', '溫差', '4.141', '✅'),
        ('PWF 功率異常', '#fbbf24', '功率超出範圍 [3500, 9000] W',      '功率', '4.011', '✅'),
        ('OSF 過負荷',   '#a78bfa', '功率×磨耗超型別門檻',               '功率×磨耗', '4.257', '✅'),
    ]
    sc1, sc2, sc3 = st.columns(3)
    for col, (name, c, rule, top1, shap_val, check) in zip([sc1, sc2, sc3], shap_data):
        col.markdown(f"""
        <div class="kpi-card" style="border-left:4px solid {c}">
            <div style="color:{c};font-weight:700;font-size:.9rem">{name}</div>
            <div style="color:#8888aa;font-size:.75rem;margin:.3rem 0">
                物理規則：{rule}
            </div>
            <div style="color:#c0c8ff;font-size:.85rem">
                SHAP Top-1：<b style="color:{c}">{top1}</b>
                <span style="color:#8888aa;font-size:.75rem"> = {shap_val}</span>
            </div>
            <div style="font-size:1.2rem;margin-top:.3rem">{check} 符合物理規則</div>
        </div>""", unsafe_allow_html=True)
