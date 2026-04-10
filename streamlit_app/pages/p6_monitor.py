"""p6_monitor.py — 即時監控看板"""
import warnings; warnings.filterwarnings('ignore')
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd, numpy as np, time
from pages.loader import (load_bundle, load_sim, FAILURE_LABELS,
                           FAILURE_LONG, sec)

FEAT_COLS = ['Air temperature [K]','Process temperature [K]',
             'Rotational speed [rpm]','Torque [Nm]','Tool wear [min]',
             'Power','Power wear','Temperature difference',
             'Temperature power','Type_L','Type_M']

STATUS = {
    0: ('#4cde80','#0a1a0f','🟢'),
    1: ('#f97316','#1a0f00','🔴'),
    2: ('#fbbf24','#1a1200','🟠'),
    3: ('#a78bfa','#100a1a','🟡'),
}

def show():
    b      = load_bundle()
    scaler = b['scaler']
    models = b['models']
    sim_df = load_sim()

    st.markdown("# 📡 即時監控看板")
    st.markdown("""<p style='color:#8888aa;margin-top:-.4rem'>
    模擬銑床從正常→散熱不良→功率異常→過負荷的完整動態過程</p>""",
    unsafe_allow_html=True)

    # ── 控制列 ───────────────────────────────────────────────────────
    c1,c2,c3,c4 = st.columns([1,1,1,2])
    with c1: speed  = st.selectbox("▶ 播放速度",["0.5×","1×","2×","5×"],index=1)
    with c2: window = st.slider("視窗（筆）",20,100,50,10)
    with c3:
        mdl_names = list(models.keys())
        default_i = mdl_names.index('Random Forest') if 'Random Forest' in mdl_names else 0
        sel_m = st.selectbox("預測模型",mdl_names,index=default_i)
    with c4:
        cb1,cb2 = st.columns(2)
        run_btn = cb1.button("▶ 開始播放",use_container_width=True,type="primary")
        rst_btn = cb2.button("⏹ 重置",    use_container_width=True)

    speed_map = {"0.5×":0.18,"1×":0.09,"2×":0.045,"5×":0.018}
    delay    = speed_map[speed]
    pred_mdl = models[sel_m]

    if rst_btn:
        st.session_state.pop('mon_step',None)
        st.session_state.pop('mon_hist',None)
        st.rerun()

    if 'mon_step' not in st.session_state: st.session_state.mon_step = 0
    if 'mon_hist' not in st.session_state: st.session_state.mon_hist = []

    step = st.session_state.mon_step
    hist = st.session_state.mon_hist

    banner_ph = st.empty()
    kpi_ph    = st.empty()
    chart_ph  = st.empty()
    log_ph    = st.empty()

    def render(hist):
        if not hist:
            banner_ph.info("按下「▶ 開始播放」啟動模擬（共 200 步）")
            return
        cur = hist[-1]
        cls = cur['pred_cls']
        fc,bg,em = STATUS[cls]

        # 橫幅
        banner_ph.markdown(f"""
        <div style='background:{bg};border:2px solid {fc};border-radius:12px;
                    padding:.9rem 1.5rem;display:flex;align-items:center;gap:1rem'>
            <span style='font-size:2rem'>{em}</span>
            <div>
                <div style='font-size:1.3rem;font-weight:700;color:{fc}'>
                    {FAILURE_LABELS[cls][0]}
                </div>
                <div style='font-size:.80rem;color:#aaaacc'>
                    時間步 t={cur['timestamp']} ·
                    情境：{cur.get('scenario','—')} ·
                    信心度 {cur['conf']:.1%}
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # KPI
        k1,k2,k3,k4,k5 = kpi_ph.columns(5)
        def kc(col,v,u,lb,c='#c0c8ff'):
            col.markdown(f"""
            <div class="kpi-card" style="padding:.65rem .9rem;text-align:center">
                <div style="font-family:monospace;font-size:1.15rem;font-weight:700;color:{c}">
                    {v}<span style="font-size:.68rem;color:#8888aa;margin-left:.15rem">{u}</span>
                </div>
                <div style="font-size:.67rem;color:#8888aa">{lb}</div>
            </div>""", unsafe_allow_html=True)
        kc(k1,f"{cur.get('Air temperature [K]',0):.1f}",'K','空氣溫度','#7bb4f7')
        kc(k2,f"{cur.get('Rotational speed [rpm]',0):.0f}",'rpm','轉速','#7bf7c8')
        kc(k3,f"{cur.get('Torque [Nm]',0):.1f}",'Nm','扭矩','#f7c47b')
        kc(k4,f"{cur.get('Tool wear [min]',0):.0f}",'min','磨耗','#f77b7b')
        kc(k5,f"{cur.get('Power',0)/1000:.1f}",'kW','功率','#c47bf7')

        # 折線圖
        hdf = pd.DataFrame(hist[-window:])
        xs  = hdf['timestamp'].tolist()

        fig = make_subplots(rows=3,cols=2,
            subplot_titles=('空氣/製程溫度 [K]','轉速 [rpm]',
                            '扭矩 [Nm]','刀具磨耗 [min]',
                            '功率 [kW]','預測機率'),
            vertical_spacing=0.13,horizontal_spacing=0.09)

        def al(r,c,ys,nm,col,fill=False):
            fig.add_trace(go.Scatter(
                x=xs,y=list(ys),name=nm,mode='lines',
                line=dict(color=col,width=1.8),
                fill='tozeroy' if fill else None,
            ),row=r,col=c)

        al(1,1,hdf['Air temperature [K]'],   '空氣','#7bb4f7')
        al(1,1,hdf['Process temperature [K]'],'製程','#f7c47b')
        al(1,2,hdf['Rotational speed [rpm]'], '轉速','#7bf7c8')
        al(2,1,hdf['Torque [Nm]'],            '扭矩','#c47bf7')
        al(2,2,hdf['Tool wear [min]'],        '磨耗','#f77b7b',fill=True)
        power_kw = hdf.get('Power',pd.Series([0]*len(hdf)))/1000
        al(3,1,power_kw,                      '功率','#f7a07b',fill=True)

        # 機率長條
        probs = cur.get('all_probs',[.25,.25,.25,.25])
        fig.add_trace(go.Bar(
            x=[FAILURE_LONG[i] for i in range(4)],y=probs,
            marker_color=[STATUS[i][0] for i in range(4)],
            showlegend=False,
            hovertemplate='%{x}：%{y:.3f}<extra></extra>'),
            row=3,col=2)

        # 故障區標記
        for h in hist[-window:]:
            if h['pred_cls']!=0:
                fc2 = STATUS[h['pred_cls']][0]
                for rr,cc2 in [(1,1),(1,2),(2,1),(2,2),(3,1)]:
                    fig.add_vrect(
                        x0=h['timestamp']-.5,x1=h['timestamp']+.5,
                        fillcolor=fc2,opacity=0.10,line_width=0,
                        row=rr,col=cc2)

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Noto Sans TC',color='#c0c8ff',size=9),
            margin=dict(l=6,r=6,t=44,b=6),
            height=520,showlegend=True,
            legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(size=8),
                        orientation='h',y=-0.06))
        for rr in range(1,4):
            for cc2 in range(1,3):
                fig.update_xaxes(gridcolor='#2a2a4a',row=rr,col=cc2)
                fig.update_yaxes(gridcolor='#2a2a4a',row=rr,col=cc2)
        chart_ph.plotly_chart(fig, use_container_width=True)

        # 事件日誌
        evts = [h for h in reversed(hist) if h['pred_cls']!=0][:8]
        if evts:
            rows_html = ''
            for ev in evts:
                ec = STATUS[ev['pred_cls']][0]
                rows_html += (
                    f"<tr style='border-bottom:1px solid #2a2a4a'>"
                    f"<td style='padding:.25rem .5rem;color:#8888aa;font-family:monospace'>"
                    f"t={ev['timestamp']}</td>"
                    f"<td style='padding:.25rem .5rem'>"
                    f"<span style='color:{ec}'>{FAILURE_LABELS[ev['pred_cls']][1]} "
                    f"{FAILURE_LABELS[ev['pred_cls']][0]}</span></td>"
                    f"<td style='padding:.25rem .5rem;color:#8888aa'>{ev.get('scenario','—')}</td>"
                    f"<td style='padding:.25rem .5rem;font-family:monospace;color:{ec}'>"
                    f"{ev['conf']:.1%}</td></tr>"
                )
            log_ph.markdown(
                f"<div class='kpi-card' style='padding:.8rem 1rem'>"
                f"<div class='sec-title' style='margin:.1rem 0 .4rem'>⚠️ 異常事件日誌</div>"
                f"<table style='width:100%;border-collapse:collapse;font-size:.79rem'>"
                f"<thead><tr style='color:#6666aa;font-size:.70rem'>"
                f"<th>時間步</th><th>預測故障</th><th>情境</th><th>信心度</th></tr></thead>"
                f"<tbody>{rows_html}</tbody></table></div>",
                unsafe_allow_html=True)
        else:
            log_ph.markdown(
                "<div class='kpi-card' style='color:#4cde80;padding:.7rem 1rem'>"
                "✅ 目前無異常事件</div>",
                unsafe_allow_html=True)

    # ── 播放迴圈 ─────────────────────────────────────────────────────
    if run_btn or step > 0:
        if run_btn and step < len(sim_df):
            bar = st.progress(0, text="播放中…")
            for idx in range(step, len(sim_df)):
                row  = sim_df.iloc[idx]
                X_r  = pd.DataFrame([row[FEAT_COLS].values], columns=FEAT_COLS)
                X_sc = pd.DataFrame(scaler.transform(X_r), columns=FEAT_COLS)
                prob = pred_mdl.predict_proba(X_sc)[0]
                if len(prob) < 4:
                    p4 = np.zeros(4)
                    for i,c in enumerate(pred_mdl.classes_): p4[int(c)]=prob[i]
                    prob = p4
                pcls = int(np.argmax(prob))
                entry = row.to_dict()
                entry.update({'pred_cls':pcls,'conf':float(prob[pcls]),
                              'all_probs':prob.tolist()})
                hist.append(entry)
                st.session_state.mon_hist = hist
                st.session_state.mon_step = idx + 1
                render(hist)
                bar.progress((idx+1)/len(sim_df),
                             text=f"播放中… {idx+1}/{len(sim_df)}")
                time.sleep(delay)
            bar.empty()
            st.success(f"✅ 播放完畢，共 {len(sim_df)} 步")
        else:
            render(hist)
            if step >= len(sim_df):
                st.info("播放已結束，按「重置」可重新播放。")
    else:
        render([])

    # ── 情境說明 ─────────────────────────────────────────────────────
    with st.expander("📖 模擬腳本情境說明"):
        scens = [
            ("t=0–54",    "正常運作",    "#4cde80",
             "機台正常運作，數值緩慢老化，刀具磨耗線性增加。"),
            ("t=55–75",   "散熱不良 HDF","#f97316",
             "溫差縮小至 <8.6K，轉速降至 ~1300 rpm，觸發 HDF 故障。"),
            ("t=76–94",   "恢復正常",    "#4cde80","恢復後磨耗持續累積至 ~110 min。"),
            ("t=95–115",  "功率異常 PWF","#fbbf24",
             "扭矩飆升至 ~64 Nm，功率超過 9,000W，觸發 PWF 故障。"),
            ("t=116–129", "恢復正常",    "#4cde80","磨耗累積至 ~150 min。"),
            ("t=130–152", "磨耗警示 TWF","#888888",
             "磨耗超 200 min，進入 Stage 1 換刀警示區間（非 ML 預測範圍）。"),
            ("t=163–180", "過負荷 OSF",  "#a78bfa",
             "高扭矩 × 高磨耗超過 M 型門檻（12,000 Nm·min），觸發 OSF 故障。"),
        ]
        c1,c2 = st.columns(2)
        for i,(tr,name,c,desc) in enumerate(scens):
            col = c1 if i%2==0 else c2
            col.markdown(f"""
            <div class="kpi-card" style="border-left:3px solid {c};
                         padding:.65rem .9rem;margin-bottom:.45rem">
                <div style="display:flex;justify-content:space-between;margin-bottom:.15rem">
                    <span style="color:{c};font-weight:700;font-size:.84rem">{name}</span>
                    <span style="color:#8888aa;font-size:.71rem;font-family:monospace">{tr}</span>
                </div>
                <div style="font-size:.78rem;color:#b0b0cc">{desc}</div>
            </div>""", unsafe_allow_html=True)
