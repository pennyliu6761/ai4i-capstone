"""p6_monitor.py — 即時監控看板（優化版，修復卡頓）

卡頓根本原因：
  1. make_subplots 每步重建 6 個子圖
  2. add_vrect 對每個故障點 × 5 個子圖 = 最多 250 次 API 呼叫
  3. hist 串列隨步數增長，每步都掃描整個歷史

修復策略：
  1. 改用 4 個獨立小圖取代 make_subplots
  2. vrect 改為背景色 shape，只計算一次
  3. 只傳窗口資料，hist 只保留窗口大小 × 2
  4. 移除每步重建 legend
"""
import warnings; warnings.filterwarnings('ignore')
import streamlit as st
import plotly.graph_objects as go
import pandas as pd, numpy as np, time
from pages.loader import (load_bundle, load_sim, FAILURE_LABELS, FAILURE_LONG, sec)

# FEAT_COLS 從 bundle 動態取得，見 show() 函式

STATUS = {
    0: ('#4cde80','#0a1a0f','🟢'),
    1: ('#f97316','#1a0f00','🔴'),
    2: ('#fbbf24','#1a1200','🟠'),
    3: ('#a78bfa','#100a1a','🟡'),
}

def _base(height=200):
    return dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Noto Sans TC', color='#c0c8ff', size=9),
        margin=dict(l=44, r=8, t=32, b=28),
        height=height, showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=8),
                    orientation='h', y=-0.25),
        xaxis=dict(gridcolor='#2a2a4a'),
        yaxis=dict(gridcolor='#2a2a4a'),
    )

def _make_shapes(window_df):
    """故障區背景色：用 shapes list（比 add_vrect 快得多）"""
    shapes = []
    for _, row in window_df[window_df['pred_cls'] != 0].iterrows():
        t = row['timestamp']
        fc = STATUS[row['pred_cls']][0]
        shapes.append(dict(
            type='rect', xref='x', yref='paper',
            x0=t-0.5, x1=t+0.5, y0=0, y1=1,
            fillcolor=fc, opacity=0.12, line_width=0,
        ))
    return shapes

def show():
    b         = load_bundle()
    scaler    = b['scaler']
    models    = b['models']
    sim_df    = load_sim()
    FEAT_COLS = b['feat_cols']   # 與 scaler 一致（SAFE 格式）

    # simulation_data.csv 欄位名是原始格式（含括號）
    # 建立對應關係：原始名 → SAFE 名
    SIM_COLS = ['Air temperature [K]', 'Process temperature [K]',
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                'Power', 'Power wear', 'Temperature difference',
                'Temperature power', 'Type_L', 'Type_M']

    st.markdown("# 📡 即時監控看板")
    st.markdown("<p style='color:#8888aa;margin-top:-.4rem'>"
                "模擬銑床從正常→散熱不良→功率異常→過負荷的完整動態過程</p>",
                unsafe_allow_html=True)

    # ── 控制列 ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1: speed  = st.selectbox("▶ 播放速度", ["0.5×","1×","2×","5×"], index=1)
    with c2: window = st.slider("視窗（筆）", 20, 80, 40, 10)
    with c3:
        mdl_names = list(models.keys())
        def_i = mdl_names.index('Random Forest') if 'Random Forest' in mdl_names else 0
        sel_m = st.selectbox("預測模型", mdl_names, index=def_i)
    with c4:
        cb1, cb2 = st.columns(2)
        run_btn = cb1.button("▶ 開始播放", use_container_width=True, type="primary")
        rst_btn = cb2.button("⏹ 重置",     use_container_width=True)

    speed_map = {"0.5×": 0.18, "1×": 0.09, "2×": 0.045, "5×": 0.018}
    delay    = speed_map[speed]
    pred_mdl = models[sel_m]

    if rst_btn:
        for k in ['mon_step','mon_hist','mon_log']:
            st.session_state.pop(k, None)
        st.rerun()

    if 'mon_step' not in st.session_state: st.session_state.mon_step = 0
    if 'mon_hist' not in st.session_state: st.session_state.mon_hist = []
    if 'mon_log'  not in st.session_state: st.session_state.mon_log  = []

    step = st.session_state.mon_step
    hist = st.session_state.mon_hist   # 只保留最近 window*2 筆（節省記憶體）
    log  = st.session_state.mon_log    # 故障事件日誌（獨立保存）

    # ── 佔位符 ───────────────────────────────────────────────────────
    banner_ph = st.empty()
    kpi_ph    = st.empty()

    # 用 2×2 格局的 4 個獨立小圖（比 make_subplots 快）
    row1_l, row1_r = st.columns(2)
    row2_l, row2_r = st.columns(2)
    ch_temp  = row1_l.empty()
    ch_rpm   = row1_r.empty()
    ch_wear  = row2_l.empty()
    ch_prob  = row2_r.empty()
    log_ph   = st.empty()

    # ── 渲染函式 ─────────────────────────────────────────────────────
    _render_counter = [0]  # mutable counter for unique keys
    def render(hist, log):
        _render_counter[0] += 1
        _rk = _render_counter[0]  # unique render key
        if not hist:
            banner_ph.info("按下「▶ 開始播放」啟動模擬（共 200 步）")
            return

        cur  = hist[-1]
        cls  = cur['pred_cls']
        fc, bg, em = STATUS[cls]

        # 橫幅
        banner_ph.markdown(
            f"<div style='background:{bg};border:2px solid {fc};border-radius:12px;"
            f"padding:.8rem 1.4rem;display:flex;align-items:center;gap:1rem'>"
            f"<span style='font-size:1.8rem'>{em}</span>"
            f"<div><div style='font-size:1.2rem;font-weight:700;color:{fc}'>"
            f"{FAILURE_LABELS[cls][0]}</div>"
            f"<div style='font-size:.78rem;color:#aaaacc'>"
            f"t={cur['timestamp']} · {cur.get('scenario','—')} · 信心度 {cur['conf']:.1%}"
            f"</div></div></div>",
            unsafe_allow_html=True)

        # KPI
        k1,k2,k3,k4,k5 = kpi_ph.columns(5)
        def kc(col, v, u, lb, c='#c0c8ff'):
            col.markdown(
                f"<div class='kpi-card' style='padding:.6rem .8rem;text-align:center'>"
                f"<div style='font-family:monospace;font-size:1.1rem;font-weight:700;color:{c}'>"
                f"{v}<span style='font-size:.65rem;color:#8888aa;margin-left:.1rem'>{u}</span></div>"
                f"<div style='font-size:.65rem;color:#8888aa'>{lb}</div></div>",
                unsafe_allow_html=True)
        kc(k1, f"{cur.get('Air temperature K',0):.1f}", 'K',   '空氣溫度', '#7bb4f7')
        kc(k2, f"{cur.get('Rotational speed rpm',0):.0f}",'rpm','轉速',    '#7bf7c8')
        kc(k3, f"{cur.get('Torque Nm',0):.1f}",          'Nm',  '扭矩',    '#f7c47b')
        kc(k4, f"{cur.get('Tool wear min',0):.0f}",       'min', '磨耗',   '#f77b7b')
        kc(k5, f"{cur.get('Power',0)/1000:.1f}",            'kW',  '功率',   '#c47bf7')

        # 窗口資料（只取最近 window 筆，不掃描全部 hist）
        wdf    = pd.DataFrame(hist)
        xs     = wdf['timestamp'].tolist()
        shapes = _make_shapes(wdf)  # 一次計算所有故障背景

        def line_fig(title, traces, shapes, height=200):
            """輕量單圖，不用 make_subplots"""
            fig = go.Figure()
            for ys, name, color, fill in traces:
                fig.add_trace(go.Scatter(
                    x=xs, y=list(ys), name=name, mode='lines',
                    line=dict(color=color, width=1.8),
                    fill='tozeroy' if fill else None,
                ))
            layout = _base(height)
            layout['title'] = dict(text=title, font=dict(size=10), x=0.02, y=0.96)
            layout['shapes'] = shapes
            fig.update_layout(**layout)
            return fig

        # 溫度圖
        ch_temp.plotly_chart(line_fig('空氣 / 製程溫度 [K]', [
            (wdf['Air temperature K'],    '空氣', '#7bb4f7', False),
            (wdf['Process temperature K'],'製程', '#f7c47b', False),
        ], shapes), use_container_width=True, key=f'ch_temp_{_rk}')

        # 轉速 + 扭矩圖（合一）
        ch_rpm.plotly_chart(line_fig('轉速 [rpm] / 扭矩 [Nm]', [
            (wdf['Rotational speed rpm'], '轉速', '#7bf7c8', False),
            (wdf['Torque Nm'],            '扭矩', '#c47bf7', False),
        ], shapes), use_container_width=True, key=f'ch_rpm_{_rk}')

        # 磨耗 + 功率圖（合一）
        pw_kw = wdf.get('Power', pd.Series([0]*len(wdf))) / 1000
        ch_wear.plotly_chart(line_fig('磨耗 [min] / 功率 [kW]', [
            (wdf['Tool wear min'], '磨耗', '#f77b7b', True),
            (pw_kw,                  '功率', '#f7a07b', False),
        ], shapes), use_container_width=True, key=f'ch_wear_{_rk}')

        # 機率長條圖
        probs = cur.get('all_probs', [.25,.25,.25,.25])
        fig_bar = go.Figure(go.Bar(
            x=[FAILURE_LONG[i] for i in range(4)], y=probs,
            marker_color=[STATUS[i][0] for i in range(4)],
            showlegend=False,
            hovertemplate='%{x}：%{y:.3f}<extra></extra>',
        ))
        layout_bar = _base(200)
        layout_bar['title'] = dict(text='預測機率', font=dict(size=10), x=0.02, y=0.96)
        layout_bar['showlegend'] = False
        fig_bar.update_layout(**layout_bar)
        ch_prob.plotly_chart(fig_bar, use_container_width=True, key=f'ch_prob_{_rk}')

        # 事件日誌（從獨立 log 列表讀取，不掃描 hist）
        if log:
            rows_html = ''
            for ev in log[-8:]:
                ec = STATUS[ev['pred_cls']][0]
                rows_html += (
                    f"<tr style='border-bottom:1px solid #2a2a4a'>"
                    f"<td style='padding:.22rem .45rem;color:#8888aa;font-family:monospace'>t={ev['t']}</td>"
                    f"<td style='padding:.22rem .45rem'><span style='color:{ec}'>"
                    f"{FAILURE_LABELS[ev['pred_cls']][1]} {FAILURE_LABELS[ev['pred_cls']][0]}</span></td>"
                    f"<td style='padding:.22rem .45rem;color:#8888aa'>{ev.get('scenario','—')}</td>"
                    f"<td style='padding:.22rem .45rem;font-family:monospace;color:{ec}'>{ev['conf']:.1%}</td></tr>"
                )
            log_ph.markdown(
                f"<div class='kpi-card' style='padding:.7rem 1rem'>"
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
                # 從 sim_df（原始欄位名）取值，轉成 numpy array 給 scaler
                raw_vals = [row[c] for c in SIM_COLS]
                X_r  = np.array([raw_vals], dtype=float)
                X_sc = pd.DataFrame(scaler.transform(X_r), columns=FEAT_COLS)
                prob = pred_mdl.predict_proba(X_sc)[0]
                if len(prob) < 4:
                    p4 = np.zeros(4)
                    for i, c in enumerate(pred_mdl.classes_): p4[int(c)] = prob[i]
                    prob = p4
                pcls = int(np.argmax(prob))

                entry = {
                    'timestamp': int(row['timestamp']),
                    'scenario':  row.get('scenario', '—'),
                    'pred_cls':  pcls,
                    'conf':      float(prob[pcls]),
                    'all_probs': prob.tolist(),
                    'air_temp':  float(row.get('Air temperature [K]', 0)),
                    'proc_temp': float(row.get('Process temperature [K]', 0)),
                    'rpm':       float(row.get('Rotational speed [rpm]', 0)),
                    'torque':    float(row.get('Torque [Nm]', 0)),
                    'wear':      float(row.get('Tool wear [min]', 0)),
                    'Power':                   float(row.get('Power', 0)),
                }
                # 只保留窗口大小 × 2 的歷史（防止串列無限增長）
                hist.append(entry)
                if len(hist) > window * 2:
                    hist = hist[-window:]

                # 故障事件單獨記錄
                if pcls != 0:
                    log.append({'t': entry['timestamp'], 'pred_cls': pcls,
                                'conf': entry['conf'], 'scenario': entry['scenario']})
                    if len(log) > 50: log = log[-50:]

                st.session_state.mon_hist = hist
                st.session_state.mon_log  = log
                st.session_state.mon_step = idx + 1

                render(hist[-window:], list(reversed(log)))
                bar.progress((idx+1)/len(sim_df),
                             text=f"播放中… {idx+1}/{len(sim_df)}")
                time.sleep(delay)

            bar.empty()
            st.success(f"✅ 播放完畢，共 {len(sim_df)} 步")

        else:
            render(hist[-window:] if hist else [], list(reversed(log)))
            if step >= len(sim_df):
                st.info("播放已結束，按「重置」可重新播放。")
    else:
        render([], [])

    # ── 情境說明 ─────────────────────────────────────────────────────
    with st.expander("📖 模擬腳本情境說明"):
        scens = [
            ("t=0–54",    "正常運作",    "#4cde80",
             "機台正常運作，數值緩慢老化，刀具磨耗線性增加。"),
            ("t=55–75",   "散熱不良 HDF","#f97316",
             "溫差縮小至 <8.6K，轉速降至 ~1300 rpm，觸發 HDF 故障。"),
            ("t=76–94",   "恢復正常",    "#4cde80", "恢復後磨耗持續累積至 ~110 min。"),
            ("t=95–115",  "功率異常 PWF","#fbbf24",
             "扭矩飆升至 ~64 Nm，功率超過 9,000W，觸發 PWF 故障。"),
            ("t=116–129", "恢復正常",    "#4cde80", "磨耗累積至 ~150 min。"),
            ("t=130–152", "磨耗警示 TWF","#888888",
             "磨耗超 200 min，進入 Stage 1 換刀警示區（非 ML 預測範圍）。"),
            ("t=163–180", "過負荷 OSF",  "#a78bfa",
             "高扭矩 × 高磨耗超過 M 型門檻（12,000 Nm·min），觸發 OSF 故障。"),
        ]
        c1, c2 = st.columns(2)
        for i, (tr, name, c, desc) in enumerate(scens):
            col = c1 if i % 2 == 0 else c2
            col.markdown(
                f"<div class='kpi-card' style='border-left:3px solid {c};"
                f"padding:.6rem .9rem;margin-bottom:.4rem'>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:.1rem'>"
                f"<span style='color:{c};font-weight:700;font-size:.84rem'>{name}</span>"
                f"<span style='color:#8888aa;font-size:.71rem;font-family:monospace'>{tr}</span></div>"
                f"<div style='font-size:.78rem;color:#b0b0cc'>{desc}</div></div>",
                unsafe_allow_html=True)
