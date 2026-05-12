"""p3_comparison.py — 模型比較分析（v6）"""
import warnings; warnings.filterwarnings('ignore')
import streamlit as st
import plotly.graph_objects as go
import pandas as pd, numpy as np
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, f1_score, recall_score)
from pages.loader import (load_bundle, load_results, mcolor,
                           FAILURE_SHORT, FAILURE_LONG, FAIL_COLORS,
                           ML_MODELS, DL_MODELS, FEAT_SHORT, sec, plotly_base)

PALETTE = ['#7bf7c8', '#7bb4f7', '#f7c47b', '#c47bf7', '#f77b7b',
           '#7bf7f7', '#f7a07b', '#a0f77b', '#f77ba0', '#a07bf7']

def hrex(h, a=0.1):
    r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    return f'rgba({r},{g},{b},{a})'

@st.cache_data(show_spinner='計算預測結果…')
def get_preds():
    b    = load_bundle()
    X_te = b['X_test']
    y_te = np.array(b['y_test']).astype(int)
    out  = {}
    for nm, mdl in b['models'].items():
        try:
            yp    = mdl.predict(X_te).astype(int)
            yprob = mdl.predict_proba(X_te)
            if yprob.shape[1] < 4:
                p4 = np.zeros((len(yp), 4))
                for i, c in enumerate(mdl.classes_):
                    p4[:, int(c)] = yprob[:, i]
                yprob = p4
        except Exception:
            yp    = np.zeros(len(y_te), dtype=int)
            yprob = np.zeros((len(y_te), 4)); yprob[:, 0] = 1.
        out[nm] = (yp, yprob)
    return out, y_te

def show():
    b      = load_bundle()
    df     = load_results()
    preds, y_arr = get_preds()
    models = b['models']

    st.markdown("# 📊 模型比較分析")
    st.markdown("<p style='color:#8888aa;margin-top:-.4rem'>10 個模型的完整評估（7 個分頁）</p>",
                unsafe_allow_html=True)

    tabs = st.tabs([
        "📋 指標表", "🕸️ 雷達圖", "📉 ROC-AUC",
        "🎯 PR 曲線", "🗂️ 混淆矩陣", "🔑 特徵重要性", "⚔️ 一對一比較"
    ])

    # ── Tab 1 指標表 ─────────────────────────────────────────────────
    with tabs[0]:
        sec("完整指標比較表（Stage 2，10 個模型）")
        col_map = {'Accuracy':'準確率', 'MCC':'MCC', 'F1w':'F1(加權)',
                   'AUC':'ROC-AUC', 'Recall_macro':'Recall(macro)', 'Train_s':'訓練時間(s)'}
        avail = [c for c in col_map if c in df.columns]
        disp  = df[avail].copy().rename(columns=col_map)

        def hl(s):
            if s.name in ['MCC', 'F1(加權)', 'ROC-AUC', '準確率', 'Recall(macro)']:
                return ['background:rgba(123,247,200,.18);color:#7bf7c8;font-weight:700'
                        if v == s.max() else '' for v in s]
            return [''] * len(s)

        fmt = {c: '{:.4f}' for c in ['MCC', 'ROC-AUC', 'F1(加權)', 'Recall(macro)']}
        fmt.update({'準確率': '{:.2%}', '訓練時間(s)': '{:.1f}s'})
        st.dataframe(disp.style.apply(hl, axis=0).format(fmt).set_properties(**{
            'background': 'rgba(20,20,40,.6)', 'color': '#c0c8ff',
            'border': '1px solid #2a2a4a'}),
            use_container_width=True, height=380)

        sec("兩階段 MCC 進展")
        three = b.get('three_stage', {})
        rows  = []
        for m in list(models.keys()):
            t = three.get(m, {})
            s1 = t.get('排除RNF')
            s2 = t.get('排除TWF_RNF')
            rows.append({
                '模型': m,
                'Stage 1（排除 RNF）':     f"{s1:.4f}" if s1 else '—',
                'Stage 2（排除 TWF&RNF）': f"{s2:.4f}" if s2 else '—',
                'MCC 提升': f"+{s2-s1:.3f}" if s1 and s2 else '—',
            })
        st.dataframe(pd.DataFrame(rows).set_index('模型'),
                     use_container_width=True, height=400)

    # ── Tab 2 雷達圖 ─────────────────────────────────────────────────
    with tabs[1]:
        sec("多指標雷達圖")
        sel = st.multiselect("選擇模型", list(df.index), default=list(df.index)[:4])
        axes_opts = {c: c for c in ['MCC', 'AUC', 'F1w', 'Accuracy', 'Recall_macro']
                     if c in df.columns}
        axes_zh   = {'MCC': 'MCC', 'AUC': 'ROC-AUC', 'F1w': 'F1(加權)',
                     'Accuracy': '準確率', 'Recall_macro': 'Recall(macro)'}
        sel_ax    = st.multiselect("選擇指標", list(axes_opts.keys()),
                                   format_func=lambda x: axes_zh.get(x, x),
                                   default=list(axes_opts.keys())[:4])
        if len(sel) >= 2 and len(sel_ax) >= 3:
            cats = [axes_zh.get(a, a) for a in sel_ax] + [axes_zh.get(sel_ax[0], sel_ax[0])]
            rcols = sel_ax + [sel_ax[0]]
            fig   = go.Figure()
            for i, nm in enumerate(sel):
                c    = PALETTE[i % len(PALETTE)]
                vals = [df.loc[nm, rc] for rc in rcols]
                r, g, bv = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=cats, fill='toself', name=nm,
                    line=dict(color=c, width=2.5),
                    fillcolor=f'rgba({r},{g},{bv},0.1)'))
            rng_min = max(0.5, df[list(axes_opts.values())].min().min() - 0.05)
            fig.update_layout(**plotly_base(
                height=480,
                polar=dict(
                    bgcolor='rgba(14,14,30,.9)',
                    radialaxis=dict(visible=True, range=[rng_min, 1.0],
                        gridcolor='#2a2a4a', tickfont=dict(size=9, color='#8888aa')),
                    angularaxis=dict(gridcolor='#2a2a4a', tickfont=dict(size=11)),
                ),
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10),
                            orientation='h', y=-0.08),
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("請選擇 ≥ 2 個模型且 ≥ 3 個指標。")

    # ── Tab 3 ROC ────────────────────────────────────────────────────
    with tabs[2]:
        sec("ROC-AUC 曲線（One-vs-Rest）")
        cl, cr = st.columns([1, 2])
        with cl:
            sel_roc = st.multiselect("模型", list(models.keys()),
                default=list(models.keys())[:3], key='roc_m')
            cls_opt = [f'類別 {i}：{FAILURE_LONG[i]}' for i in range(4)]
            sel_cls = st.selectbox("故障類別", cls_opt, key='roc_c')
            ci      = int(sel_cls.split('：')[0].split()[-1])
        with cr:
            fig = go.Figure()
            fig.add_shape(type='line', x0=0, x1=1, y0=0, y1=1,
                line=dict(color='#444466', dash='dot', width=1))
            for i, nm in enumerate(sel_roc):
                if nm not in preds: continue
                _, yprob = preds[nm]
                yb = (y_arr == ci).astype(int)
                fp, tp, _ = roc_curve(yb, yprob[:, ci])
                av = auc(fp, tp); c = PALETTE[i % len(PALETTE)]
                fig.add_trace(go.Scatter(x=fp, y=tp, name=f'{nm}（{av:.3f}）',
                    line=dict(color=c, width=2.5),
                    fill='tozeroy', fillcolor=hrex(c, .04)))
            fig.update_layout(**plotly_base(
                height=400,
                xaxis=dict(title='FPR', gridcolor='#2a2a4a'),
                yaxis=dict(title='TPR', gridcolor='#2a2a4a'),
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
            ))
            st.plotly_chart(fig, use_container_width=True)

        sec("AUC 熱圖（各模型 × 各故障類別）")
        auc_rows = []
        for nm in models:
            if nm not in preds: continue
            _, yprob = preds[nm]
            row = []
            for c in range(4):
                yb = (y_arr == c).astype(int)
                if yb.sum() == 0: row.append(np.nan); continue
                fp, tp, _ = roc_curve(yb, yprob[:, c])
                row.append(round(auc(fp, tp), 4))
            auc_rows.append(row)
        adf = pd.DataFrame(auc_rows, index=list(models.keys()),
                           columns=[FAILURE_SHORT[i] for i in range(4)])
        fig_h = go.Figure(go.Heatmap(
            z=adf.values, x=list(adf.columns), y=list(adf.index),
            colorscale=[[0, '#1a1a3a'], [.5, 'rgba(74,108,247,.8)'], [1, 'rgba(123,247,200,1)']],
            zmin=.85, zmax=1.,
            text=[[f'{v:.4f}' for v in r] for r in adf.values],
            texttemplate='%{text}', textfont=dict(size=10),
            colorbar=dict(tickfont=dict(color='#8888aa'), thickness=12)))
        fig_h.update_layout(**plotly_base(
            height=360,
            xaxis=dict(title='故障類別'),
            yaxis=dict(title='', tickfont=dict(size=10)),
        ))
        st.plotly_chart(fig_h, use_container_width=True)

    # ── Tab 4 PR ─────────────────────────────────────────────────────
    with tabs[3]:
        sec("精確率-召回率曲線")
        pl, pr = st.columns([1, 2])
        with pl:
            sel_pr = st.multiselect("模型", list(models.keys()),
                default=list(models.keys())[:3], key='pr_m')
            sel_pc = st.selectbox("故障類別",
                [f'類別 {i}：{FAILURE_LONG[i]}' for i in range(4)],
                index=1, key='pr_c')
            pci = int(sel_pc.split('：')[0].split()[-1])
        with pr:
            prev = (y_arr == pci).mean()
            fig  = go.Figure()
            fig.add_shape(type='line', x0=0, x1=1, y0=prev, y1=prev,
                line=dict(color='#444466', dash='dot'))
            fig.add_annotation(x=.8, y=prev+.04,
                text=f'隨機基準={prev:.3f}', showarrow=False,
                font=dict(size=9, color='#666688'))
            for i, nm in enumerate(sel_pr):
                if nm not in preds: continue
                _, yprob = preds[nm]
                yb = (y_arr == pci).astype(int)
                pv, rv, _ = precision_recall_curve(yb, yprob[:, pci])
                ap = average_precision_score(yb, yprob[:, pci])
                c  = PALETTE[i % len(PALETTE)]
                fig.add_trace(go.Scatter(x=rv, y=pv, name=f'{nm}（AP={ap:.3f}）',
                    line=dict(color=c, width=2.5),
                    fill='tozeroy', fillcolor=hrex(c, .05)))
            fig.update_layout(**plotly_base(
                height=380,
                xaxis=dict(title='Recall', gridcolor='#2a2a4a'),
                yaxis=dict(title='Precision', gridcolor='#2a2a4a'),
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
            ))
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5 混淆矩陣 ───────────────────────────────────────────────
    with tabs[4]:
        sec("混淆矩陣")
        avail    = [n for n in df.index if n in models]
        col_sel, col_mode = st.columns([2, 1])
        with col_sel:
            sel_cm = st.selectbox("選擇模型", avail,
                index=avail.index('LightGBM') if 'LightGBM' in avail else 0)
        with col_mode:
            norm = st.radio("顯示方式", ["兩者", "計數", "正規化"], horizontal=True)

        yp_cm, _ = preds.get(sel_cm, (np.zeros(len(y_arr), dtype=int), None))
        cm   = confusion_matrix(y_arr, yp_cm, labels=list(range(4)))
        cm_n = np.divide(cm.astype(float), cm.sum(axis=1, keepdims=True),
                         where=cm.sum(axis=1, keepdims=True) != 0)
        lbs  = [FAILURE_SHORT[i] for i in range(4)]

        def cm_fig(mat, title, col_h, fmt_s):
            fig = go.Figure(go.Heatmap(
                z=mat, x=lbs, y=lbs,
                colorscale=[[0, 'rgba(26,26,46,1)'], [.5, 'rgba(74,108,247,.6)'],
                            [1, 'rgba(123,247,200,1)']],
                text=[[fmt_s.format(v) for v in r] for r in mat],
                texttemplate='%{text}', textfont=dict(size=12), showscale=True,
                colorbar=dict(tickfont=dict(color='#8888aa'), thickness=12)))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Noto Sans TC', color='#c0c8ff'),
                margin=dict(l=8, r=8, t=44, b=8),
                title=dict(text=title, font=dict(size=13, color=col_h)),
                height=360,
                xaxis=dict(title='預測', side='bottom'),
                yaxis=dict(title='實際', autorange='reversed'))
            return fig

        mcc_v = df.loc[sel_cm, 'MCC']
        if norm == "兩者":
            cc1, cc2 = st.columns(2)
            cc1.plotly_chart(cm_fig(cm,   f"原始計數  (MCC={mcc_v:.4f})", '#7bf7c8', '{:.0f}'),
                use_container_width=True)
            cc2.plotly_chart(cm_fig(cm_n, "列正規化（各類召回率）",       '#7bb4f7', '{:.2f}'),
                use_container_width=True)
        elif norm == "計數":
            st.plotly_chart(cm_fig(cm,   f"原始計數  (MCC={mcc_v:.4f})", '#7bf7c8', '{:.0f}'),
                use_container_width=True)
        else:
            st.plotly_chart(cm_fig(cm_n, "列正規化（各類召回率）",        '#7bb4f7', '{:.2f}'),
                use_container_width=True)

        from sklearn.metrics import classification_report
        rpt = classification_report(y_arr, yp_cm, target_names=lbs,
                                    output_dict=True, zero_division=0)
        rdf = pd.DataFrame(rpt).T.iloc[:-3][['precision', 'recall', 'f1-score', 'support']]
        rdf['support'] = rdf['support'].astype(int)

        def hl_f1(val):
            try:
                v = float(val)
                if v >= 0.95:   return 'background-color:#0d3d1f;color:#4cde80'
                elif v >= 0.85: return 'background-color:#3d2f00;color:#fbbf24'
                else:           return 'background-color:#3d0f0f;color:#f87171'
            except: return ''

        st.dataframe(
            rdf.style
               .format({'precision': '{:.3f}', 'recall': '{:.3f}', 'f1-score': '{:.3f}'})
               .map(hl_f1, subset=["f1-score"]),
            use_container_width=True, height=220)

    # ── Tab 6 特徵重要性 ─────────────────────────────────────────────
    with tabs[5]:
        sec("特徵重要性分析")
        shap_by_class = b.get('shap_by_class', {})
        tabnet_fi     = b.get('tabnet_fi', {})
        xgb_fi        = b.get('xgb_fi', {})

        FAILURE_MAP_S2 = {1: 'HDF 散熱不良', 2: 'PWF 功率異常', 3: 'OSF 過負荷'}
        FC_COLOR       = {'HDF 散熱不良': '#f97316', 'PWF 功率異常': '#fbbf24', 'OSF 過負荷': '#a78bfa'}

        sec("XGBoost SHAP — 各故障類別 Top-3 特徵")
        if shap_by_class:
            sc1, sc2, sc3 = st.columns(3)
            for col, cls in zip([sc1, sc2, sc3], [1, 2, 3]):
                lname = FAILURE_MAP_S2[cls]
                fc    = FC_COLOR[lname]
                top3  = shap_by_class.get(cls, [])
                col.markdown(f"**{lname}**")
                for fn, sv in top3:
                    col.markdown(f"""
                    <div style='display:flex;justify-content:space-between;
                                padding:.25rem .6rem;background:rgba(255,255,255,.03);
                                border-radius:6px;margin-bottom:.2rem;font-size:.83rem'>
                        <span style='color:#c0c8ff'>{fn}</span>
                        <span style='color:{fc};font-family:monospace'>{sv:.4f}</span>
                    </div>""", unsafe_allow_html=True)

        sec("XGBoost vs TabNet 全局特徵重要性對比")
        if xgb_fi and tabnet_fi:
            fi_xgb_s = pd.Series(xgb_fi).reindex(FEAT_SHORT).fillna(0)
            fi_tn_s  = pd.Series(tabnet_fi).reindex(FEAT_SHORT).fillna(0)
            order    = fi_xgb_s.sort_values().index
            fig = go.Figure()
            fig.add_trace(go.Bar(y=order, x=fi_xgb_s[order].values,
                orientation='h', name='XGBoost', marker_color='#f7c47b', opacity=0.85))
            fig.add_trace(go.Bar(y=order, x=fi_tn_s[order].values,
                orientation='h', name='TabNet', marker_color='#5b9fd4', opacity=0.85))
            fig.update_layout(**plotly_base(
                height=400, barmode='group',
                xaxis=dict(title='重要性分數', gridcolor='#2a2a4a'),
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
            ))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("兩種不同機制的模型（樹狀 vs 稀疏注意力）給出高度一致的特徵排序，說明模型學到了物理領域知識。")

    # ── Tab 7 一對一比較 ─────────────────────────────────────────────
    with tabs[6]:
        sec("⚔️ 一對一模型比較")
        all_m = list(models.keys())
        h1c, h2c = st.columns(2)
        with h1c:
            m_a = st.selectbox("🔵 模型 A", all_m,
                index=all_m.index('LightGBM') if 'LightGBM' in all_m else 0, key='h2h_a')
        with h2c:
            m_b = st.selectbox("🔴 模型 B", all_m,
                index=all_m.index('MLP (upgraded)') if 'MLP (upgraded)' in all_m else 1, key='h2h_b')

        if m_a == m_b: st.warning("請選擇兩個不同的模型。"); return
        ypa, yproba = preds[m_a]; ypb, yprobb = preds[m_b]

        def mets(yp, yprob):
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_arr, yp)
            mcc = matthews_corrcoef(y_arr, yp)
            f1w = f1_score(y_arr, yp, average='weighted', zero_division=0)
            aucs = []
            for c in range(4):
                yb = (y_arr == c).astype(int)
                if yb.sum() == 0: continue
                fp, tp, _ = roc_curve(yb, yprob[:, c])
                aucs.append(auc(fp, tp))
            return acc, mcc, f1w, float(np.mean(aucs))

        aa, ma, fa, ua = mets(ypa, yproba)
        ab, mb, fb, ub = mets(ypb, yprobb)

        sec("指標計分板")
        for metric, va, vb, fmt in [
            ('準確率', aa, ab, '{:.2%}'), ('MCC', ma, mb, '{:.4f}'),
            ('F1（加權）', fa, fb, '{:.4f}'), ('ROC-AUC', ua, ub, '{:.4f}'),
        ]:
            cols = st.columns([2, 1.8, 1.8, .6, 1])
            win  = '🔵' if va > vb else '🔴' if vb > va else '🟡'
            cols[0].markdown(f"<span style='color:#c0c8ff'>{metric}</span>", unsafe_allow_html=True)
            cols[1].markdown(f"<code style='color:#7bb4f7'>{fmt.format(va)}</code>", unsafe_allow_html=True)
            cols[2].markdown(f"<code style='color:#f77b7b'>{fmt.format(vb)}</code>", unsafe_allow_html=True)
            cols[3].markdown(win)
            cols[4].markdown(f"<span style='color:#8888aa;font-size:.82rem'>{abs(va-vb):.4f}</span>",
                             unsafe_allow_html=True)

        sec("混淆矩陣並排")
        cma = confusion_matrix(y_arr, ypa, labels=list(range(4)))
        cmb = confusion_matrix(y_arr, ypb, labels=list(range(4)))
        lbs_s = [FAILURE_SHORT[i] for i in range(4)]

        def mini_cm(mat, title, ch):
            fig = go.Figure(go.Heatmap(
                z=mat, x=lbs_s, y=lbs_s,
                colorscale=[[0, '#1a1a3a'], [1, ch]], showscale=False,
                text=[[str(int(v)) for v in r] for r in mat],
                texttemplate='%{text}', textfont=dict(size=12)))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Noto Sans TC', color='#c0c8ff'),
                margin=dict(l=5, r=5, t=40, b=5),
                title=dict(text=title, font=dict(size=12, color=ch)), height=320,
                xaxis=dict(title='預測', side='bottom', tickfont=dict(size=9)),
                yaxis=dict(title='實際', autorange='reversed', tickfont=dict(size=9)))
            return fig

        cc1, cc2 = st.columns(2)
        cc1.plotly_chart(mini_cm(cma, f'🔵 {m_a}', '#7bb4f7'), use_container_width=True)
        cc2.plotly_chart(mini_cm(cmb, f'🔴 {m_b}', '#f77b7b'), use_container_width=True)

        sec("樣本一致性分析")
        both_ok  = ((ypa == y_arr) & (ypb == y_arr)).sum()
        a_only   = ((ypa == y_arr) & (ypb != y_arr)).sum()
        b_only   = ((ypb == y_arr) & (ypa != y_arr)).sum()
        bad_both = ((ypa != y_arr) & (ypb != y_arr)).sum()
        n = len(y_arr)
        ag1, ag2, ag3, ag4 = st.columns(4)
        for col, lbl, cnt, c in zip([ag1, ag2, ag3, ag4],
            ['兩者皆正確', '僅 A 正確', '僅 B 正確', '兩者皆錯'],
            [both_ok, a_only, b_only, bad_both],
            ['#7bf7c8', '#7bb4f7', '#f77b7b', '#f87171']):
            col.markdown(f"""
            <div class="kpi-card" style="border-left:3px solid {c};padding:.8rem 1rem">
                <div style="font-family:monospace;font-size:1.4rem;color:{c}">{cnt}</div>
                <div style="font-size:.72rem;color:#8888aa">{lbl}<br>{cnt/n:.1%}</div>
            </div>""", unsafe_allow_html=True)
