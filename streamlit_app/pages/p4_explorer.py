"""p4_explorer.py — 資料探索"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd, numpy as np
from pages.loader import load_bundle, FAILURE_LONG, FAILURE_SHORT, FAIL_COLORS, sec, plotly_base

@st.cache_data
def load_raw():
    import pickle, os
    path = os.path.join(os.path.dirname(__file__),'..','models','app_bundle.pkl')
    with open(path,'rb') as f: b=pickle.load(f)
    sc = b['scaler']
    X  = np.vstack([b['X_train'],b['X_test']])
    y  = np.concatenate([b['y_train'],b['y_test']])
    X_orig = sc.inverse_transform(X)
    df = pd.DataFrame(X_orig, columns=b['feat_cols'])
    df['故障類別'] = y.astype(int)
    return df

def show():
    df = load_raw()
    st.markdown("# 🔬 資料探索")
    st.markdown("<p style='color:#8888aa;margin-top:-.4rem'>探索第二階段資料集（排除 TWF & RNF）的特徵分布與相關性</p>",
                unsafe_allow_html=True)

    t1,t2,t3,t4 = st.tabs(["⚖️ 類別分佈","📉 特徵分布","🔗 相關係數","📖 故障說明"])

    with t1:
        sec("類別分佈（SMOTE 前後）")
        counts = df['故障類別'].value_counts().sort_index()
        smote_after = {0:6546,1:6546,2:6546,3:6546}
        c1,c2 = st.columns(2)
        for col,(title,data) in zip([c1,c2],[
            ('SMOTE 前（原始）',{i:counts.get(i,0) for i in range(4)}),
            ('SMOTE 後（平衡）',smote_after),
        ]):
            lbls = [FAILURE_LONG[i] for i in sorted(data)]
            vals = [data[i] for i in sorted(data)]
            fig = go.Figure(go.Pie(
                labels=lbls,values=vals,hole=0.5,
                marker=dict(colors=FAIL_COLORS,line=dict(color='#0d0d1a',width=2)),
                textinfo='label+percent',textfont=dict(size=11),
                hovertemplate='<b>%{label}</b><br>%{value:,}筆<br>%{percent}<extra></extra>'))
            fig.add_annotation(text=f'<b>{sum(vals):,}</b><br>筆',
                x=0.5,y=0.5,showarrow=False,font=dict(size=13,color='#c0c8ff'),align='center')
            fig.update_layout(**plotly_base(height=320,
                title=dict(text=title,font=dict(size=12)),showlegend=False))
            col.plotly_chart(fig, use_container_width=True)

    with t2:
        sec("特徵分布（各故障 vs 正常）")
        num_feats = [c for c in df.columns if c!='故障類別' and 'Type' not in c]
        feat_zh = {
            'Air temperature [K]':'空氣溫度 [K]',
            'Process temperature [K]':'製程溫度 [K]',
            'Rotational speed [rpm]':'轉速 [rpm]',
            'Torque [Nm]':'扭矩 [Nm]',
            'Tool wear [min]':'刀具磨耗 [min]',
            'Power':'功率 [W]','Power wear':'功率×磨耗',
            'Temperature difference':'溫差 [K]','Temperature power':'溫差/功率',
        }
        sel = st.selectbox("選擇特徵",num_feats,
            format_func=lambda x:feat_zh.get(x,x),index=5)
        fig = go.Figure()
        for cls in sorted(df['故障類別'].unique()):
            sub = df[df['故障類別']==cls][sel].dropna()
            if len(sub)<2: continue
            fig.add_trace(go.Violin(
                y=sub,name=FAILURE_LONG[int(cls)],
                fillcolor=FAIL_COLORS[int(cls)],
                line_color=FAIL_COLORS[int(cls)],
                opacity=0.72,box_visible=True,meanline_visible=True,
                points=False if len(sub)>300 else 'all'))
        fig.update_layout(**plotly_base(
            height=400,
            yaxis=dict(title=feat_zh.get(sel,sel),gridcolor='#2a2a4a'),
            violingap=0.18,violinmode='overlay',
            legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(size=10))))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            df.groupby('故障類別')[sel].describe()[['mean','std','min','50%','max']]
              .rename(index=FAILURE_LONG)
              .rename(columns={'mean':'平均','std':'標準差','min':'最小','50%':'中位數','max':'最大'})
              .style.format('{:.2f}').background_gradient(cmap='coolwarm',axis=0),
            use_container_width=True)

    with t3:
        sec("特徵相關係數熱圖")
        num_cols = [c for c in df.columns if 'Type' not in c]
        corr = df[num_cols].corr()
        zh = {'Air temperature [K]':'空氣溫度','Process temperature [K]':'製程溫度',
              'Rotational speed [rpm]':'轉速','Torque [Nm]':'扭矩',
              'Tool wear [min]':'刀具磨耗','Power':'功率','Power wear':'功率×磨耗',
              'Temperature difference':'溫差','Temperature power':'溫差/功率','故障類別':'故障類別'}
        corr.index=corr.columns=[zh.get(c,c) for c in corr.index]
        fig=go.Figure(go.Heatmap(
            z=corr.values,x=list(corr.columns),y=list(corr.index),
            colorscale='RdBu_r',zmid=0,zmin=-1,zmax=1,
            text=[[f'{v:.2f}' for v in r] for r in corr.values],
            texttemplate='%{text}',textfont=dict(size=9),
            colorbar=dict(tickfont=dict(color='#8888aa'),thickness=14)))
        fig.update_layout(**plotly_base(
            height=500,
            xaxis=dict(tickangle=-35,tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9))))
        st.plotly_chart(fig, use_container_width=True)

    with t4:
        sec("故障類型說明")
        cards=[
            (0,'正常運作','⚙️','#4cde80',
             '機台在所有規格範圍內正常運作，無故障發生。',
             '8,183','所有值在正常範圍內','—'),
            (1,'散熱不良 HDF','🌡️','#f97316',
             '溫差（製程溫度−空氣溫度）小於 8.6K，且轉速低於 1,380 rpm 時觸發。',
             '83','溫差<8.6K 且 轉速<1380rpm','溫差 → 轉速'),
            (2,'功率異常 PWF','⚡','#fbbf24',
             '轉速×扭矩（功率）超出正常範圍 [3,500, 9,000] W 時觸發。',
             '925','功率<3500W 或 >9000W','功率'),
            (3,'過負荷 OSF','💪','#a78bfa',
             '功率×磨耗超過型別門檻：L型 11,000、M型 12,000、H型 13,000 Nm·min。',
             '297','功率×磨耗超門檻','功率×磨耗'),
        ]
        cl,cr=st.columns(2)
        for i,(cls,name,em,c,desc,cnt,trig,feat) in enumerate(cards):
            col=cl if i%2==0 else cr
            col.markdown(f"""
            <div class="kpi-card" style="border-left:4px solid {c};padding:1rem 1.2rem">
                <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.5rem">
                    <span style="font-size:1.4rem">{em}</span>
                    <div>
                        <div style="font-weight:700;color:{c};font-size:.9rem">{name}</div>
                        <div style="font-size:.70rem;color:#8888aa">類別 {cls} · {cnt} 筆樣本</div>
                    </div>
                </div>
                <div style="font-size:.82rem;color:#b0b0cc;margin-bottom:.4rem">{desc}</div>
                <div style="font-size:.74rem;color:#6666aa">
                    <b style="color:#8888aa">觸發條件：</b>{trig}<br>
                    <b style="color:#8888aa">SHAP 主特徵：</b>{feat}
                </div>
            </div>""", unsafe_allow_html=True)
