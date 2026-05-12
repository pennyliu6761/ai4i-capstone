"""loader.py — 共用常數與資料載入（v6）"""
import os, pickle
import streamlit as st
import pandas as pd
import numpy as np

_DIR    = os.path.dirname(__file__)
BUNDLE  = os.path.join(_DIR, '..', 'models', 'app_bundle.pkl')
SIM_CSV = os.path.join(_DIR, '..', 'models', 'simulation_data.csv')
BAT_CSV = os.path.join(_DIR, '..', 'models', 'batch_test.csv')

# ── 故障類別定義（Stage 2：0=正常, 1=HDF, 2=PWF, 3=OSF）────────────
FAILURE_LABELS = {
    0: ('正常運作',      '🟢', '#4cde80'),
    1: ('散熱不良 HDF', '🔴', '#f97316'),
    2: ('功率異常 PWF', '🟠', '#fbbf24'),
    3: ('過負荷 OSF',   '🟡', '#a78bfa'),
}
FAILURE_LONG  = {0: '正常運作', 1: '散熱不良', 2: '功率異常', 3: '過負荷'}
FAILURE_SHORT = {0: '正常', 1: 'HDF', 2: 'PWF', 3: 'OSF'}
PHYSICS = {
    0: '所有感測器值在正常範圍內',
    1: '溫差 < 8.6K 且 轉速 < 1,380 rpm',
    2: '功率 < 3,500W 或 > 9,000W',
    3: '扭矩 × 磨耗超過型別門檻（L:11,000 / M:12,000 / H:13,000）',
}
FAIL_COLORS = ['#4cde80', '#f97316', '#fbbf24', '#a78bfa']

# ── 模型分組（Stage 2，10 個模型）────────────────────────────────────
ML_MODELS = ['KNN', 'Decision Tree', 'Random Forest',
             'Gradient Boosting', 'XGBoost', 'LightGBM']
DL_MODELS = ['MLP (original)', 'MLP (upgraded)', 'TabNet',
             'Stacking (XGB+LGBM→RF)']

MODEL_COLOR = {
    'KNN':                    '#aaaacc',
    'Decision Tree':          '#a0c4ff',
    'Random Forest':          '#7bb4f7',
    'Gradient Boosting':      '#5b9fd4',
    'XGBoost':                '#f7c47b',
    'LightGBM':               '#7bf7c8',
    'MLP (original)':         '#d4a0f7',
    'MLP (upgraded)':         '#c47bf7',
    'TabNet':                 '#5b9fd4',
    'Stacking (XGB+LGBM→RF)': '#f77b7b',
}

def mcolor(name):
    return MODEL_COLOR.get(name, '#888888')

# FEAT_COLS_SAFE 順序對應的中文短名
FEAT_SHORT = ['空氣溫度', '製程溫度', '轉速', '扭矩', '刀具磨耗',
              '功率', '功率×磨耗', '溫差', '溫差/功率', '類型L', '類型M']

FEAT_ZH = {
    'Air temperature K':      '空氣溫度',
    'Process temperature K':  '製程溫度',
    'Rotational speed rpm':   '轉速',
    'Torque Nm':              '扭矩',
    'Tool wear min':          '刀具磨耗',
    'Power':                  '功率',
    'Power wear':             '功率×磨耗',
    'Temperature difference': '溫差',
    'Temperature power':      '溫差/功率',
    'Type_L':                 '類型L',
    'Type_M':                 '類型M',
}

@st.cache_resource(show_spinner='載入模型中…')
def load_bundle():
    with open(BUNDLE, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_results():
    b = load_bundle()
    return b['results']

@st.cache_data
def load_sim():
    return pd.read_csv(SIM_CSV)

def kpi_card(col, val, label, color='#7bf7c8'):
    col.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-val' style='color:{color}'>{val}</div>
        <div class='kpi-lbl'>{label}</div>
    </div>""", unsafe_allow_html=True)

def sec(title):
    st.markdown(f"<div class='sec-title'>{title}</div>",
                unsafe_allow_html=True)

def plotly_base(margin=None, **kw):
    m = margin or dict(l=8, r=8, t=40, b=8)
    d = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Noto Sans TC', color='#c0c8ff'),
        margin=m,
    )
    d.update(kw)
    return d
