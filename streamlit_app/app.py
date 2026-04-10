import streamlit as st

st.set_page_config(
    page_title="AI4I 兩階段預測性維護系統",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{ font-family:'Noto Sans TC',sans-serif; }
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0f0f1a 0%,#1a1a2e 100%);
    border-right:1px solid #2a2a4a;
}
[data-testid="stSidebar"] *{ color:#e0e0f0 !important; }
.stApp{ background:#0d0d1a; }
.main .block-container{ padding:1.8rem 2.2rem 3rem; max-width:1420px; }
.kpi-card{
    background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
    border:1px solid #2a2a4a; border-radius:14px;
    padding:1.1rem 1.4rem; margin-bottom:.9rem;
    transition:border-color .2s;
}
.kpi-card:hover{ border-color:#4a6cf7; }
.kpi-val{ font-family:'JetBrains Mono',monospace; font-size:1.9rem; font-weight:700; line-height:1.1; }
.kpi-lbl{ font-size:.72rem; text-transform:uppercase; letter-spacing:.10em; color:#8888aa; margin-top:.2rem; }
.sec-title{
    font-size:1.15rem; font-weight:700; color:#c0c8ff;
    border-left:4px solid #4a6cf7; padding-left:.7rem;
    margin:1.4rem 0 .9rem;
}
[data-testid="stPlotlyChart"]>div{ border-radius:12px; overflow:hidden; border:1px solid #2a2a4a; }
h1,h2,h3{ color:#e0e0ff !important; }
p,li{ color:#b0b0cc; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ AI4I 預測性維護\n#### 兩階段決策支援系統")
    st.markdown("---")
    st.markdown("""
    <div style='font-size:.76rem;color:#f7c47b;background:#1a1400;border:1px solid #f7c47b;
                border-radius:8px;padding:.6rem .9rem;margin-bottom:.7rem;line-height:1.9'>
    <b>第一階段：EDA 規則決策</b><br>
    RNF → 排除，不建模<br>
    TWF → 磨耗>200min 換刀警示
    </div>
    <div style='font-size:.76rem;color:#4cde80;background:#0a1a0f;border:1px solid #4cde80;
                border-radius:8px;padding:.6rem .9rem;margin-bottom:1rem;line-height:1.9'>
    <b>第二階段：ML/DL 智慧分類</b><br>
    HDF ‧ PWF ‧ OSF → 6 個模型<br>
    最佳 MCC = 0.9736
    </div>
    """, unsafe_allow_html=True)
    st.markdown("##### 📍 頁面導覽")
    page = st.radio("", [
        "🏠  總覽儀表板",
        "🔮  即時預測器",
        "📊  模型比較分析",
        "🔬  資料探索",
        "📡  即時監控看板",
        "📂  批次預測上傳",
        "📖  研究方法論",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style='font-size:.74rem;color:#6666aa;line-height:1.9'>
    <b style='color:#8888cc'>資料集</b><br>AI4I 2020 · 10,000 筆<br><br>
    <b style='color:#8888cc'>最佳模型</b><br>Random Forest · MCC=0.9736<br><br>
    <b style='color:#8888cc'>系所</b><br>工業工程與管理學系<br>畢業專題
    </div>
    """, unsafe_allow_html=True)

if   "總覽"   in page: from pages import p1_overview;    p1_overview.show()
elif "預測器" in page: from pages import p2_predictor;   p2_predictor.show()
elif "比較"   in page: from pages import p3_comparison;  p3_comparison.show()
elif "探索"   in page: from pages import p4_explorer;    p4_explorer.show()
elif "監控"   in page: from pages import p6_monitor;     p6_monitor.show()
elif "批次"   in page: from pages import p7_batch;       p7_batch.show()
elif "方法論" in page: from pages import p5_methodology; p5_methodology.show()
