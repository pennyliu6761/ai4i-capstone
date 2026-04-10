# ⚙️ AI4I Predictive Maintenance — Streamlit App

A full-featured interactive dashboard for the AI4I 2020 milling machine failure classification project.

## 📁 Project Structure

```
streamlit_app/
├── app.py                  # Main entry point + navigation
├── requirements.txt
├── README.md
├── models/
│   ├── app_bundle.pkl      # All 12 trained models + scaler + data
│   └── results.csv         # Pre-computed evaluation metrics
└── pages/
    ├── __init__.py
    ├── loader.py           # Shared data loader (cached)
    ├── p1_overview.py      # Overview Dashboard
    ├── p2_predictor.py     # Live Predictor
    ├── p3_comparison.py    # Model Comparison
    ├── p4_explorer.py      # Data Explorer
    └── p5_methodology.py   # Research Methodology
```

## 🚀 Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-team>/<repo-name>.git
cd <repo-name>

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

## ☁️ Deploy to Streamlit Community Cloud (Free)

1. Push this folder to a **public** GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo
4. Set **Main file path** = `app.py`
5. Click **Deploy** — live URL in ~2 minutes

> ⚠️ `models/app_bundle.pkl` is ~200 MB. Use **Git LFS** for large files:
> ```bash
> git lfs install
> git lfs track "*.pkl"
> git add .gitattributes
> git commit -m "track pkl with LFS"
> ```

## 📊 Pages

| Page | Description |
|---|---|
| 🏠 Overview Dashboard | KPI cards, MCC leaderboard, radar top-4, efficiency bubble chart |
| 🔮 Live Predictor | Slider inputs → all 12 models predict simultaneously with confidence |
| 📊 Model Comparison | Metrics table, radar chart, confusion matrices, feature importance |
| 🔬 Data Explorer | Class imbalance (SMOTE before/after), violin plots, correlation heatmap, failure cards |
| 📖 Methodology | Step-by-step pipeline, key bugs fixed, lessons learned |

## 🏆 Best Results

| Model | MCC | AUC | Train Time |
|---|---|---|---|
| XGBoost (Tuned) | **0.7821** | **0.9834** | 198s |
| Stacking (XGB+LGBM, meta=RF) | 0.7732 | 0.8981 | 10s |
| Random Forest | 0.7695 | 0.9402 | 5s |
