import streamlit as st
import pickle
import pandas as pd
import shap
import numpy as np
from pathlib import Path
from streamlit_shap import st_shap

st.set_page_config(page_title="Prediction Engine — EduPredict", page_icon="⬡", layout="wide")

# ── Global Design System ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: #03050f;
    background-image:
        radial-gradient(ellipse 70% 55% at 15% 5%,  rgba(79,70,229,0.22) 0%, transparent 60%),
        radial-gradient(ellipse 55% 45% at 85% 85%, rgba(139,92,246,0.18) 0%, transparent 55%),
        radial-gradient(ellipse 40% 35% at 50% 50%, rgba(236,72,153,0.06) 0%, transparent 50%);
    min-height: 100vh;
}

[data-testid="stSidebar"] {
    background: rgba(6,8,24,0.9) !important;
    border-right: 1px solid rgba(99,102,241,0.18) !important;
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
}

/* Nav */
[data-testid="stSidebarNav"] a { border-radius: 10px !important; margin: 2px 0 !important; transition: background 0.2s !important; }
[data-testid="stSidebarNav"] a:hover { background: rgba(99,102,241,0.15) !important; }
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: linear-gradient(90deg, rgba(99,102,241,0.25), rgba(168,85,247,0.15)) !important;
    border-left: 3px solid #818cf8 !important;
}

h1, h2, h3, h4 { color: #f1f5f9 !important; }
p { color: #94a3b8; line-height: 1.7; }

/* ── Page header ─────────────────────────── */
.page-header {
    background: linear-gradient(135deg, rgba(79,70,229,0.18) 0%, rgba(124,58,237,0.10) 50%, rgba(6,8,24,0) 100%);
    border: 1px solid rgba(99,102,241,0.28);
    border-top: 1px solid rgba(129,140,248,0.4);
    border-radius: 20px;
    padding: 2rem 2.4rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 0 rgba(129,140,248,0.15) inset, 0 20px 60px rgba(79,70,229,0.08);
}
.page-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(168,85,247,0.15), transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.page-header-tag {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 0.6rem;
}
.page-header h1 {
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    color: #f1f5f9 !important;
    margin: 0 0 0.5rem 0 !important;
    letter-spacing: -0.02em;
}
.page-header p { color: #94a3b8; font-size: 0.9rem; margin: 0; }

/* ── Dataset selector pill ───────────────── */
.ds-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 1.2rem;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.pill-uci   { background: rgba(52,211,153,0.12); border: 1px solid rgba(52,211,153,0.35); color: #34d399; }
.pill-xapi  { background: rgba(96,165,250,0.12); border: 1px solid rgba(96,165,250,0.35); color: #60a5fa; }
.pill-oulad { background: rgba(251,146,60,0.12); border: 1px solid rgba(251,146,60,0.35); color: #fb923c; }

/* ── Result card ─────────────────────────── */
.result-card {
    background: linear-gradient(160deg, rgba(79,70,229,0.07) 0%, rgba(15,10,40,0.6) 60%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 20px;
    padding: 1.8rem;
    height: 100%;
    box-shadow: 0 0 0 0.5px rgba(99,102,241,0.1) inset;
}
.result-card-title {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.result-card-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.3), transparent);
}

/* ── Outcome display ─────────────────────── */
.outcome-display {
    text-align: center;
    padding: 2rem 1rem 1.6rem;
    border-radius: 18px;
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}
.outcome-display::before {
    content: '';
    position: absolute;
    top: 0; left: 50%; transform: translateX(-50%);
    width: 60%; height: 1px;
    background: currentColor;
    opacity: 0.3;
}
/* .outcome-emoji replaced by .outcome-icon */
.outcome-label  { font-size: 1.65rem; font-weight: 900; display: block; margin-bottom: 0.35rem; letter-spacing: -0.02em; }
.outcome-sub    { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.6; }
.out-high        { background: rgba(52,211,153,0.1);  border: 1px solid rgba(52,211,153,0.3);  box-shadow: 0 8px 30px rgba(52,211,153,0.08); }
.out-high .outcome-label  { color: #34d399; text-shadow: 0 0 20px rgba(52,211,153,0.4); }
.out-medium      { background: rgba(96,165,250,0.1);  border: 1px solid rgba(96,165,250,0.3);  box-shadow: 0 8px 30px rgba(96,165,250,0.08); }
.out-medium .outcome-label { color: #60a5fa; text-shadow: 0 0 20px rgba(96,165,250,0.4); }
.out-low         { background: rgba(251,113,133,0.1); border: 1px solid rgba(251,113,133,0.3); box-shadow: 0 8px 30px rgba(251,113,133,0.08); }
.out-low .outcome-label   { color: #fb7185; text-shadow: 0 0 20px rgba(251,113,133,0.4); }
.out-distinction { background: rgba(251,191,36,0.1);  border: 1px solid rgba(251,191,36,0.3);  box-shadow: 0 8px 30px rgba(251,191,36,0.08); }
.out-distinction .outcome-label { color: #fbbf24; text-shadow: 0 0 20px rgba(251,191,36,0.4); }

/* ── Confidence bars ─────────────────────── */
.conf-section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.8rem;
}
.conf-item { margin-bottom: 0.7rem; }
.conf-top {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.3rem;
}
.conf-name { color: #94a3b8; font-size: 0.8rem; font-weight: 500; }
.conf-pct  { color: #e2e8f0; font-size: 0.8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.conf-track {
    height: 7px;
    background: rgba(255,255,255,0.08);
    border-radius: 100px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #4f46e5, #a855f7);
    transition: width 0.5s ease;
}

/* ── SHAP panel ─────────────────────────── */
.shap-panel {
    background: rgba(10,8,35,0.7);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 20px;
    padding: 1.8rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.shap-panel-title {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.shap-panel-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.3), transparent);
}
.shap-legend {
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
}
.legend-item { display: flex; align-items: center; gap: 0.4rem; font-size: 0.78rem; color: #64748b; }
.legend-dot  { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }


/* ── Icon system ─────────────────────────── */
.icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.icon svg {
    width: var(--icon-sz, 14px);
    height: var(--icon-sz, 14px);
    stroke: currentColor;
    stroke-width: 1.75;
    stroke-linecap: round;
    stroke-linejoin: round;
    fill: none;
    display: block;
}

/* ── Outcome icon wrapper ───────────────── */
.outcome-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 52px; height: 52px;
    border-radius: 14px;
    margin: 0 auto 0.9rem;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
}
.outcome-icon svg {
    width: 24px; height: 24px;
    stroke: currentColor;
    stroke-width: 1.75;
    stroke-linecap: round;
    stroke-linejoin: round;
    fill: none;
}
.out-high   .outcome-icon { background: rgba(52,211,153,0.12); border-color: rgba(52,211,153,0.25); color: #34d399; }
.out-medium .outcome-icon { background: rgba(96,165,250,0.12); border-color: rgba(96,165,250,0.25); color: #60a5fa; }
.out-low    .outcome-icon { background: rgba(251,113,133,0.12); border-color: rgba(251,113,133,0.25); color: #fb7185; }
.out-distinction .outcome-icon { background: rgba(251,191,36,0.12); border-color: rgba(251,191,36,0.25); color: #fbbf24; }

/* ── Conf name icon ─────────────────────── */
.conf-name-icon {
    display: inline-flex; align-items: center; gap: 0.4rem;
}
.conf-name-icon svg {
    width: 12px; height: 12px;
    stroke: currentColor; stroke-width: 2;
    stroke-linecap: round; stroke-linejoin: round; fill: none;
    flex-shrink: 0;
}

/* ── Section label icon ─────────────────── */
.section-icon {
    display: inline-flex; align-items: center;
    margin-right: 0.55rem;
}
.section-icon svg {
    width: 13px; height: 13px;
    stroke: #818cf8; stroke-width: 2;
    stroke-linecap: round; stroke-linejoin: round; fill: none;
}

/* ── Legend swatch ───────────────────────── */
.legend-item { display: flex; align-items: center; gap: 0.5rem; font-size: 0.78rem; color: #64748b; }
.legend-swatch { width: 8px; height: 8px; border-radius: 2px; flex-shrink: 0; }

/* ── Header tag icon ─────────────────────── */
.header-icon {
    display: inline-flex; align-items: center; gap: 0.45rem;
}
.header-icon svg {
    width: 12px; height: 12px;
    stroke: #818cf8; stroke-width: 2.25;
    stroke-linecap: round; stroke-linejoin: round; fill: none;
}

/* ── Streamlit widget overrides ─────────────────────────── */
.stSelectbox label, .stSlider label, .stMultiSelect label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
.stProgress > div > div { background: linear-gradient(90deg, #4f46e5, #a855f7) !important; }
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Sidebar expander */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stExpander"] summary { color: #c7d2fe !important; font-weight: 700 !important; font-size: 0.86rem !important; }
[data-testid="stExpander"] > div > div { border-top: 1px solid rgba(99,102,241,0.12) !important; }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = Path("models")

@st.cache_resource
def load_model_artifacts(dataset_name):
    with open(MODELS_DIR / f"{dataset_name}_model.pkl",    "rb") as f: model    = pickle.load(f)
    with open(MODELS_DIR / f"{dataset_name}_explainer.pkl","rb") as f: explainer= pickle.load(f)
    with open(MODELS_DIR / f"{dataset_name}_pipeline.pkl", "rb") as f: pipeline = pickle.load(f)
    return model, explainer, pipeline

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-header-tag"><span class="header-icon"><svg viewBox='0 0 24 24'><polygon points='13 2 3 14 12 14 11 22 21 10 12 10 13 2'/></svg>Interactive Prediction</span></div>
    <h1>Prediction Engine</h1>
    <p>Configure a student profile in the sidebar — the <strong>XGBoost-centred</strong> predictor
       (stacked ensemble for classification) returns the performance outcome in real time, with a
       SHAP force plot explaining each decision.</p>
</div>
""", unsafe_allow_html=True)

# ── Dataset selector ──────────────────────────────────────────────────────────
col_sel, _ = st.columns([2, 3])
with col_sel:
    dataset_choice = st.selectbox(
        "**Analytical Context**",
        ("UCI (High School Grades)", "xAPI (K-12 Online Learning)", "UCI Dropout & Academic Success"),
        help="Each dataset uses a separately trained stacking ensemble.",
    )

dataset_map = {
    "UCI (High School Grades)":          ("uci",     "regression",     "pill-uci",     "REG · Final Grade G3"),
    "xAPI (K-12 Online Learning)":       ("xapi",    "classification", "pill-xapi",    "CLF · Low / Medium / High"),
    "UCI Dropout & Academic Success":    ("dropout", "classification", "pill-oulad",   "CLF · Dropout / Enrolled / Graduate"),
}
dataset_key, task_type, pill_cls, pill_label = dataset_map[dataset_choice]
st.markdown(f'<span class="ds-pill {pill_cls}">{pill_label}</span>', unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
try:
    model, explainer, pipeline = load_model_artifacts(dataset_key)
except FileNotFoundError:
    st.error("Model artifacts not found. Run: `python src/train.py`")
    st.stop()

raw_features = pipeline['raw_features']
cat_cols     = pipeline['cat_cols']
num_cols     = pipeline['num_cols']
cat_classes  = pipeline.get('cat_classes', {})

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## Student Profile Builder")
st.sidebar.markdown("<div style='height:1px;background:rgba(99,102,241,0.2);margin:0.5rem 0 1rem'></div>", unsafe_allow_html=True)

def _max_val(f):
    fl = f.lower()
    if any(k in fl for k in ["absences","score","hands","visited","discussion","view","mean","median","max_score","min_score","std","submit"]): return 100.0
    if "credits" in fl: return 300.0
    if "attempts" in fl: return 10.0
    return 20.0

user_input = {}
half = len(raw_features) // 2

with st.sidebar.expander("Academic & Demographics", expanded=True):
    for feature in raw_features[:half]:
        if feature in num_cols:
            mv = _max_val(feature)
            user_input[feature] = st.slider(feature.replace("_"," ").title(), 0.0, mv, mv/2, key=feature)
        elif feature in cat_cols:
            classes = cat_classes.get(feature, ["Unknown"])
            user_input[feature] = st.selectbox(feature.replace("_"," ").title(), classes, key=feature)

with st.sidebar.expander("Behavioural & Engagement", expanded=True):
    for feature in raw_features[half:]:
        if feature in num_cols:
            mv = _max_val(feature)
            user_input[feature] = st.slider(feature.replace("_"," ").title(), 0.0, mv, mv/2, key=feature+"_b")
        elif feature in cat_cols:
            classes = cat_classes.get(feature, ["Unknown"])
            user_input[feature] = st.selectbox(feature.replace("_"," ").title(), classes, key=feature+"_b")

# ── Preprocessing ─────────────────────────────────────────────────────────────
input_df = pd.DataFrame([user_input])
if num_cols:
    input_df[num_cols] = pipeline['num_imputer'].transform(input_df[num_cols])
if cat_cols:
    input_df[cat_cols] = pipeline['cat_imputer'].transform(input_df[cat_cols])
    input_df[cat_cols] = pipeline['encoder'].transform(input_df[cat_cols].astype(str))
input_df_scaled   = pd.DataFrame(pipeline['scaler'].transform(input_df), columns=input_df.columns)
input_df_selected = input_df_scaled[pipeline['selected_features']]

# ── Results layout ────────────────────────────────────────────────────────────
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
res_col, shap_col = st.columns([1, 2], gap="large")

# class info — covers both xAPI (0=Low,1=Med,2=High) and Dropout (0=Dropout,1=Enrolled,2=Graduate)
class_info = {
    0: ("Low / Dropout",    "out-low",         "<svg viewBox='0 0 24 24'><path d='M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z'/><line x1='12' y1='9' x2='12' y2='13'/><line x1='12' y1='17' x2='12.01' y2='17'/></svg>",        "#fb7185"),
    1: ("Medium / Enrolled","out-medium",       "<svg viewBox='0 0 24 24'><polyline points='22 7 13.5 15.5 8.5 10.5 2 17'/><polyline points='16 7 22 7 22 13'/></svg>",     "#60a5fa"),
    2: ("High / Graduate",  "out-high",         "<svg viewBox='0 0 24 24'><path d='M22 11.08V12a10 10 0 1 1-5.93-9.14'/><polyline points='22 4 12 14.01 9 11.01'/></svg>", "#34d399"),
    3: ("Distinction",      "out-distinction",  "<svg viewBox='0 0 24 24'><polygon points='12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2'/></svg>",         "#fbbf24"),
}

with res_col:
    st.markdown("<div class='result-card'><div class='result-card-title'><span class='section-icon'><svg viewBox='0 0 24 24'><line x1='18' y1='20' x2='18' y2='10'/><line x1='12' y1='20' x2='12' y2='4'/><line x1='6' y1='20' x2='6' y2='14'/></svg></span>Prediction Result</div>", unsafe_allow_html=True)

    if task_type == 'regression':
        pred = model.predict(input_df_selected)[0]
        if pred >= 16:   band, cls, icon = "Excellent", "out-high",        "<svg viewBox='0 0 24 24'><circle cx='12' cy='8' r='6'/><path d='M15.477 12.89L17 22l-5-3-5 3 1.523-9.11'/></svg>"
        elif pred >= 12: band, cls, icon = "Good",      "out-high",        "<svg viewBox='0 0 24 24'><polyline points='22 7 13.5 15.5 8.5 10.5 2 17'/><polyline points='16 7 22 7 22 13'/></svg>"
        elif pred >= 8:  band, cls, icon = "Moderate",  "out-medium",      "<svg viewBox='0 0 24 24'><line x1='18' y1='20' x2='18' y2='10'/><line x1='12' y1='20' x2='12' y2='4'/><line x1='6' y1='20' x2='6' y2='14'/></svg>"
        else:            band, cls, icon = "At Risk",   "out-low",         "<svg viewBox='0 0 24 24'><path d='M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z'/><line x1='12' y1='9' x2='12' y2='13'/><line x1='12' y1='17' x2='12.01' y2='17'/></svg>"
        st.markdown(f"""
        <div class="outcome-display {cls}">
            <div class="outcome-icon">{icon}</div>
            <span class="outcome-label">G3 Score: {pred:.1f} / 20</span>
            <span class="outcome-sub">Performance Band: {band}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        pred_probs = model.predict_proba(input_df_selected)[0]
        pred_class = model.predict(input_df_selected)[0]
        label, cls, icon, _ = class_info.get(int(pred_class), ("Unknown", "out-medium", "<svg viewBox='0 0 24 24'><circle cx='12' cy='12' r='10'/><path d='M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3'/><line x1='12' y1='17' x2='12.01' y2='17'/></svg>", "#94a3b8"))
        st.markdown(f"""
        <div class="outcome-display {cls}">
            <div class="outcome-icon">{icon}</div>
            <span class="outcome-label">{label}</span>
            <span class="outcome-sub">Predicted Performance Class</span>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bars (custom rendered)
        st.markdown('<div class="conf-section-label">Ensemble Confidence</div>', unsafe_allow_html=True)
        for i, p in enumerate(pred_probs):
            lbl, _, ico, col = class_info.get(i, (f"Class {i}", "", "", "#818cf8"))
            pct = p * 100
            fill_w = f"{pct:.1f}%"
            st.markdown(f"""
            <div class="conf-item">
                <div class="conf-top">
                    <span class="conf-name"><span class="conf-name-icon">{ico}{lbl}</span></span>
                    <span class="conf-pct">{pct:.1f}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{fill_w};background:linear-gradient(90deg,{col}99,{col});"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with shap_col:
    st.markdown("""
    <div class="shap-panel">
        <div class="shap-panel-title"><span class="section-icon"><svg viewBox='0 0 24 24'><rect x='4' y='4' width='16' height='16' rx='2'/><rect x='9' y='9' width='6' height='6'/><line x1='9' y1='2' x2='9' y2='4'/><line x1='15' y1='2' x2='15' y2='4'/><line x1='9' y1='20' x2='9' y2='22'/><line x1='15' y1='20' x2='15' y2='22'/><line x1='20' y1='9' x2='22' y2='9'/><line x1='20' y1='14' x2='22' y2='14'/><line x1='2' y1='9' x2='4' y2='9'/><line x1='2' y1='14' x2='4' y2='14'/></svg></span>SHAP Force Plot — Why this prediction?</div>
        <div class="shap-legend">
            <div class="legend-item"><div class="legend-swatch" style="background:#ef4444;"></div>Pushes higher</div>
            <div class="legend-item"><div class="legend-swatch" style="background:#3b82f6;"></div>Pushes lower</div>
            <div class="legend-item"><div class="legend-swatch" style="background:#334155;"></div>Width = magnitude</div>
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("Computing SHAP explanation..."):
        shap_values = explainer.shap_values(input_df_selected)
        try:
            if task_type == 'regression':
                val     = shap_values[0] if isinstance(shap_values, list) else shap_values
                exp_val = explainer.expected_value
                st_shap(shap.force_plot(float(exp_val), val, input_df_selected), height=180)
            else:
                if isinstance(shap_values, list):
                    val     = shap_values[int(pred_class)]
                    exp_val = (explainer.expected_value[int(pred_class)]
                               if hasattr(explainer.expected_value, '__len__')
                               else explainer.expected_value)
                else:
                    val     = shap_values
                    exp_val = explainer.expected_value
                st_shap(shap.force_plot(float(exp_val), val, input_df_selected), height=180)
        except Exception as e:
            st.warning(f"Force plot unavailable: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Feature contributions table
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    try:
        if isinstance(shap_values, list):
            sv = shap_values[int(pred_class)] if task_type == 'classification' else shap_values[0]
        else:
            sv = shap_values
        sv_flat = np.array(sv).flatten()
        feat_imp = pd.DataFrame({
            "Feature":    pipeline['selected_features'],
            "SHAP Value": sv_flat,
            "Impact":     ["↑ Increases" if v > 0 else "↓ Decreases" for v in sv_flat],
        }).reindex(pd.Series(np.abs(sv_flat)).sort_values(ascending=False).index)

        st.markdown("""
        <div class="shap-panel">
            <div class="shap-panel-title"><span class="section-icon"><svg viewBox='0 0 24 24'><line x1='8' y1='6' x2='21' y2='6'/><line x1='8' y1='12' x2='21' y2='12'/><line x1='8' y1='18' x2='21' y2='18'/><line x1='3' y1='6' x2='3.01' y2='6'/><line x1='3' y1='12' x2='3.01' y2='12'/><line x1='3' y1='18' x2='3.01' y2='18'/></svg></span>Top Feature Contributions</div>
        """, unsafe_allow_html=True)
        st.dataframe(
            feat_imp.head(8)
                .style.background_gradient(subset=["SHAP Value"], cmap="RdYlGn")
                .format({"SHAP Value": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception:
        pass