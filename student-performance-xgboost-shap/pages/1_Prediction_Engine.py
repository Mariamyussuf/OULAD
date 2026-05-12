import streamlit as st
import pickle
import pandas as pd
import shap
import numpy as np
from pathlib import Path
from streamlit_shap import st_shap

st.set_page_config(page_title="Prediction Engine — EduPredict", page_icon="🤖", layout="wide")

# ── Shared CSS (same dark theme) ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }
[data-testid="stSidebar"] { background: rgba(255,255,255,0.04); border-right: 1px solid rgba(255,255,255,0.08); }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stSlider > div > div { background: rgba(102,126,234,0.3) !important; }

/* Page header */
.page-header {
    background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
    border: 1px solid rgba(102,126,234,0.3);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.8rem;
}
.page-header h1 { font-size: 1.8rem; font-weight: 800; color: #e2e8f0 !important; margin: 0 0 0.4rem 0; }
.page-header p  { color: #94a3b8; font-size: 0.92rem; margin: 0; }

/* Dataset badge */
.dataset-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.badge-uci   { background: rgba(52,211,153,0.15); border: 1px solid rgba(52,211,153,0.4); color: #34d399; }
.badge-xapi  { background: rgba(96,165,250,0.15); border: 1px solid rgba(96,165,250,0.4); color: #60a5fa; }
.badge-oulad { background: rgba(251,146,60,0.15); border: 1px solid rgba(251,146,60,0.4);  color: #fb923c; }

/* Prediction card */
.pred-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.8rem;
    height: 100%;
}
.pred-card h3 { color: #e2e8f0; font-size: 1rem; font-weight: 700; margin: 0 0 1.2rem 0; }

/* Outcome pill */
.outcome-pill {
    text-align: center;
    padding: 1rem;
    border-radius: 12px;
    margin: 0.6rem 0;
    font-weight: 700;
    font-size: 1.4rem;
}
.outcome-high        { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.outcome-medium      { background: rgba(96,165,250,0.15); color: #60a5fa; border: 1px solid rgba(96,165,250,0.3); }
.outcome-low         { background: rgba(251,113,133,0.15); color: #fb7185; border: 1px solid rgba(251,113,133,0.3); }
.outcome-distinction { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }

/* Confidence bar label */
.conf-label { color: #94a3b8; font-size: 0.8rem; margin-bottom: 0.2rem; }

/* SHAP card */
.shap-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 1.8rem;
}
.shap-card h3 { color: #e2e8f0; font-size: 1rem; font-weight: 700; margin: 0 0 0.6rem 0; }
.shap-card p  { color: #64748b; font-size: 0.83rem; margin: 0 0 1rem 0; }

h1, h2, h3, h4 { color: #e2e8f0 !important; }
p { color: #94a3b8; }
.stSelectbox label, .stSlider label { color: #a0aec0 !important; font-size: 0.85rem !important; }
.stProgress > div > div { background: linear-gradient(90deg, #667eea, #f093fb) !important; }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = Path("models")

@st.cache_resource
def load_model_artifacts(dataset_name):
    with open(MODELS_DIR / f"{dataset_name}_model.pkl",    "rb") as f: model    = pickle.load(f)
    with open(MODELS_DIR / f"{dataset_name}_explainer.pkl","rb") as f: explainer= pickle.load(f)
    with open(MODELS_DIR / f"{dataset_name}_pipeline.pkl", "rb") as f: pipeline = pickle.load(f)
    return model, explainer, pipeline

# ── Page header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>🤖 Predictive Analytics Engine</h1>
    <p>Configure a student profile using the sidebar controls, then watch the stacking ensemble
       predict the performance outcome — with a real-time SHAP force plot explaining every decision.</p>
</div>
""", unsafe_allow_html=True)

# ── Dataset selector ─────────────────────────────────────────────────────────
dataset_choice = st.selectbox(
    "**Select Analytical Context:**",
    ("UCI (High School Grades)", "xAPI (K-12 Online Learning)", "OULAD (University Online Courses)"),
    help="Each dataset uses a separately trained stacking ensemble."
)

dataset_map = {
    "UCI (High School Grades)":         ("uci",   "regression",     "badge-uci"),
    "xAPI (K-12 Online Learning)":      ("xapi",  "classification", "badge-xapi"),
    "OULAD (University Online Courses)":("oulad", "classification", "badge-oulad"),
}
dataset_key, task_type, badge_class = dataset_map[dataset_choice]

badge_labels = {"uci": "📈 Regression Task", "xapi": "🏷️ 3-Class Classification", "oulad": "🏷️ 4-Class Classification"}
st.markdown(f'<span class="dataset-badge {badge_class}">{badge_labels[dataset_key]}</span>', unsafe_allow_html=True)

# ── Load artifacts ───────────────────────────────────────────────────────────
try:
    model, explainer, pipeline = load_model_artifacts(dataset_key)
except FileNotFoundError:
    st.error("⚠️ Model artifacts not found. Please run the training pipeline first: `python src/train.py`")
    st.stop()

raw_features  = pipeline['raw_features']
cat_cols      = pipeline['cat_cols']
num_cols      = pipeline['num_cols']
cat_classes   = pipeline.get('cat_classes', {})

# ── Sidebar profile builder ──────────────────────────────────────────────────
st.sidebar.markdown("## 🎛️ Student Profile")
st.sidebar.markdown("---")

user_input = {}
half = len(raw_features) // 2

def _max_val(feature):
    if any(k in feature.lower() for k in ["absences","score","hands","visited","discussion","view","mean","median","max_score","min_score","std","submit"]):
        return 100.0
    if "credits" in feature.lower(): return 300.0
    if "attempts" in feature.lower(): return 10.0
    return 20.0

with st.sidebar.expander("📚 Academic & Demographics", expanded=True):
    for feature in raw_features[:half]:
        if feature in num_cols:
            mv = _max_val(feature)
            user_input[feature] = st.slider(feature.replace("_"," ").title(), 0.0, mv, mv / 2, key=feature)
        elif feature in cat_cols:
            classes = cat_classes.get(feature, ["Unknown"])
            user_input[feature] = st.selectbox(feature.replace("_"," ").title(), classes, key=feature)

with st.sidebar.expander("💬 Behavioural & Engagement", expanded=True):
    for feature in raw_features[half:]:
        if feature in num_cols:
            mv = _max_val(feature)
            user_input[feature] = st.slider(feature.replace("_"," ").title(), 0.0, mv, mv / 2, key=feature+"_b")
        elif feature in cat_cols:
            classes = cat_classes.get(feature, ["Unknown"])
            user_input[feature] = st.selectbox(feature.replace("_"," ").title(), classes, key=feature+"_b")

# ── Preprocessing ─────────────────────────────────────────────────────────────
input_df = pd.DataFrame([user_input])
if num_cols: input_df[num_cols] = pipeline['num_imputer'].transform(input_df[num_cols])
if cat_cols:
    input_df[cat_cols] = pipeline['cat_imputer'].transform(input_df[cat_cols])
    input_df[cat_cols] = pipeline['encoder'].transform(input_df[cat_cols].astype(str))
input_df_scaled   = pd.DataFrame(pipeline['scaler'].transform(input_df), columns=input_df.columns)
input_df_selected = input_df_scaled[pipeline['selected_features']]

# ── Results layout ────────────────────────────────────────────────────────────
st.markdown("---")
res_col, shap_col = st.columns([1, 2], gap="large")

with res_col:
    st.markdown('<div class="pred-card"><h3>📊 Prediction Result</h3>', unsafe_allow_html=True)

    if task_type == 'regression':
        pred = model.predict(input_df_selected)[0]
        # Map score to band
        if pred >= 15:   band, cls = "High", "outcome-high"
        elif pred >= 10: band, cls = "Medium", "outcome-medium"
        else:            band, cls = "Low", "outcome-low"
        st.markdown(f'<div class="outcome-pill {cls}">G3 Score: {pred:.1f}/20</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="outcome-pill {cls}">Band: {band}</div>', unsafe_allow_html=True)
    else:
        pred_probs = model.predict_proba(input_df_selected)[0]
        pred_class = model.predict(input_df_selected)[0]

        class_info = {
            0: ("Low / Fail",       "outcome-low",         "📉"),
            1: ("Medium / Withdrawn","outcome-medium",     "📊"),
            2: ("High / Pass",       "outcome-high",       "📈"),
            3: ("Distinction",       "outcome-distinction", "🌟"),
        }
        label, pill_cls, icon = class_info.get(int(pred_class), ("Unknown", "outcome-medium", "❓"))
        st.markdown(f'<div class="outcome-pill {pill_cls}">{icon} {label}</div>', unsafe_allow_html=True)

        st.markdown("<br>**Ensemble Confidence:**", unsafe_allow_html=True)
        for i, p in enumerate(pred_probs):
            lbl, _, ico = class_info.get(i, (f"Class {i}", "outcome-medium", ""))
            st.markdown(f'<div class="conf-label">{ico} {lbl}: {p*100:.1f}%</div>', unsafe_allow_html=True)
            st.progress(float(p))

    st.markdown('</div>', unsafe_allow_html=True)

with shap_col:
    st.markdown('<div class="shap-card"><h3>🔍 SHAP Force Plot — Why this prediction?</h3><p>Red features push the score higher · Blue features push it lower · Width = magnitude of impact</p>', unsafe_allow_html=True)

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
            st.warning(f"Force plot unavailable for this configuration: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Top features table
    st.markdown("<br>", unsafe_allow_html=True)
    try:
        if isinstance(shap_values, list):
            sv = shap_values[int(pred_class)] if task_type == 'classification' else shap_values[0]
        else:
            sv = shap_values

        sv_flat = np.array(sv).flatten()
        feat_imp = pd.DataFrame({
            "Feature":     pipeline['selected_features'],
            "SHAP Value":  sv_flat,
            "Direction":   ["↑ Increases risk" if v > 0 else "↓ Decreases risk" for v in sv_flat],
        }).reindex(pd.Series(np.abs(sv_flat)).sort_values(ascending=False).index)

        st.markdown("**Top Feature Contributions for this Student:**")
        st.dataframe(
            feat_imp.head(8).style
                .background_gradient(subset=["SHAP Value"], cmap="RdYlGn")
                .format({"SHAP Value": "{:.4f}"}),
            use_container_width=True,
            hide_index=True
        )
    except Exception:
        pass
