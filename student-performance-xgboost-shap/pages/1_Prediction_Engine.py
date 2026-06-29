import streamlit as st
import pickle
import pandas as pd
import shap
import numpy as np

# ── Monkeypatch SHAP KernelExplainer to fix empty run broadcast bug ──────────
_orig_kernel_run = shap.explainers._kernel.KernelExplainer.run
def _patched_kernel_run(self):
    if self.nsamplesAdded == self.nsamplesRun:
        return
    return _orig_kernel_run(self)
shap.explainers._kernel.KernelExplainer.run = _patched_kernel_run

from pathlib import Path
from streamlit_shap import st_shap

st.set_page_config(page_title="Prediction Engine — EduPredict", page_icon="⬡", layout="wide")

# ── Global Design System ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: #0B0F19;
    min-height: 100vh;
}

[data-testid="stSidebar"] {
    background: #070A13 !important;
    border-right: 1px solid #1E293B !important;
}
[data-testid="stSidebar"] * { color: #94A3B8 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #111827 !important;
    border: 1px solid #1E293B !important;
    color: #F1F5F9 !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: #06B6D4 !important;
}

/* Nav */
[data-testid="stSidebarNav"] { padding-top: 1rem; }
[data-testid="stSidebarNav"] a { border-radius: 8px !important; margin: 4px 0 !important; transition: background 0.15s, color 0.15s !important; }
[data-testid="stSidebarNav"] a:hover { background: #111827 !important; color: #F1F5F9 !important; }
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: #1E293B !important;
    border-left: 3px solid #06B6D4 !important;
    color: #F8FAFC !important;
}

h1, h2, h3, h4 { color: #F8FAFC !important; }
p { color: #94A3B8; line-height: 1.7; }

/* ── Page header ─────────────────────────── */
.page-header {
    background: #111827;
    border: 1px solid #1E293B;
    border-top: 1px solid #334155;
    border-radius: 16px;
    padding: 2rem 2.4rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    text-align: center;
}
.page-header-tag { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #38BDF8; margin-bottom: 0.6rem; font-family: 'JetBrains Mono', monospace; }
.page-header h1  { font-size: 1.8rem !important; font-weight: 800 !important; color: #F8FAFC !important; margin: 0 0 0.5rem 0 !important; letter-spacing: -0.02em; }
.page-header p   { color: #94A3B8; font-size: 0.9rem; margin: 0; }

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
    font-family: 'JetBrains Mono', monospace;
}
.pill-uci   { background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.3); color: #34D399; }
.pill-xapi  { background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.3); color: #38BDF8; }
.pill-oulad { background: rgba(251,146,60,0.1); border: 1px solid rgba(251,146,60,0.3); color: #FBBF24; }

/* ── Result card ─────────────────────────── */
.result-card {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 1.8rem;
    height: 100%;
}
.result-card-title {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38BDF8;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.result-card-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1E293B;
}

/* ── Outcome display ─────────────────────── */
.outcome-display {
    text-align: center;
    padding: 1.8rem 1rem;
    border-radius: 12px;
    margin-bottom: 1.4rem;
    border: 1px solid transparent;
}
.outcome-label  { font-size: 1.5rem; font-weight: 800; display: block; margin-bottom: 0.35rem; font-family: 'JetBrains Mono', monospace; }
.outcome-sub    { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.8; }
.out-high        { background: #064E3B; border-color: #059669; color: #34D399; }
.out-high .outcome-label  { color: #34D399; }
.out-medium      { background: #1E3A8A; border-color: #2563EB; color: #60A5FA; }
.out-medium .outcome-label { color: #60A5FA; }
.out-low         { background: #7F1D1D; border-color: #DC2626; color: #F87171; }
.out-low .outcome-label   { color: #F87171; }
.out-distinction { background: #78350F; border-color: #D97706; color: #FBBF24; }
.out-distinction .outcome-label { color: #FBBF24; }

/* ── Confidence bars ─────────────────────── */
.conf-section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 0.8rem;
}
.conf-item { margin-bottom: 0.7rem; }
.conf-top {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.3rem;
}
.conf-name { color: #94A3B8; font-size: 0.8rem; font-weight: 500; }
.conf-pct  { color: #F1F5F9; font-size: 0.8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.conf-track {
    height: 6px;
    background: #1E293B;
    border-radius: 3px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 3px;
    background: #06B6D4;
    transition: width 0.5s ease;
}

/* ── SHAP panel ─────────────────────────── */
.shap-panel {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
}
.shap-panel-title {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38BDF8;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.shap-panel-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1E293B;
}
.shap-legend {
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
}
.legend-item { display: flex; align-items: center; gap: 0.4rem; font-size: 0.78rem; color: #cbd5e1; }
.legend-swatch { width: 8px; height: 8px; border-radius: 2px; flex-shrink: 0; }

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
    width: 44px; height: 44px;
    border-radius: 8px;
    margin: 0 auto 0.9rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
}
.outcome-icon svg {
    width: 20px; height: 20px;
    stroke: currentColor;
    stroke-width: 1.75;
    stroke-linecap: round;
    stroke-linejoin: round;
    fill: none;
}
.out-high   .outcome-icon { color: #34D399; }
.out-medium .outcome-icon { color: #60A5FA; }
.out-low    .outcome-icon { color: #F87171; }
.out-distinction .outcome-icon { color: #FBBF24; }

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
    stroke: #38BDF8; stroke-width: 2;
    stroke-linecap: round; stroke-linejoin: round; fill: none;
}

/* ── Header tag icon ─────────────────────── */
.header-icon {
    display: inline-flex; align-items: center; gap: 0.45rem;
}
.header-icon svg {
    width: 12px; height: 12px;
    stroke: #38BDF8; stroke-width: 2.25;
    stroke-linecap: round; stroke-linejoin: round; fill: none;
}

/* ── Streamlit widget overrides ─────────────────────────── */
.stSelectbox label, .stSlider label, .stMultiSelect label, .stCheckbox label {
    color: #94A3B8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
.stProgress > div > div { background: #06B6D4 !important; }
.stDataFrame { border-radius: 8px; overflow: hidden; border: 1px solid #1E293B; }
.stSlider > div > div > div {
    background: #06B6D4 !important;
}

/* Sidebar expander */
[data-testid="stExpander"] {
    background: #111827 !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stExpander"] summary { color: #F1F5F9 !important; font-weight: 700 !important; font-size: 0.86rem !important; }
[data-testid="stExpander"] > div > div { border-top: 1px solid #1E293B !important; }

/* Make SHAP force plot iframe legible with a white background */
iframe {
    background-color: #ffffff !important;
    border-radius: 12px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

MODELS_DIR = Path("models")

# ── Human-readable feature labels (all datasets) ─────────────────────────────
FEATURE_LABELS = {
    # UCI Student Performance
    'school': 'School Name', 'sex': 'Gender', 'age': 'Age',
    'address': 'Home Address (Urban/Rural)', 'famsize': 'Family Size',
    'Pstatus': 'Parent Cohabitation Status', 'Medu': "Mother's Education Level",
    'Fedu': "Father's Education Level", 'Mjob': "Mother's Job",
    'Fjob': "Father's Job", 'reason': 'Reason for Choosing School',
    'guardian': 'Student Guardian', 'traveltime': 'Home-to-School Travel Time',
    'studytime': 'Weekly Study Time', 'failures': 'Past Class Failures',
    'schoolsup': 'Extra Educational Support', 'famsup': 'Family Educational Support',
    'paid': 'Extra Paid Classes', 'activities': 'Extracurricular Activities',
    'nursery': 'Attended Nursery School', 'higher': 'Wants Higher Education',
    'internet': 'Internet Access at Home', 'romantic': 'In a Romantic Relationship',
    'famrel': 'Family Relationship Quality (1-5)', 'freetime': 'Free Time After School (1-5)',
    'goout': 'Going Out with Friends (1-5)', 'Dalc': 'Workday Alcohol Consumption (1-5)',
    'Walc': 'Weekend Alcohol Consumption (1-5)', 'health': 'Current Health Status (1-5)',
    'absences': 'Number of School Absences', 'G1': 'First Period Grade (0-20)',
    'G2': 'Second Period Grade (0-20)',
    # xAPI Educational Data
    'gender': 'Gender', 'NationalITy': 'Nationality',
    'PlaceofBirth': 'Place of Birth', 'StageID': 'Educational Stage',
    'GradeID': 'Grade Level', 'SectionID': 'Classroom Section',
    'Topic': 'Course Topic', 'Semester': 'Semester',
    'Relation': 'Parent Responsible for Student', 'raisedhands': 'Raised Hands in Class',
    'VisITedResources': 'Visited Learning Resources',
    'AnnouncementsView': 'Announcements Viewed',
    'Discussion': 'Discussion Group Participation',
    'ParentAnsweringSurvey': 'Parent Answered School Survey',
    'ParentschoolSatisfaction': 'Parent School Satisfaction',
    'StudentAbsenceDays': 'Student Absence Days',
    # UCI Dropout & Academic Success
    'Marital status': 'Marital Status', 'Application mode': 'Application Mode',
    'Application order': 'Application Preference Order', 'Course': 'Course Enrolled',
    'Daytime/evening attendance': 'Daytime / Evening Attendance',
    'Previous qualification': 'Previous Qualification', 'Nacionality': 'Nationality',
    'Previous qualification (grade)': 'Previous Qualification Grade',
    "Mother's qualification": "Mother's Qualification",
    "Father's qualification": "Father's Qualification",
    "Mother's occupation": "Mother's Occupation",
    "Father's occupation": "Father's Occupation",
    'Admission grade': 'Admission Grade', 'Displaced': 'Displaced Student',
    'Educational special needs': 'Special Educational Needs',
    'Debtor': 'Has Outstanding Debt', 'Tuition fees up to date': 'Tuition Fees Up to Date',
    'Gender': 'Gender', 'Scholarship holder': 'Scholarship Holder',
    'Age at enrollment': 'Age at Enrolment', 'International': 'International Student',
    'Curricular units 1st sem (credited)': '1st Sem Units Credited',
    'Curricular units 1st sem (enrolled)': '1st Sem Units Enrolled',
    'Curricular units 1st sem (evaluations)': '1st Sem Evaluations',
    'Curricular units 1st sem (approved)': '1st Sem Units Approved',
    'Curricular units 1st sem (grade)': '1st Sem Average Grade',
    'Curricular units 1st sem (without evaluations)': '1st Sem Without Evaluations',
    'Curricular units 2nd sem (credited)': '2nd Sem Units Credited',
    'Curricular units 2nd sem (enrolled)': '2nd Sem Units Enrolled',
    'Curricular units 2nd sem (evaluations)': '2nd Sem Evaluations',
    'Curricular units 2nd sem (approved)': '2nd Sem Units Approved',
    'Curricular units 2nd sem (grade)': '2nd Sem Average Grade',
    'Curricular units 2nd sem (without evaluations)': '2nd Sem Without Evaluations',
    'Unemployment rate': 'Unemployment Rate (%)', 'Inflation rate': 'Inflation Rate (%)',
    'GDP': 'GDP Growth (%)',
}

def _friendly(name: str) -> str:
    """Return a human-readable label for a raw feature name."""
    return FEATURE_LABELS.get(name, name.replace('_', ' ').title())


# ── Cached artifact loader with subsampled background ────────────────────────
# @st.cache_resource ensures this runs ONCE per dataset per session.
# Key design decisions:
#   1. bg_df is subsampled to ≤100 rows before KernelExplainer.
#      A large background causes nsamples to be exhausted before SHAP can write,
#      producing the "(0,3) broadcast" ValueError.
#   2. predict_fn uses default-arg binding (_fn=, _c=) to capture values at
#      definition time, preventing Python late-binding bugs in loops/closures.
@st.cache_resource
def load_model_artifacts(dataset_name):
    model_path     = MODELS_DIR / f"{dataset_name}_model.pkl"
    explainer_path = MODELS_DIR / f"{dataset_name}_explainer.pkl"
    pipeline_path  = MODELS_DIR / f"{dataset_name}_pipeline.pkl"

    with open(model_path,     "rb") as f: model     = pickle.load(f)
    with open(explainer_path, "rb") as f: explainer = pickle.load(f)
    with open(pipeline_path,  "rb") as f: pipeline  = pickle.load(f)

    # Reconstruct background DataFrame
    bg_raw = (
        explainer.data.data
        if hasattr(explainer.data, 'data')
        else np.array(explainer.data)
    )
    bg_df = pd.DataFrame(bg_raw, columns=pipeline['selected_features'])

    # Subsample background to ≤100 rows.
    # KernelExplainer allocates self.y with shape (nsamples * N, D).
    # When N is large, even nsamples="auto" can overflow the pre-allocated
    # array. 100 rows is the SHAP-recommended maximum for KernelExplainer.
    if len(bg_df) > 100:
        bg_df = shap.sample(bg_df, 100, random_state=42)

    cols = list(pipeline['selected_features'])

    # Default-arg binding prevents late-binding closure bugs.
    if hasattr(model, 'predict_proba'):
        def predict_fn(X, _fn=model.predict_proba, _c=cols):
            return _fn(pd.DataFrame(X, columns=_c))
    else:
        def predict_fn(X, _fn=model.predict, _c=cols):
            return _fn(pd.DataFrame(X, columns=_c))

    explainer = shap.KernelExplainer(predict_fn, bg_df)
    return model, explainer, pipeline


# ── SHAP caching by input hash ───────────────────────────────────────────────
# Avoids recomputing SHAP on every widget interaction (Streamlit reruns the
# whole script on each slider/selectbox change). We store the result in
# session_state keyed by (dataset, input_hash) so it is only recomputed when
# the student profile actually changes.
def _compute_shap(explainer, input_df_selected: pd.DataFrame,
                  dataset_key: str, task_type: str):
    cache_key = f"shap_{dataset_key}_{pd.util.hash_pandas_object(input_df_selected).sum()}"
    if cache_key not in st.session_state:
        # nsamples="auto" lets SHAP compute the correct count
        # (2*M + 2^M, capped at a safe maximum) rather than a hard-coded 200
        # that could be too low for the feature count, causing the (0,3) error.
        st.session_state[cache_key] = explainer.shap_values(
            input_df_selected, nsamples="auto"
        )
    return st.session_state[cache_key]


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-header-tag">
        <span class="header-icon">
            <svg viewBox='0 0 24 24'><polygon points='13 2 3 14 12 14 11 22 21 10 12 10 13 2'/></svg>
            Interactive Prediction
        </span>
    </div>
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
    "UCI (High School Grades)":        ("uci",     "regression",     "pill-uci",   "REG · Final Grade G3"),
    "xAPI (K-12 Online Learning)":     ("xapi",    "classification", "pill-xapi",  "CLF · Low / Medium / High"),
    "UCI Dropout & Academic Success":  ("dropout", "classification", "pill-oulad", "CLF · Dropout / Enrolled / Graduate"),
}
dataset_key, task_type, pill_cls, pill_label = dataset_map[dataset_choice]
st.markdown(f'<span class="ds-pill {pill_cls}">{pill_label}</span>', unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
try:
    model, explainer, pipeline = load_model_artifacts(dataset_key)
except FileNotFoundError:
    st.error(
        "Model artifacts not found for this dataset. "
        "Please run: `python src/train.py` — then refresh this page."
    )
    st.stop()
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}")
    st.stop()

raw_features = pipeline['raw_features']
cat_cols     = pipeline['cat_cols']
num_cols     = pipeline['num_cols']
cat_classes  = pipeline.get('cat_classes', {})

# ── Student Profile Builder (inline) ─────────────────────────────────────────
SVG_USER = "<svg viewBox='0 0 24 24'><path d='M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2'/><circle cx='12' cy='7' r='4'/></svg>"
st.markdown(
    f"<div class='section-label' style='margin-top:0.5rem'>"
    f"<span class='section-icon'>{SVG_USER}</span>Student Profile Builder</div>",
    unsafe_allow_html=True,
)

# Ordinal features displayed as descriptive dropdowns: {feature: {label: value}}
ORDINAL_OPTIONS = {
    'Medu': {
        "None (0)": 0, "Primary – 4th Grade (1)": 1,
        "5th to 9th Grade (2)": 2, "Secondary Education (3)": 3,
        "Higher Education (4)": 4,
    },
    'Fedu': {
        "None (0)": 0, "Primary – 4th Grade (1)": 1,
        "5th to 9th Grade (2)": 2, "Secondary Education (3)": 3,
        "Higher Education (4)": 4,
    },
    'traveltime': {
        "Under 15 minutes": 1, "15 – 30 minutes": 2,
        "30 min – 1 hour": 3, "Over 1 hour": 4,
    },
    'studytime': {
        "Under 2 hours / week": 1, "2 – 5 hours / week": 2,
        "5 – 10 hours / week": 3, "Over 10 hours / week": 4,
    },
    'famrel': {
        "1 – Very Bad": 1, "2 – Bad": 2, "3 – Average": 3,
        "4 – Good": 4, "5 – Excellent": 5,
    },
    'freetime': {
        "1 – Very Low": 1, "2 – Low": 2, "3 – Average": 3,
        "4 – High": 4, "5 – Very High": 5,
    },
    'goout': {
        "1 – Very Low": 1, "2 – Low": 2, "3 – Average": 3,
        "4 – High": 4, "5 – Very High": 5,
    },
    'Dalc': {
        "1 – Very Low": 1, "2 – Low": 2, "3 – Moderate": 3,
        "4 – High": 4, "5 – Very High": 5,
    },
    'Walc': {
        "1 – Very Low": 1, "2 – Low": 2, "3 – Moderate": 3,
        "4 – High": 4, "5 – Very High": 5,
    },
    'health': {
        "1 – Very Poor": 1,
        "2 – Poor": 2,          # was incorrectly mapped to 3 in original
        "3 – Average": 3,
        "4 – Good": 4,
        "5 – Very Good": 5,
    },
    'failures': {
        "None": 0, "1 failure": 1, "2 failures": 2,
        "3 failures": 3, "4 or more": 4,
    },
}

# Per-feature slider configuration: (min, max, default, step)
SLIDER_CONFIG = {
    # ── UCI Student Performance ──────────────────────────────
    'age':        (15.0, 22.0, 17.0, 1.0),
    'absences':   (0.0, 93.0, 4.0, 1.0),
    'G1':         (0.0, 20.0, 10.0, 1.0),
    'G2':         (0.0, 20.0, 10.0, 1.0),
    # ── xAPI Educational Data ────────────────────────────────
    'raisedhands':       (0.0, 100.0, 50.0, 1.0),
    'VisITedResources':  (0.0, 100.0, 50.0, 1.0),
    'AnnouncementsView': (0.0, 100.0, 50.0, 1.0),
    'Discussion':        (0.0, 100.0, 50.0, 1.0),
    # ── UCI Dropout & Academic Success ───────────────────────
    'Application mode':       (1.0, 57.0, 1.0, 1.0),
    'Application order':      (0.0, 9.0, 1.0, 1.0),
    'Previous qualification': (1.0, 43.0, 1.0, 1.0),
    'Previous qualification (grade)':                 (0.0, 200.0, 130.0, 1.0),
    "Mother's qualification":                         (1.0, 44.0, 19.0, 1.0),
    "Father's qualification":                         (1.0, 44.0, 19.0, 1.0),
    "Mother's occupation":                            (0.0, 46.0, 5.0, 1.0),
    "Father's occupation":                            (0.0, 46.0, 5.0, 1.0),
    'Admission grade':                                (0.0, 200.0, 130.0, 1.0),
    'Age at enrollment':                              (17.0, 70.0, 20.0, 1.0),
    'Curricular units 1st sem (credited)':            (0.0, 20.0, 0.0, 1.0),
    'Curricular units 1st sem (enrolled)':            (0.0, 26.0, 6.0, 1.0),
    'Curricular units 1st sem (evaluations)':         (0.0, 45.0, 8.0, 1.0),
    'Curricular units 1st sem (approved)':            (0.0, 26.0, 5.0, 1.0),
    'Curricular units 1st sem (grade)':               (0.0, 19.0, 12.0, 0.5),
    'Curricular units 1st sem (without evaluations)': (0.0, 12.0, 0.0, 1.0),
    'Curricular units 2nd sem (credited)':            (0.0, 19.0, 0.0, 1.0),
    'Curricular units 2nd sem (enrolled)':            (0.0, 23.0, 6.0, 1.0),
    'Curricular units 2nd sem (evaluations)':         (0.0, 33.0, 8.0, 1.0),
    'Curricular units 2nd sem (approved)':            (0.0, 22.0, 5.0, 1.0),
    'Curricular units 2nd sem (grade)':               (0.0, 19.0, 12.0, 0.5),
    'Curricular units 2nd sem (without evaluations)': (0.0, 12.0, 0.0, 1.0),
    'Unemployment rate':  (0.0, 25.0, 11.0, 0.1),
    'Inflation rate':     (-2.0, 5.0, 1.0, 0.1),
    'GDP':                (-5.0, 5.0, 1.0, 0.1),
}

CATEGORICAL_MAPPINGS = {
    # UCI Student Performance
    'school':   {'GP': 'Gabriel Pereira', 'MS': 'Mousinho da Silveira'},
    'sex':      {'F': 'Female', 'M': 'Male'},
    'address':  {'R': 'Rural', 'U': 'Urban'},
    'famsize':  {'GT3': 'Greater than 3 members', 'LE3': '3 or fewer members'},
    'Pstatus':  {'A': 'Apart', 'T': 'Together'},
    'Mjob': {
        'at_home': 'At Home', 'health': 'Health Care',
        'other': 'Other', 'services': 'Civil Services', 'teacher': 'Teacher',
    },
    'Fjob': {
        'at_home': 'At Home', 'health': 'Health Care',
        'other': 'Other', 'services': 'Civil Services', 'teacher': 'Teacher',
    },
    'reason': {
        'course': 'Course Preference', 'home': 'Close to Home',
        'other': 'Other', 'reputation': 'School Reputation',
    },
    'guardian':    {'father': 'Father', 'mother': 'Mother', 'other': 'Other'},
    'schoolsup':   {'no': 'No', 'yes': 'Yes'},
    'famsup':      {'no': 'No', 'yes': 'Yes'},
    'paid':        {'no': 'No', 'yes': 'Yes'},
    'activities':  {'no': 'No', 'yes': 'Yes'},
    'nursery':     {'no': 'No', 'yes': 'Yes'},
    'higher':      {'no': 'No', 'yes': 'Yes'},
    'internet':    {'no': 'No', 'yes': 'Yes'},
    'romantic':    {'no': 'No', 'yes': 'Yes'},
    # xAPI Educational Data
    'gender': {'F': 'Female', 'M': 'Male'},
    'NationalITy': {
        'Egypt': 'Egypt', 'Iran': 'Iran', 'Iraq': 'Iraq', 'Jordan': 'Jordan',
        'KW': 'Kuwait', 'KuwaIT': 'Kuwait', 'Lybia': 'Libya', 'Morocco': 'Morocco',
        'Palestine': 'Palestine', 'SaudiArabia': 'Saudi Arabia', 'Syria': 'Syria',
        'Tunis': 'Tunisia', 'USA': 'USA', 'lebanon': 'Lebanon', 'venzuela': 'Venezuela',
    },
    'PlaceofBirth': {
        'Egypt': 'Egypt', 'Iran': 'Iran', 'Iraq': 'Iraq', 'Jordan': 'Jordan',
        'KuwaIT': 'Kuwait', 'Lybia': 'Libya', 'Morocco': 'Morocco',
        'Palestine': 'Palestine', 'SaudiArabia': 'Saudi Arabia', 'Syria': 'Syria',
        'Tunis': 'Tunisia', 'USA': 'USA', 'lebanon': 'Lebanon', 'venzuela': 'Venezuela',
    },
    'StageID': {
        'HighSchool': 'High School', 'MiddleSchool': 'Middle School', 'lowerlevel': 'Lower Level',
    },
    'Semester':  {'F': 'Fall', 'S': 'Spring'},
    'Relation':  {'Father': 'Father', 'Mum': 'Mother'},
    'ParentAnsweringSurvey':     {'No': 'No', 'Yes': 'Yes'},
    'ParentschoolSatisfaction':  {'Bad': 'Bad / Dissatisfied', 'Good': 'Good / Satisfied'},
    'StudentAbsenceDays':        {'Above-7': 'Above 7 days', 'Under-7': 'Under 7 days'},
}


def _slider_range(feature):
    """Return (min, max, default, step) for a feature, with sensible fallback."""
    return SLIDER_CONFIG.get(feature, (0.0, 20.0, 10.0, 1.0))


def _render_feature(feature, suffix=""):
    """Render a single feature widget and return the raw value for the model."""
    label = _friendly(feature)
    key   = feature + suffix

    # 1. Ordinal features → descriptive dropdown, returns numeric value
    if feature in ORDINAL_OPTIONS:
        opts = ORDINAL_OPTIONS[feature]
        labels = list(opts.keys())
        choice = st.selectbox(label, labels, index=len(labels) // 2, key=key)
        return opts[choice]

    # 2. Numeric features → slider with correct range
    if feature in num_cols:
        lo, hi, default, step = _slider_range(feature)
        return st.slider(label, lo, hi, default, step=step, key=key)

    # 3. Categorical features → dropdown with dataset classes (mapped to friendly names)
    if feature in cat_cols:
        classes = cat_classes.get(feature, ["Unknown"])
        mapping = CATEGORICAL_MAPPINGS.get(feature, {})
        display_to_raw = {mapping.get(c, c): c for c in classes}
        labels = list(display_to_raw.keys())
        choice = st.selectbox(label, labels, key=key)
        return display_to_raw[choice]

    return None


with st.form("prediction_form", border=False):
    user_input = {}
    half = len(raw_features) // 2

    prof_col1, prof_col2 = st.columns(2, gap="large")
    with prof_col1:
        with st.expander("Academic & Demographics", expanded=True):
            for feature in raw_features[:half]:
                val = _render_feature(feature)
                if val is not None:
                    user_input[feature] = val

    with prof_col2:
        with st.expander("Behavioural & Engagement", expanded=True):
            for feature in raw_features[half:]:
                val = _render_feature(feature, suffix="_b")
                if val is not None:
                    user_input[feature] = val
    
    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
    st.form_submit_button("Run Prediction 🚀", type="primary", use_container_width=True)


# ── Preprocessing ─────────────────────────────────────────────────────────────
input_df = pd.DataFrame([user_input])
if num_cols:
    input_df[num_cols] = pipeline['num_imputer'].transform(input_df[num_cols])
if cat_cols:
    input_df[cat_cols] = pipeline['cat_imputer'].transform(input_df[cat_cols])
    input_df[cat_cols] = pipeline['encoder'].transform(input_df[cat_cols].astype(str))
input_df_scaled = pd.DataFrame(
    pipeline['scaler'].transform(input_df), columns=input_df.columns
)
# .copy() prevents downstream mutations from poisoning the cached DataFrame
input_df_selected = input_df_scaled[pipeline['selected_features']].copy()


# ── Class metadata ────────────────────────────────────────────────────────────
# Covers xAPI (0=Low, 1=Medium, 2=High) and Dropout (0=Dropout, 1=Enrolled, 2=Graduate)
class_info = {
    0: (
        "Low / Dropout", "out-low",
        "<svg viewBox='0 0 24 24'><path d='M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z'/><line x1='12' y1='9' x2='12' y2='13'/><line x1='12' y1='17' x2='12.01' y2='17'/></svg>",
        "#fb7185",
    ),
    1: (
        "Medium / Enrolled", "out-medium",
        "<svg viewBox='0 0 24 24'><polyline points='22 7 13.5 15.5 8.5 10.5 2 17'/><polyline points='16 7 22 7 22 13'/></svg>",
        "#60a5fa",
    ),
    2: (
        "High / Graduate", "out-high",
        "<svg viewBox='0 0 24 24'><path d='M22 11.08V12a10 10 0 1 1-5.93-9.14'/><polyline points='22 4 12 14.01 9 11.01'/></svg>",
        "#34d399",
    ),
    3: (
        "Distinction", "out-distinction",
        "<svg viewBox='0 0 24 24'><polygon points='12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2'/></svg>",
        "#fbbf24",
    ),
}

# ── Results layout ────────────────────────────────────────────────────────────
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

st.markdown(
    "<div class='result-card'>"
    "<div class='result-card-title'>"
    "<span class='section-icon'>"
    "<svg viewBox='0 0 24 24'>"
    "<line x1='18' y1='20' x2='18' y2='10'/>"
    "<line x1='12' y1='20' x2='12' y2='4'/>"
    "<line x1='6' y1='20' x2='6' y2='14'/>"
    "</svg></span>Prediction Result</div>",
    unsafe_allow_html=True,
)

if task_type == 'regression':
    pred = model.predict(input_df_selected)[0]
    if   pred >= 16: band, cls, icon = "Excellent", "out-high",   "<svg viewBox='0 0 24 24'><circle cx='12' cy='8' r='6'/><path d='M15.477 12.89L17 22l-5-3-5 3 1.523-9.11'/></svg>"
    elif pred >= 12: band, cls, icon = "Good",      "out-high",   "<svg viewBox='0 0 24 24'><polyline points='22 7 13.5 15.5 8.5 10.5 2 17'/><polyline points='16 7 22 7 22 13'/></svg>"
    elif pred >= 8:  band, cls, icon = "Moderate",  "out-medium", "<svg viewBox='0 0 24 24'><line x1='18' y1='20' x2='18' y2='10'/><line x1='12' y1='20' x2='12' y2='4'/><line x1='6' y1='20' x2='6' y2='14'/></svg>"
    else:            band, cls, icon = "At Risk",   "out-low",    "<svg viewBox='0 0 24 24'><path d='M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z'/><line x1='12' y1='9' x2='12' y2='13'/><line x1='12' y1='17' x2='12.01' y2='17'/></svg>"

    st.markdown(f"""
    <div class="outcome-display {cls}">
        <div class="outcome-icon">{icon}</div>
        <span class="outcome-label">G3 Score: {pred:.1f} / 20</span>
        <span class="outcome-sub">Performance Band: {band}</span>
    </div>
    """, unsafe_allow_html=True)

else:
    pred_probs = model.predict_proba(input_df_selected)[0]
    pred_class = int(model.predict(input_df_selected)[0])
    label, cls, icon, _ = class_info.get(
        pred_class,
        ("Unknown", "out-medium",
         "<svg viewBox='0 0 24 24'><circle cx='12' cy='12' r='10'/><path d='M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3'/><line x1='12' y1='17' x2='12.01' y2='17'/></svg>",
         "#94a3b8"),
    )
    st.markdown(f"""
    <div class="outcome-display {cls}">
        <div class="outcome-icon">{icon}</div>
        <span class="outcome-label">{label}</span>
        <span class="outcome-sub">Predicted Performance Class</span>
    </div>
    """, unsafe_allow_html=True)

    # Confidence bars — one column per class
    conf_cols = st.columns(len(pred_probs))
    for i, (p, cc) in enumerate(zip(pred_probs, conf_cols)):
        lbl, _, ico, col = class_info.get(i, (f"Class {i}", "", "", "#818cf8"))
        pct    = p * 100
        fill_w = f"{pct:.1f}%"
        with cc:
            st.markdown(f"""
            <div class="conf-item">
                <div class="conf-top">
                    <span class="conf-name">
                        <span class="conf-name-icon">{ico}{lbl}</span>
                    </span>
                    <span class="conf-pct">{pct:.1f}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{fill_w};background:linear-gradient(90deg,{col}99,{col});"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close result-card


# ── SHAP Force Plot ───────────────────────────────────────────────────────────
input_display = input_df_selected.rename(columns=FEATURE_LABELS)

st.markdown("""
<div class="shap-panel">
    <div class="shap-panel-title">
        <span class="section-icon">
            <svg viewBox='0 0 24 24'>
                <rect x='4' y='4' width='16' height='16' rx='2'/>
                <rect x='9' y='9' width='6' height='6'/>
                <line x1='9' y1='2' x2='9' y2='4'/><line x1='15' y1='2' x2='15' y2='4'/>
                <line x1='9' y1='20' x2='9' y2='22'/><line x1='15' y1='20' x2='15' y2='22'/>
                <line x1='20' y1='9' x2='22' y2='9'/><line x1='20' y1='14' x2='22' y2='14'/>
                <line x1='2' y1='9' x2='4' y2='9'/><line x1='2' y1='14' x2='4' y2='14'/>
            </svg>
        </span>
        SHAP Force Plot — Why this prediction?
    </div>
    <div class="shap-legend">
        <div class="legend-item">
            <div class="legend-swatch" style="background:#ef4444;"></div>Pushes higher
        </div>
        <div class="legend-item">
            <div class="legend-swatch" style="background:#3b82f6;"></div>Pushes lower
        </div>
        <div class="legend-item">
            <div class="legend-swatch" style="background:#334155;"></div>Width = magnitude
        </div>
    </div>
""", unsafe_allow_html=True)

with st.spinner("Computing SHAP explanation…"):
    shap_values = _compute_shap(explainer, input_df_selected, dataset_key, task_type)
    try:
        if task_type == 'regression':
            # KernelExplainer on a regressor returns ndarray (not a list)
            val     = shap_values[0] if isinstance(shap_values, list) else shap_values
            exp_val = explainer.expected_value
            if hasattr(exp_val, '__len__'):
                exp_val = float(exp_val[0])
            st_shap(shap.force_plot(float(exp_val), np.array(val).flatten(), input_display), height=280)
        else:
            if isinstance(shap_values, list):
                val     = shap_values[pred_class]
                exp_val = (
                    explainer.expected_value[pred_class]
                    if hasattr(explainer.expected_value, '__len__')
                    else explainer.expected_value
                )
            else:
                val     = shap_values
                exp_val = explainer.expected_value
            st_shap(shap.force_plot(float(exp_val), np.array(val).flatten(), input_display), height=280)
    except Exception as e:
        st.warning(
            f"Force plot could not be rendered: {e}\n\n"
            "The feature contributions table below still shows the full SHAP breakdown."
        )

st.markdown('</div>', unsafe_allow_html=True)  # close shap-panel


# ── Feature contributions table ───────────────────────────────────────────────
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
try:
    if isinstance(shap_values, list):
        sv = shap_values[pred_class] if task_type == 'classification' else shap_values[0]
    else:
        sv = shap_values
    sv_flat = np.array(sv).flatten()

    feat_imp = (
        pd.DataFrame({
            "Feature":    [_friendly(f) for f in pipeline['selected_features']],
            "SHAP Value": sv_flat,
            "Impact":     ["↑ Increases" if v > 0 else "↓ Decreases" for v in sv_flat],
        })
        .iloc[pd.Series(np.abs(sv_flat)).sort_values(ascending=False).index]
    )

    active_features = feat_imp[np.abs(feat_imp["SHAP Value"]) > 0.001].head(3)
    if len(active_features) > 0:
        details = []
        for _, row in active_features.iterrows():
            direction = "increased the outcome" if row['SHAP Value'] > 0 else "decreased the outcome"
            details.append(f"**{row['Feature']}** ({direction})")
        
        if len(details) == 1:
            reasons = details[0]
        elif len(details) == 2:
            reasons = f"{details[0]} and {details[1]}"
        else:
            reasons = f"{details[0]}, followed by {details[1]}, and {details[2]}"
        
        pred_label_str = band if task_type == 'regression' else label
        explanation_html = f"<div style='background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.3); padding:1.2rem; border-radius:12px; color:#E0F2FE; margin-bottom:1.5rem; line-height:1.6; font-size:0.9rem;'><strong style='color:#38BDF8;font-size:1.05rem;'><svg viewBox='0 0 24 24' style='width:18px;height:18px;vertical-align:-3px;margin-right:6px;' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'></circle><line x1='12' y1='16' x2='12' y2='12'></line><line x1='12' y1='8' x2='12.01' y2='8'></line></svg>AI Explanation</strong><br><br>The model predicted <strong>{pred_label_str}</strong>. Based on SHAP analysis, the most significant factors driving this specific prediction were {reasons}.</div>"
        st.markdown(explanation_html, unsafe_allow_html=True)

    st.markdown("""
    <div class="shap-panel">
        <div class="shap-panel-title">
            <span class="section-icon">
                <svg viewBox='0 0 24 24'>
                    <line x1='8'  y1='6'  x2='21' y2='6'/>
                    <line x1='8'  y1='12' x2='21' y2='12'/>
                    <line x1='8'  y1='18' x2='21' y2='18'/>
                    <line x1='3'  y1='6'  x2='3.01' y2='6'/>
                    <line x1='3'  y1='12' x2='3.01' y2='12'/>
                    <line x1='3'  y1='18' x2='3.01' y2='18'/>
                </svg>
            </span>
            Top Feature Contributions
        </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        feat_imp.head(10)
            .style.background_gradient(subset=["SHAP Value"], cmap="RdYlGn")
            .format({"SHAP Value": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)  # close shap-panel

except Exception:
    pass