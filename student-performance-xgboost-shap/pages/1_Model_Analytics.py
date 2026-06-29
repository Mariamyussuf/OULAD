import streamlit as st
from pathlib import Path
from PIL import Image
import pickle
import pandas as pd
import numpy as np
import shap

# ── Monkeypatch SHAP KernelExplainer to fix empty run broadcast bug ──────────
_orig_kernel_run = shap.explainers._kernel.KernelExplainer.run
def _patched_kernel_run(self):
    if self.nsamplesAdded == self.nsamplesRun:
        return
    return _orig_kernel_run(self)
shap.explainers._kernel.KernelExplainer.run = _patched_kernel_run

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Model Analytics — EduPredict", page_icon="◈", layout="wide")

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

/* ── Dataset tabs ─────────────────────────── */
[data-baseweb="tab-list"] { gap: 8px !important; background: transparent !important; border-bottom: none !important; }
[data-baseweb="tab"] {
    background: #111827 !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
    color: #94A3B8 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.15s ease !important;
}
[data-baseweb="tab"]:hover { background: #1E293B !important; border-color: #334155 !important; color: #F1F5F9 !important; }
[aria-selected="true"] {
    background: #1E293B !important;
    border-color: #06B6D4 !important;
    color: #F8FAFC !important;
}

/* ── Metric cards ─────────────────────────── */
.metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 1.8rem 1.4rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: var(--card-accent, #38BDF8);
}
.metric-card:nth-child(1) { --card-accent: #38BDF8; }
.metric-card:nth-child(2) { --card-accent: #34D399; }
.metric-card:nth-child(3) { --card-accent: #F472B6; }
.metric-val {
    font-size: 2rem; font-weight: 700;
    color: #F8FAFC;
    display: block; line-height: 1.2; margin-bottom: 0.35rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.03em;
}
.metric-card:nth-child(1) .metric-val { color: #38BDF8; }
.metric-card:nth-child(2) .metric-val { color: #34D399; }
.metric-card:nth-child(3) .metric-val { color: #F472B6; }
.metric-label { color: #64748B; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }
.metric-sub   { color: #64748B; font-size: 0.72rem; margin-top: 0.4rem; font-family: 'JetBrains Mono', monospace; }

/* ── Section divider ─────────────────────── */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #38BDF8;
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1E293B;
}

/* ── Report box ──────────────────────────── */
.report-box {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-left: 3px solid #059669;
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.79rem;
    color: #6EE7B7;
    white-space: pre;
    overflow-x: auto;
    line-height: 1.75;
}

/* ── Image panels ─────────────────────────── */
.img-panel {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 1.4rem;
}
.img-panel img, .img-panel [data-testid="stImage"] {
    background-color: #ffffff !important;
    border-radius: 8px;
    padding: 8px;
}
.img-panel-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #38BDF8;
    margin-bottom: 0.9rem;
    font-family: 'JetBrains Mono', monospace;
}
.img-caption { color: #64748B; font-size: 0.75rem; margin-top: 0.6rem; text-align: center; font-style: italic; }

/* ── SHAP panel ──────────────────────────── */
.shap-global-panel {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 1.6rem;
    margin-top: 0.5rem;
}
.shap-global-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #38BDF8;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.shap-global-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1E293B;
}

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

/* ── Tab icon ─────────────────────────────── */
.tab-icon-wrap { display: inline-flex; align-items: center; gap: 0.45rem; }
.tab-icon-wrap svg {
    width: 13px; height: 13px;
    stroke: currentColor; stroke-width: 2;
    stroke-linecap: round; stroke-linejoin: round; fill: none;
    flex-shrink: 0;
}

/* ── Metric card icon ────────────────────── */
.metric-icon-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 38px; height: 38px;
    border-radius: 8px;
    margin: 0 auto 0.8rem;
    background: #1E293B;
    border: 1px solid #334155;
    color: var(--card-accent, #38BDF8);
}
.metric-icon-wrap svg {
    width: 18px; height: 18px;
    stroke: currentColor; stroke-width: 1.75;
    stroke-linecap: round; stroke-linejoin: round; fill: none;
}

/* ── Gap badge icon ──────────────────────── */
.gap-badge-icon { display: inline-flex; align-items: center; gap: 0.35rem; }
.gap-badge-icon svg {
    width: 11px; height: 11px;
    stroke: currentColor; stroke-width: 2.25;
    stroke-linecap: round; stroke-linejoin: round; fill: none;
}

/* ── Gap score badge ─────────────────────── */
.gap-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.25rem 0.8rem; border-radius: 100px;
    font-size: 0.72rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.gap-good { background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.3); color: #34D399; }
.gap-warn { background: rgba(251,191,36,0.1);  border: 1px solid rgba(251,191,36,0.3);  color: #FBBF24; }
.gap-bad  { background: rgba(251,113,133,0.1); border: 1px solid rgba(251,113,133,0.3); color: #F87171; }

/* ── Expander styling ────────────────────── */
[data-testid="stExpander"] {
    background: #111827 !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
    margin-bottom: 1rem !important;
}
[data-testid="stExpander"] summary { color: #F1F5F9 !important; font-weight: 700 !important; font-size: 0.86rem !important; }
[data-testid="stExpander"] > div > div { border-top: 1px solid #1E293B !important; }

/* ── Widget label overrides ──────────────── */
.stSelectbox label, .stSlider label, .stCheckbox label {
    color: #94A3B8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

MODELS_DIR = Path("models")

# ── Inline SVG icon set (stroke-based, inherits currentColor) ─────────────────
SVG_REFRESH    = "<svg viewBox='0 0 24 24'><polyline points='23 4 23 10 17 10'/><polyline points='1 20 1 14 7 14'/><path d='M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15'/></svg>"
SVG_TARGET     = "<svg viewBox='0 0 24 24'><circle cx='12' cy='12' r='10'/><circle cx='12' cy='12' r='6'/><circle cx='12' cy='12' r='2'/></svg>"
SVG_RULER      = "<svg viewBox='0 0 24 24'><path d='M21.3 8.7L8.7 21.3a1 1 0 0 1-1.4 0l-4.6-4.6a1 1 0 0 1 0-1.4L15.3 2.7a1 1 0 0 1 1.4 0l4.6 4.6a1 1 0 0 1 0 1.4z'/><path d='M7.5 10.5l2 2M10.5 7.5l2 2M13.5 4.5l2 2M4.5 13.5l2 2'/></svg>"
SVG_FILE_TEXT  = "<svg viewBox='0 0 24 24'><path d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/><polyline points='14 2 14 8 20 8'/><line x1='16' y1='13' x2='8' y2='13'/><line x1='16' y1='17' x2='8' y2='17'/><polyline points='10 9 9 9 8 9'/></svg>"
SVG_LINE_CHART = "<svg viewBox='0 0 24 24'><polyline points='22 12 18 12 15 21 9 3 6 12 2 12'/></svg>"
SVG_BRAIN      = "<svg viewBox='0 0 24 24'><path d='M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.94-3.04A2.5 2.5 0 0 1 9.5 2z'/><path d='M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.94-3.04A2.5 2.5 0 0 0 14.5 2z'/></svg>"
SVG_TARGET_SM  = "<svg viewBox='0 0 24 24'><circle cx='12' cy='12' r='10'/><circle cx='12' cy='12' r='6'/><circle cx='12' cy='12' r='2'/></svg>"

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-header-tag"><span class="header-icon"><svg viewBox='0 0 24 24'><line x1='18' y1='20' x2='18' y2='10'/><line x1='12' y1='20' x2='12' y2='4'/><line x1='6' y1='20' x2='6' y2='14'/></svg>Evaluation Dashboard</span></div>
    <h1>Model Analytics</h1>
    <p>5-Fold Cross-Validation · Precision / Recall / F1 · Confusion Matrices ·
       ROC Curves · Global SHAP Feature Importance</p>
</div>
""", unsafe_allow_html=True)

# ── Presentation Settings (inline) ────────────────────────────────────────────
with st.expander("📊 Presentation Controls", expanded=False):
    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        chart_theme = st.selectbox(
            "Chart Theme Style",
            ["High-Contrast Light (For Slides/Projectors)", "EduPredict Dark (Seamless Dashboard Theme)"],
            help="Under bright classroom or auditorium lights, a white-background chart has much better visibility."
        )
    with pc2:
        font_scale = st.select_slider(
            "Label Font Size Scale",
            options=["Standard", "Large", "Extra Large"],
            value="Standard",
            help="Increase this if people in the back rows cannot read the labels."
        )
    with pc3:
        num_features = st.slider(
            "Max Features to Display",
            min_value=5,
            max_value=20,
            value=15,
            help="Show more or fewer variables in the driver list."
        )
    with pc4:
        show_grid = st.checkbox(
            "Show Vertical Grid Lines",
            value=True
        )

# ── Dataset tabs ──────────────────────────────────────────────────────────────
datasets = [
    ("UCI — High School",          "uci",     "regression",     "REG", "#34d399", 0.8784, 0.0367, 0.8265),
    ("xAPI — K-12 Online",         "xapi",    "classification", "CLF", "#60a5fa", 0.7713, 0.0278, 0.7500),
    ("UCI Dropout & Success",      "dropout", "classification", "CLF", "#fb923c", 0.0,    0.0,    0.0),
]
tabs = st.tabs([f"{name}" for name, _, _, _, _, _, _, _ in datasets])

for tab, (name, key, task, ico, colour, cv_known, std_known, test_known) in zip(tabs, datasets):
    with tab:
        report_path = MODELS_DIR / f"{key}_report.txt"

        if not report_path.exists():
            st.warning(f"No results for **{name}**. Run `python src/train.py` first.")
            continue

        try:
            with open(report_path, encoding='utf-8') as f:
                raw = f.read()
        except UnicodeDecodeError:
            with open(report_path, encoding='cp1252') as f:
                raw = f.read()

        # ── Parse headline numbers ─────────────────────────────────────────
        cv_score, cv_std, acc, macro_f1 = None, None, None, None
        rmse_val = None
        for line in raw.splitlines():
            if "5-Fold Cross Validation" in line:
                parts = (line.split(":")[-1].strip().split("±")
                         if "±" in line
                         else line.split(":")[-1].strip().split("+/-"))
                if len(parts) == 2:
                    try:
                        cv_score = float(parts[0].strip())
                        cv_std   = float(parts[1].strip())
                    except: pass
            if "accuracy" in line and "weighted" not in line and "macro" not in line and acc is None:
                try: acc = float(line.strip().split()[-2])
                except: pass
            if "macro avg" in line:
                try: macro_f1 = float(line.strip().split()[-2])
                except: pass
            if "R² Score:" in line:
                try: acc = float(line.strip().split(":")[-1].strip())
                except: pass
            if "RMSE:" in line:
                try: rmse_val = float(line.strip().split(":")[-1].strip())
                except: pass

        # ── CV vs Test gap badge ───────────────────────────────────────────
        gap_html = ""
        if cv_score is not None and acc is not None:
            gap = acc - cv_score
            gap_sign = "+" if gap >= 0 else ""
            if abs(gap) < 0.03:
                gap_cls, gap_icon, gap_note = "gap-good", "<svg viewBox='0 0 24 24'><polyline points='20 6 9 17 4 12'/></svg>", "Minimal gap"
            elif abs(gap) < 0.08:
                gap_cls, gap_icon, gap_note = "gap-warn", "<svg viewBox='0 0 24 24'><path d='M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z'/><line x1='12' y1='9' x2='12' y2='13'/><line x1='12' y1='17' x2='12.01' y2='17'/></svg>", "Moderate gap"
            else:
                gap_cls, gap_icon, gap_note = "gap-bad",  "<svg viewBox='0 0 24 24'><circle cx='12' cy='12' r='10'/><line x1='15' y1='9' x2='9' y2='15'/><line x1='9' y1='9' x2='15' y2='15'/></svg>", "Significant gap"
            gap_html = f'<span class="gap-badge {gap_cls}"><span class="gap-badge-icon">{gap_icon}{gap_note} ({gap_sign}{gap:.4f})</span></span>'

        # ── Metric cards ───────────────────────────────────────────────────
        cv_disp  = f"{cv_score:.4f}" if cv_score is not None else "—"
        std_disp = f"±{cv_std:.4f}"  if cv_std   is not None else ""
        acc_disp = f"{acc:.4f}"      if acc       is not None else "—"
        f1_disp  = f"{macro_f1:.4f}" if macro_f1  is not None else "N/A"
        metric_label = "R² Score" if task == 'regression' else "Accuracy"

        extras = ""
        if task == 'regression' and rmse_val is not None:
            extras = f'<div class="metric-sub">RMSE: {rmse_val:.4f}</div>'

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-icon-wrap">{SVG_REFRESH}</div>
                <span class="metric-val">{cv_disp}</span>
                <div class="metric-label">5-Fold CV Score</div>
                <div class="metric-sub">{std_disp}</div>
            </div>
            <div class="metric-card">
                <div class="metric-icon-wrap">{SVG_TARGET}</div>
                <span class="metric-val">{acc_disp}</span>
                <div class="metric-label">Test {metric_label}</div>
                {extras if extras else '<div class="metric-sub">Hold-out set</div>'}
            </div>
            <div class="metric-card">
                <div class="metric-icon-wrap">{SVG_RULER}</div>
                <span class="metric-val">{f1_disp}</span>
                <div class="metric-label">Macro F1-Score</div>
                <div class="metric-sub">All classes avg</div>
            </div>
        </div>
        {f'<div style="margin-bottom:1.5rem">{gap_html}</div>' if gap_html else ''}
        """, unsafe_allow_html=True)

        # ── Full report ────────────────────────────────────────────────────
        st.markdown(f"<div class='section-label'><span class='section-icon'>{SVG_FILE_TEXT}</span>Full Classification Report</div>", unsafe_allow_html=True)
        st.markdown(f'<div class="report-box">{raw}</div>', unsafe_allow_html=True)

        # ── Visual metrics ─────────────────────────────────────────────────
        cm_path  = MODELS_DIR / f"{key}_cm.png"
        roc_path = MODELS_DIR / f"{key}_roc.png"

        if cm_path.exists() or roc_path.exists():
            st.markdown(f"<div class='section-label'><span class='section-icon'>{SVG_LINE_CHART}</span>Visual Evaluation</div>", unsafe_allow_html=True)
            vcol1, vcol2 = st.columns(2, gap="medium")
            with vcol1:
                if cm_path.exists():
                    st.markdown('<div class="img-panel"><div class="img-panel-title">Confusion Matrix</div>', unsafe_allow_html=True)
                    st.image(Image.open(cm_path), use_container_width=True)
                    st.markdown('<div class="img-caption">Diagonal = correct predictions · Off-diagonal = misclassifications</div></div>', unsafe_allow_html=True)
            with vcol2:
                if roc_path.exists():
                    st.markdown('<div class="img-panel"><div class="img-panel-title">ROC Curve (One-vs-Rest)</div>', unsafe_allow_html=True)
                    st.image(Image.open(roc_path), use_container_width=True)
                    st.markdown('<div class="img-caption">AUC closer to 1.0 = stronger class separation</div></div>', unsafe_allow_html=True)

        # ── Global SHAP ────────────────────────────────────────────────────
        explainer_path = MODELS_DIR / f"{key}_explainer.pkl"
        pipeline_path  = MODELS_DIR / f"{key}_pipeline.pkl"

        if explainer_path.exists() and pipeline_path.exists():
            st.markdown(f"<div class='section-label'><span class='section-icon'>{SVG_BRAIN}</span>Global SHAP Feature Importance</div>", unsafe_allow_html=True)
            with st.spinner("Generating global SHAP summary..."):
                try:
                    with open(explainer_path, "rb") as f: explainer = pickle.load(f)
                    with open(pipeline_path,  "rb") as f: pl = pickle.load(f)

                    # Rebuild explainer: wrap predict so SHAP's internal numpy arrays
                    # get column names back (stacking ensembles need named features).
                    bg_raw = explainer.data.data if hasattr(explainer.data, 'data') else np.array(explainer.data)
                    bg_df  = pd.DataFrame(bg_raw, columns=pl['selected_features'])
                    model_path = MODELS_DIR / f"{key}_model.pkl"
                    with open(model_path, "rb") as f: mdl = pickle.load(f)
                    cols = list(pl['selected_features'])
                    if hasattr(mdl, 'predict_proba'):
                        _raw_fn = mdl.predict_proba
                        def _predict(X, _fn=_raw_fn, _c=cols):
                            return _fn(pd.DataFrame(X, columns=_c))
                    else:
                        _raw_fn = mdl.predict
                        def _predict(X, _fn=_raw_fn, _c=cols):
                            return _fn(pd.DataFrame(X, columns=_c))
                    explainer = shap.KernelExplainer(_predict, bg_df)
                    shap_vals = explainer.shap_values(bg_df, nsamples=200)

                    if isinstance(shap_vals, list):
                        mean_shap = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
                    else:
                        mean_shap = np.abs(shap_vals)

                    mean_per_feature = mean_shap.mean(axis=0)
                    feat_names = pl['selected_features']
                    sorted_idx = np.argsort(mean_per_feature)[-num_features:]

                    # Generate textual explanation for global predominant factors
                    top_global_indices = np.argsort(mean_per_feature)[-3:][::-1]
                    top_global_feats = [feat_names[i].replace('_', ' ').title() for i in top_global_indices]
                    
                    if len(top_global_feats) > 0:
                        reasons = f"**{top_global_feats[0]}**"
                        if len(top_global_feats) > 1:
                            reasons += f" and **{top_global_feats[1]}**" if len(top_global_feats) == 2 else f", **{top_global_feats[1]}**"
                        if len(top_global_feats) > 2:
                            reasons += f", and **{top_global_feats[2]}**"
                            
                        global_explanation = f"<div style='background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.3); padding:1.2rem; border-radius:12px; color:#E0F2FE; margin-bottom:1.5rem; line-height:1.6; font-size:0.9rem;'><strong style='color:#38BDF8;font-size:1.05rem;'><svg viewBox='0 0 24 24' style='width:18px;height:18px;vertical-align:-3px;margin-right:6px;' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'></circle><line x1='12' y1='16' x2='12' y2='12'></line><line x1='12' y1='8' x2='12.01' y2='8'></line></svg>System Predominant Factors</strong><br><br>Across the entire student population, the most significant driving factors that affect the system's predictions overall are {reasons}. These variables carry the highest average weight in determining a student's predicted outcome.</div>"
                        st.markdown(global_explanation, unsafe_allow_html=True)

                    # ── Map presentation settings ──
                    is_light = "High-Contrast Light" in chart_theme
                    
                    # Backgrounds and colors
                    BG = '#ffffff' if is_light else '#04021a'
                    TEXT_COLOR = '#000000' if is_light else '#cbd5e1'
                    LABEL_COLOR = '#1e293b' if is_light else '#94a3b8'
                    TITLE_COLOR = '#0f172a' if is_light else '#c7d2fe'
                    GRID_COLOR = '#e2e8f0' if is_light else '#1e2040'
                    BAR_LABEL_COLOR = '#0f172a' if is_light else '#cbd5e1'
                    
                    # Font sizes mapping
                    if font_scale == "Extra Large":
                        sz_title, sz_label, sz_ticks, sz_bars = 16.0, 13.0, 12.0, 11.0
                    elif font_scale == "Large":
                        sz_title, sz_label, sz_ticks, sz_bars = 13.0, 11.0, 10.0, 9.0
                    else: # Standard
                        sz_title, sz_label, sz_ticks, sz_bars = 10.5, 8.5, 8.0, 7.5

                    # ── Chart ──────────────────────────────────────────────
                    fig, ax = plt.subplots(figsize=(9, 5))
                    fig.patch.set_facecolor(BG)
                    ax.set_facecolor(BG)

                    vals = mean_per_feature[sorted_idx]
                    norm_vals = vals / (vals.max() + 1e-9)
                    
                    from matplotlib.colors import LinearSegmentedColormap
                    if is_light:
                        # Deep Indigo to Cyan (high contrast for light background)
                        _cmap = LinearSegmentedColormap.from_list(
                            'ep', ['#312e81', '#4f46e5', '#06b6d4'], N=256
                        )
                    else:
                        # Vivid Purple to Cyan
                        _cmap = LinearSegmentedColormap.from_list(
                            'ep', ['#4f46e5', '#818cf8', '#22d3ee'], N=256
                        )
                    bar_colors = [_cmap(v) for v in norm_vals]

                    bars = ax.barh(
                        [feat_names[i] for i in sorted_idx],
                        vals,
                        color=bar_colors,
                        height=0.62,
                        edgecolor='none',
                    )

                    # Value labels on bars
                    for bar, v in zip(bars, vals):
                        ax.text(
                            bar.get_width() + vals.max() * 0.012,
                            bar.get_y() + bar.get_height() / 2,
                            f"{v:.4f}", va='center', ha='left',
                            color=BAR_LABEL_COLOR, fontsize=sz_bars,
                            fontfamily='monospace', fontweight='bold' if is_light else 'normal'
                        )

                    ax.set_xlabel("Mean |SHAP Value|", color=LABEL_COLOR, fontsize=sz_label, labelpad=8)
                    ax.set_title(f"Top Feature Drivers — {name}", color=TITLE_COLOR, fontsize=sz_title, fontweight='bold', pad=14)
                    ax.tick_params(colors=TEXT_COLOR, labelsize=sz_ticks)
                    ax.set_xlim(0, vals.max() * 1.22)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.xaxis.set_tick_params(length=0)
                    ax.yaxis.set_tick_params(length=0)
                    
                    if show_grid:
                        ax.grid(axis='x', color=GRID_COLOR, linewidth=0.8)
                    else:
                        ax.grid(False)
                    plt.tight_layout(pad=0.8)

                    if is_light:
                        st.markdown("""
                        <style>
                        .shap-global-panel {
                            background: #ffffff !important;
                            border: 1px solid #e2e8f0 !important;
                            box-shadow: 0 10px 30px rgba(0,0,0,0.06) !important;
                        }
                        .shap-global-title {
                            color: #0f172a !important;
                        }
                        .shap-global-title::after {
                            background: #E2E8F0 !important;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                    st.markdown(f"<div class='shap-global-panel'><div class='shap-global-title'><span class='section-icon'>{SVG_TARGET_SM}</span>Global SHAP Bar Chart</div>", unsafe_allow_html=True)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
                    st.image(buf, use_container_width=True)
                    plt.close(fig)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Table
                    shap_df = pd.DataFrame({
                        "Feature": feat_names,
                        "Mean |SHAP|": mean_per_feature
                    }).sort_values("Mean |SHAP|", ascending=False).head(10).reset_index(drop=True)
                    shap_df.index += 1

                    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
                    st.dataframe(
                        shap_df.style
                            .background_gradient(subset=["Mean |SHAP|"], cmap="plasma")
                            .format({"Mean |SHAP|": "{:.5f}"}),
                        use_container_width=True,
                    )

                except Exception as e:
                    st.info(f"Global SHAP preview unavailable: {e}")