import streamlit as st
from pathlib import Path
from PIL import Image
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Model Analytics — EduPredict", page_icon="📊", layout="wide")

# ── Global Design System ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: #060818;
    background-image:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(99,102,241,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(168,85,247,0.14) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 55% 50%, rgba(236,72,153,0.08) 0%, transparent 50%);
    min-height: 100vh;
}

[data-testid="stSidebar"] {
    background: rgba(6,8,24,0.9) !important;
    border-right: 1px solid rgba(99,102,241,0.18) !important;
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

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
    background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(168,85,247,0.08));
    border: 1px solid rgba(99,102,241,0.22);
    border-radius: 20px;
    padding: 2rem 2.4rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
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
.page-header-tag { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #818cf8; margin-bottom: 0.6rem; }
.page-header h1  { font-size: 1.9rem !important; font-weight: 800 !important; color: #f1f5f9 !important; margin: 0 0 0.5rem 0 !important; letter-spacing: -0.02em; }
.page-header p   { color: #94a3b8; font-size: 0.9rem; margin: 0; }

/* ── Dataset tabs ─────────────────────────── */
[data-baseweb="tab-list"] { gap: 8px !important; background: transparent !important; border-bottom: none !important; }
[data-baseweb="tab"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 12px !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
}
[data-baseweb="tab"]:hover { background: rgba(99,102,241,0.1) !important; border-color: rgba(99,102,241,0.3) !important; }
[aria-selected="true"] {
    background: linear-gradient(135deg, rgba(79,70,229,0.5), rgba(124,58,237,0.4)) !important;
    border-color: rgba(99,102,241,0.5) !important;
    color: #e0e7ff !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.25) !important;
}

/* ── Metric cards ─────────────────────────── */
.metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 1.6rem 1.4rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(99,102,241,0.06), transparent);
    opacity: 0;
    transition: opacity 0.2s;
}
.metric-card:hover { transform: translateY(-4px); border-color: rgba(99,102,241,0.3); box-shadow: 0 16px 32px rgba(99,102,241,0.1); }
.metric-card:hover::before { opacity: 1; }
.metric-icon { font-size: 1.6rem; margin-bottom: 0.6rem; display: block; }
.metric-val {
    font-size: 2.1rem; font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #e879f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block; line-height: 1.1; margin-bottom: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}
.metric-label { color: #64748b; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }
.metric-sub   { color: #94a3b8; font-size: 0.8rem; margin-top: 0.35rem; }

/* ── Section divider ─────────────────────── */
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #818cf8;
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.3), transparent);
}

/* ── Report box ──────────────────────────── */
.report-box {
    background: rgba(0,0,0,0.4);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #86efac;
    white-space: pre;
    overflow-x: auto;
    line-height: 1.7;
}

/* ── Image panels ─────────────────────────── */
.img-panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 1.4rem;
}
.img-panel-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 0.9rem;
}
.img-caption { color: #475569; font-size: 0.75rem; margin-top: 0.6rem; text-align: center; font-style: italic; }

/* ── SHAP panel ──────────────────────────── */
.shap-global-panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 1.6rem;
    margin-top: 0.5rem;
}
.shap-global-title {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.shap-global-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.3), transparent);
}

/* ── Gap score badge ─────────────────────── */
.gap-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.25rem 0.8rem; border-radius: 100px;
    font-size: 0.72rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.gap-good { background: rgba(52,211,153,0.12); border: 1px solid rgba(52,211,153,0.3); color: #34d399; }
.gap-warn { background: rgba(251,191,36,0.12);  border: 1px solid rgba(251,191,36,0.3);  color: #fbbf24; }
.gap-bad  { background: rgba(251,113,133,0.12); border: 1px solid rgba(251,113,133,0.3); color: #fb7185; }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = Path("models")

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-header-tag">📊 Evaluation Dashboard</div>
    <h1>Model Analytics</h1>
    <p>5-Fold Cross-Validation · Precision / Recall / F1 · Confusion Matrices ·
       ROC Curves · Global SHAP Feature Importance</p>
</div>
""", unsafe_allow_html=True)

# ── Dataset tabs ──────────────────────────────────────────────────────────────
datasets = [
    ("UCI — High School",          "uci",     "regression",     "📈", "#34d399", 0.8784, 0.0367, 0.8265),
    ("xAPI — K-12 Online",         "xapi",    "classification", "🏷️", "#60a5fa", 0.7713, 0.0278, 0.7500),
    ("UCI Dropout & Success",      "dropout", "classification", "🎓", "#fb923c", 0.0,    0.0,    0.0),
]
tabs = st.tabs([f"{ico} {name}" for name, _, _, ico, _, _, _, _ in datasets])

for tab, (name, key, task, ico, colour, cv_known, std_known, test_known) in zip(tabs, datasets):
    with tab:
        report_path = MODELS_DIR / f"{key}_report.txt"

        if not report_path.exists():
            st.warning(f"No results for **{name}**. Run `python src/train.py` first.")
            continue

        with open(report_path) as f:
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
                gap_cls, gap_note = "gap-good", "✅ Minimal gap"
            elif abs(gap) < 0.08:
                gap_cls, gap_note = "gap-warn", "⚠️ Moderate gap"
            else:
                gap_cls, gap_note = "gap-bad",  "❌ Significant gap"
            gap_html = f'<span class="gap-badge {gap_cls}">{gap_note} ({gap_sign}{gap:.4f})</span>'

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
                <span class="metric-icon">🔄</span>
                <span class="metric-val">{cv_disp}</span>
                <div class="metric-label">5-Fold CV Score</div>
                <div class="metric-sub">{std_disp}</div>
            </div>
            <div class="metric-card">
                <span class="metric-icon">🎯</span>
                <span class="metric-val">{acc_disp}</span>
                <div class="metric-label">Test {metric_label}</div>
                {extras if extras else '<div class="metric-sub">Hold-out set</div>'}
            </div>
            <div class="metric-card">
                <span class="metric-icon">📐</span>
                <span class="metric-val">{f1_disp}</span>
                <div class="metric-label">Macro F1-Score</div>
                <div class="metric-sub">All classes avg</div>
            </div>
        </div>
        {f'<div style="margin-bottom:1.5rem">{gap_html}</div>' if gap_html else ''}
        """, unsafe_allow_html=True)

        # ── Full report ────────────────────────────────────────────────────
        st.markdown('<div class="section-label">📋 Full Classification Report</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="report-box">{raw}</div>', unsafe_allow_html=True)

        # ── Visual metrics ─────────────────────────────────────────────────
        cm_path  = MODELS_DIR / f"{key}_cm.png"
        roc_path = MODELS_DIR / f"{key}_roc.png"

        if cm_path.exists() or roc_path.exists():
            st.markdown('<div class="section-label">📉 Visual Evaluation</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="section-label">🧠 Global SHAP Feature Importance</div>', unsafe_allow_html=True)
            with st.spinner("Generating global SHAP summary..."):
                try:
                    with open(explainer_path, "rb") as f: explainer = pickle.load(f)
                    with open(pipeline_path,  "rb") as f: pl = pickle.load(f)

                    bg_data = explainer.data
                    bg_df   = pd.DataFrame(bg_data, columns=pl['selected_features'])
                    shap_vals = explainer.shap_values(bg_df)

                    if isinstance(shap_vals, list):
                        mean_shap = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
                    else:
                        mean_shap = np.abs(shap_vals)

                    mean_per_feature = mean_shap.mean(axis=0)
                    feat_names = pl['selected_features']
                    sorted_idx = np.argsort(mean_per_feature)[-15:]

                    # ── Chart ──────────────────────────────────────────────
                    fig, ax = plt.subplots(figsize=(9, 5))
                    fig.patch.set_facecolor('#0a0d1e')
                    ax.set_facecolor('#0a0d1e')

                    cmap = plt.cm.get_cmap('cool')
                    vals = mean_per_feature[sorted_idx]
                    norm_vals = vals / (vals.max() + 1e-9)
                    bar_colors = [cmap(v) for v in norm_vals]

                    bars = ax.barh(
                        [feat_names[i] for i in sorted_idx],
                        vals,
                        color=bar_colors,
                        height=0.65,
                        edgecolor='none',
                    )

                    # Value labels on bars
                    for bar, v in zip(bars, vals):
                        ax.text(
                            bar.get_width() + vals.max() * 0.01,
                            bar.get_y() + bar.get_height() / 2,
                            f"{v:.4f}", va='center', ha='left',
                            color='#64748b', fontsize=7.5,
                            fontfamily='monospace'
                        )

                    ax.set_xlabel("Mean |SHAP Value|", color='#64748b', fontsize=9, labelpad=8)
                    ax.set_title(f"Top Feature Drivers — {name}", color='#c7d2fe', fontsize=11, fontweight='bold', pad=14)
                    ax.tick_params(colors='#94a3b8', labelsize=8.5)
                    ax.set_xlim(0, vals.max() * 1.18)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.xaxis.set_tick_params(length=0)
                    ax.yaxis.set_tick_params(length=0)
                    ax.grid(axis='x', color='rgba(255,255,255,0.04)', linewidth=0.8)
                    plt.tight_layout()

                    st.markdown('<div class="shap-global-panel"><div class="shap-global-title">🎯 Global SHAP Bar Chart</div>', unsafe_allow_html=True)
                    st.pyplot(fig, use_container_width=True)
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
