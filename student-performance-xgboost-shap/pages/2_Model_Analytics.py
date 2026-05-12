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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }
[data-testid="stSidebar"] { background: rgba(255,255,255,0.04); border-right: 1px solid rgba(255,255,255,0.08); }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

.page-header {
    background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
    border: 1px solid rgba(102,126,234,0.3);
    border-radius: 16px; padding: 1.8rem 2rem; margin-bottom: 1.8rem;
}
.page-header h1 { font-size: 1.8rem; font-weight: 800; color: #e2e8f0 !important; margin: 0 0 0.4rem 0; }
.page-header p  { color: #94a3b8; font-size: 0.92rem; margin: 0; }

/* Metric card */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px; padding: 1.4rem 1.2rem;
    text-align: center; margin-bottom: 1rem;
}
.metric-card .m-val {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(90deg, #667eea, #f093fb);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-card .m-label { color: #64748b; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.3rem; }
.metric-card .m-sub   { color: #94a3b8; font-size: 0.88rem; margin-top: 0.4rem; }

/* Report box */
.report-box {
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 1.4rem;
    font-family: 'Courier New', monospace; font-size: 0.82rem;
    color: #a0e9a0; white-space: pre; overflow-x: auto;
}

/* Section label */
.section-label {
    color: #a78bfa; font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    margin: 1.5rem 0 0.8rem 0;
}

/* SHAP card */
.shap-global-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px; padding: 1.6rem; margin-top: 1rem;
}
.shap-global-card h3 { color: #e2e8f0; font-size: 1rem; font-weight: 700; margin: 0 0 1rem 0; }

/* Dataset tab pills */
[data-baseweb="tab-list"] { gap: 8px; }
[data-baseweb="tab"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important; color: #94a3b8 !important;
    font-weight: 600 !important; padding: 0.5rem 1.4rem !important;
}
[aria-selected="true"] {
    background: linear-gradient(90deg,#667eea,#764ba2) !important;
    color: white !important; border-color: transparent !important;
}

h1,h2,h3,h4 { color: #e2e8f0 !important; }
p { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = Path("models")

st.markdown("""
<div class="page-header">
    <h1>📊 Model Performance Analytics</h1>
    <p>5-Fold Cross-Validation · Precision/Recall/F1 · Confusion Matrices · ROC Curves · Global SHAP Analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Dataset config ────────────────────────────────────────────────────────────
datasets = [
    ("UCI — High School",  "uci",   "regression",     "📈", "#34d399"),
    ("xAPI — K-12 Online", "xapi",  "classification", "🏷️", "#60a5fa"),
    ("OULAD — University", "oulad", "classification", "🎓", "#fb923c"),
]
tabs = st.tabs([f"{ico} {name}" for name, _, _, ico, _ in datasets])

for tab, (name, key, task, ico, colour) in zip(tabs, datasets):
    with tab:
        report_path = MODELS_DIR / f"{key}_report.txt"

        if not report_path.exists():
            st.warning(f"No results found for **{name}**. Run `python src/train.py` to train the models.")
            continue

        # Parse report
        with open(report_path) as f:
            raw = f.read()

        # ── Pull headline numbers ─────────────────────────────────────────
        cv_score, cv_std, acc, macro_f1 = None, None, None, None
        for line in raw.splitlines():
            if "5-Fold Cross Validation" in line:
                parts = line.split(":")[-1].strip().split("±") if "±" in line else line.split(":")[-1].strip().split("+/-")
                if len(parts) == 2:
                    cv_score = float(parts[0].strip())
                    cv_std   = float(parts[1].strip())
            if "accuracy" in line and "weighted" not in line and "macro" not in line and acc is None:
                try: acc = float(line.strip().split()[-2])
                except: pass
            if "macro avg" in line:
                try: macro_f1 = float(line.strip().split()[-2])
                except: pass
            if "R² Score:" in line:
                try: acc = float(line.strip().split(":")[-1].strip())
                except: pass

        # ── Metric cards ──────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        metric_label = "R² Score" if task == 'regression' else "Accuracy"
        with c1:
            cv_disp = f"{cv_score:.4f}" if cv_score is not None else "—"
            std_disp = f"± {cv_std:.4f}" if cv_std is not None else ""
            st.markdown(f"""<div class="metric-card">
                <div class="m-val">{cv_disp}</div>
                <div class="m-label">5-Fold CV Score</div>
                <div class="m-sub">{std_disp}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            acc_disp = f"{acc:.4f}" if acc is not None else "—"
            st.markdown(f"""<div class="metric-card">
                <div class="m-val">{acc_disp}</div>
                <div class="m-label">Test {metric_label}</div>
                <div class="m-sub">Hold-out 20% split</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            f1_disp = f"{macro_f1:.4f}" if macro_f1 is not None else "N/A"
            st.markdown(f"""<div class="metric-card">
                <div class="m-val">{f1_disp}</div>
                <div class="m-label">Macro F1-Score</div>
                <div class="m-sub">Across all classes</div>
            </div>""", unsafe_allow_html=True)

        # ── Classification report ─────────────────────────────────────────
        st.markdown('<div class="section-label">📋 Full Classification Report</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="report-box">{raw}</div>', unsafe_allow_html=True)

        # ── Visual metrics ────────────────────────────────────────────────
        cm_path  = MODELS_DIR / f"{key}_cm.png"
        roc_path = MODELS_DIR / f"{key}_roc.png"

        if cm_path.exists() or roc_path.exists():
            st.markdown('<div class="section-label">📉 Visual Evaluation</div>', unsafe_allow_html=True)
            vcol1, vcol2 = st.columns(2)
            with vcol1:
                if cm_path.exists():
                    st.markdown("**Confusion Matrix**")
                    st.image(Image.open(cm_path), use_container_width=True)
                    st.caption("Diagonal = correct predictions. Off-diagonal = misclassifications.")
            with vcol2:
                if roc_path.exists():
                    st.markdown("**ROC Curve (One-vs-Rest)**")
                    st.image(Image.open(roc_path), use_container_width=True)
                    st.caption("AUC closer to 1.0 = stronger class separation.")

        # ── Global SHAP (load explainer + sample of training data) ────────
        explainer_path = MODELS_DIR / f"{key}_explainer.pkl"
        pipeline_path  = MODELS_DIR / f"{key}_pipeline.pkl"

        if explainer_path.exists() and pipeline_path.exists():
            st.markdown('<div class="section-label">🧠 Global SHAP Feature Importance</div>', unsafe_allow_html=True)
            with st.spinner("Generating global SHAP summary..."):
                try:
                    with open(explainer_path, "rb") as f: explainer = pickle.load(f)
                    with open(pipeline_path,  "rb") as f: pl = pickle.load(f)

                    # Build a small representative sample for global SHAP
                    # Use the background kmeans data stored in the explainer
                    bg_data = explainer.data  # kmeans background points
                    bg_df   = pd.DataFrame(bg_data, columns=pl['selected_features'])

                    shap_vals = explainer.shap_values(bg_df)

                    # Mean absolute SHAP across classes (or single array)
                    if isinstance(shap_vals, list):
                        mean_shap = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
                    else:
                        mean_shap = np.abs(shap_vals)

                    mean_per_feature = mean_shap.mean(axis=0)
                    feat_names = pl['selected_features']

                    # Sort and plot
                    sorted_idx = np.argsort(mean_per_feature)[-15:]
                    fig, ax = plt.subplots(figsize=(8, 5))
                    fig.patch.set_facecolor('#1a1a2e')
                    ax.set_facecolor('#1a1a2e')

                    bars = ax.barh(
                        [feat_names[i] for i in sorted_idx],
                        mean_per_feature[sorted_idx],
                        color=[f"#{hex(int(102 + (234-102)*j/len(sorted_idx)))[2:].zfill(2)}"
                               f"7e{hex(int(126 + (164-126)*j/len(sorted_idx)))[2:].zfill(2)}"
                               for j in range(len(sorted_idx))]
                    )
                    # Nice gradient using colormap
                    cmap = plt.cm.get_cmap('plasma')
                    for bar, val in zip(bars, mean_per_feature[sorted_idx]):
                        bar.set_color(cmap(val / (mean_per_feature.max() + 1e-9)))

                    ax.set_xlabel("Mean |SHAP Value|", color='#94a3b8', fontsize=10)
                    ax.set_title(f"Global Feature Importance — {name}", color='#e2e8f0', fontsize=12, fontweight='bold')
                    ax.tick_params(colors='#94a3b8', labelsize=9)
                    for spine in ax.spines.values(): spine.set_edgecolor('#2d2d4e')
                    ax.xaxis.label.set_color('#94a3b8')
                    plt.tight_layout()

                    st.markdown('<div class="shap-global-card"><h3>🎯 Top Predictive Features (Global SHAP)</h3>', unsafe_allow_html=True)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    # Table
                    shap_df = pd.DataFrame({
                        "Feature": feat_names,
                        "Mean |SHAP|": mean_per_feature
                    }).sort_values("Mean |SHAP|", ascending=False).head(10).reset_index(drop=True)
                    shap_df.index += 1
                    st.dataframe(
                        shap_df.style.background_gradient(subset=["Mean |SHAP|"], cmap="plasma")
                                     .format({"Mean |SHAP|": "{:.5f}"}),
                        use_container_width=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.info(f"Global SHAP preview unavailable: {e}")
