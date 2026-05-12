import streamlit as st

st.set_page_config(
    page_title="EduPredict — Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── Google Font ───────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global Reset ──────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Dark background ───────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* ── Hero banner ────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, rgba(102,126,234,0.25), rgba(118,75,162,0.25));
    border: 1px solid rgba(102,126,234,0.35);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
    backdrop-filter: blur(10px);
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.6rem 0;
    line-height: 1.2;
}
.hero p {
    color: #a0aec0;
    font-size: 1.05rem;
    max-width: 700px;
    margin: 0 auto;
    line-height: 1.7;
}

/* ── Stat card ──────────────────────────────────────────── */
.stat-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.6rem 1.4rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
    backdrop-filter: blur(8px);
    height: 100%;
}
.stat-card:hover {
    transform: translateY(-4px);
    border-color: rgba(102,126,234,0.5);
}
.stat-icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
.stat-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #667eea, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: block;
}
.stat-label { color: #a0aec0; font-size: 0.82rem; margin-top: 0.3rem; letter-spacing: 0.05em; text-transform: uppercase; }

/* ── Section card ───────────────────────────────────────── */
.section-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(8px);
}
.section-card h3 {
    color: #e2e8f0;
    font-size: 1.1rem;
    font-weight: 700;
    margin: 0 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-card p, .section-card li {
    color: #94a3b8;
    font-size: 0.93rem;
    line-height: 1.75;
    margin: 0;
}

/* ── Author card ────────────────────────────────────────── */
.author-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15));
    border: 1px solid rgba(102,126,234,0.3);
    border-radius: 16px;
    padding: 2rem;
    height: 100%;
}
.author-card h3 { color: #c3d0e8; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin: 0 0 1.2rem 0; }
.author-row { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.9rem; }
.author-row .label { color: #64748b; font-size: 0.8rem; font-weight: 500; width: 90px; flex-shrink: 0; }
.author-row .value { color: #e2e8f0; font-size: 0.92rem; font-weight: 600; }

/* ── Pipeline steps ─────────────────────────────────────── */
.pipeline-step {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 1rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.pipeline-step:last-child { border-bottom: none; }
.step-num {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    font-weight: 700;
    font-size: 0.8rem;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
}
.step-content .step-title { color: #e2e8f0; font-weight: 600; font-size: 0.93rem; }
.step-content .step-desc  { color: #64748b; font-size: 0.83rem; margin-top: 0.2rem; }

/* ── Nav hint ───────────────────────────────────────────── */
.nav-hint {
    background: linear-gradient(90deg, rgba(102,126,234,0.2), rgba(240,147,251,0.2));
    border: 1px solid rgba(102,126,234,0.3);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
    color: #c3d0e8;
    font-size: 0.92rem;
}

/* ── General text override ──────────────────────────────── */
h1, h2, h3, h4 { color: #e2e8f0 !important; }
p, li { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎓 EduPredict Analytics Platform</h1>
    <p>
        Design and Implementation of a Student Performance Prediction System<br>
        using a <strong style="color:#a78bfa">Stacking Ensemble ML Model</strong>
        enhanced with <strong style="color:#f093fb">SHAP Explainability</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# ── Stats row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
stats = [
    ("📊", "3", "Datasets Evaluated"),
    ("🤖", "3-Layer", "Stacking Ensemble"),
    ("🧠", "SHAP", "XAI Integration"),
    ("🎯", "81%", "Best Accuracy (xAPI)"),
]
for col, (icon, val, label) in zip([c1, c2, c3, c4], stats):
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">{icon}</div>
            <span class="stat-value">{val}</span>
            <div class="stat-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main content ─────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    # Research aim
    st.markdown("""
    <div class="section-card">
        <h3>🔬 Research Aim</h3>
        <p>
        To design and implement an advanced student performance prediction system
        utilizing a <strong style="color:#a78bfa">Stacking Ensemble</strong> of
        Random Forest, Gradient Boosting, and XGBoost as the meta-learner —
        enhanced with <strong style="color:#f093fb">SHAP</strong> (SHapley Additive exPlanations)
        for transparent, educator-facing decision support.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline
    steps = [
        ("Data Collection", "UCI (High School), xAPI (K-12), OULAD (University Online)"),
        ("Preprocessing", "Mean/mode imputation · OrdinalEncoding · StandardScaler"),
        ("Class Balancing", "SMOTE resampling — eliminates minority-class bias"),
        ("Feature Selection", "SelectKBest with Mutual Information (top 80% features)"),
        ("Stacking Ensemble", "RF + GradientBoost → XGBoost meta-learner (tuned)"),
        ("SHAP Explainability", "KernelExplainer · force plots · global importance"),
    ]
    steps_html = "".join([
        f"""<div class="pipeline-step">
              <div class="step-num">{i+1}</div>
              <div class="step-content">
                <div class="step-title">{t}</div>
                <div class="step-desc">{d}</div>
              </div>
            </div>"""
        for i, (t, d) in enumerate(steps)
    ])
    st.markdown(f"""
    <div class="section-card">
        <h3>⚙️ System Pipeline</h3>
        {steps_html}
    </div>
    """, unsafe_allow_html=True)

with right:
    # Author card
    st.markdown("""
    <div class="author-card">
        <h3>🎓 Project Details</h3>
        <div class="author-row">
            <span class="label">Developer</span>
            <span class="value">Amromawhe Osemudiamen</span>
        </div>
        <div class="author-row">
            <span class="label">Matric No</span>
            <span class="value">2021/10248</span>
        </div>
        <div class="author-row">
            <span class="label">Institution</span>
            <span class="value">Bells University of Technology</span>
        </div>
        <div class="author-row">
            <span class="label">Supervisor</span>
            <span class="value">Dr O.J. Olaleye</span>
        </div>
        <div class="author-row">
            <span class="label">Department</span>
            <span class="value">Computer Science</span>
        </div>
        <div class="author-row">
            <span class="label">Degree</span>
            <span class="value">B.Tech Computer Science</span>
        </div>
        <div class="author-row">
            <span class="label">Year</span>
            <span class="value">December 2025</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Datasets
    st.markdown("""
    <div class="section-card">
        <h3>📂 Datasets Used</h3>
        <div class="pipeline-step">
            <div class="step-content">
                <div class="step-title">UCI Student Performance</div>
                <div class="step-desc">395 students · Regression · Final Grade G3</div>
            </div>
        </div>
        <div class="pipeline-step">
            <div class="step-content">
                <div class="step-title">xAPI Educational Data</div>
                <div class="step-desc">480 students · 3-class · Low / Medium / High</div>
            </div>
        </div>
        <div class="pipeline-step">
            <div class="step-content">
                <div class="step-title">OULAD (Open University)</div>
                <div class="step-desc">32,593 students · 4-class · Fail/Withdrawn/Pass/Distinction</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Nav hint ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-hint">
    👈 Use the <strong>sidebar</strong> to navigate —
    <strong>Prediction Engine</strong> to simulate student profiles
    &nbsp;|&nbsp;
    <strong>Model Analytics</strong> to explore evaluation metrics
</div>
""", unsafe_allow_html=True)
