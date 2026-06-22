import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
from streamlit_shap import st_shap
from pathlib import Path

st.set_page_config(
    page_title="EduPredict — Student Performance Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global Design System ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset ──────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Animated mesh background ───────────────────── */
.stApp {
    background-color: #03050f;
    background-image:
        radial-gradient(ellipse 70% 55% at 15% 5%,  rgba(79,70,229,0.22) 0%, transparent 60%),
        radial-gradient(ellipse 55% 45% at 85% 85%, rgba(139,92,246,0.18) 0%, transparent 55%),
        radial-gradient(ellipse 40% 35% at 50% 50%, rgba(236,72,153,0.06) 0%, transparent 50%);
    min-height: 100vh;
}

/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(6,8,24,0.85) !important;
    border-right: 1px solid rgba(99,102,241,0.18) !important;
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
}
[data-testid="stSidebarNav"] { padding-top: 1rem; }

/* Sidebar nav items */
[data-testid="stSidebarNav"] a {
    border-radius: 10px !important;
    margin: 2px 0 !important;
    transition: background 0.2s !important;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(99,102,241,0.15) !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: linear-gradient(90deg, rgba(99,102,241,0.25), rgba(168,85,247,0.15)) !important;
    border-left: 3px solid #818cf8 !important;
}

/* ── Typography overrides ─────────────────────────── */
h1, h2, h3, h4 { color: #f1f5f9 !important; }
p, li { color: #94a3b8; line-height: 1.7; }

/* ── Hero ────────────────────────────────────────── */
.hero-wrapper {
    position: relative;
    background: linear-gradient(135deg,
        rgba(99,102,241,0.14) 0%,
        rgba(168,85,247,0.11) 50%,
        rgba(236,72,153,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.28);
    border-top: 1px solid rgba(129,140,248,0.4);
    border-radius: 24px;
    padding: 4rem 3rem 3.5rem;
    margin-bottom: 2.5rem;
    text-align: center;
    overflow: hidden;
    backdrop-filter: blur(16px);
    box-shadow: 0 1px 0 rgba(129,140,248,0.15) inset, 0 24px 70px rgba(79,70,229,0.1);
}
.hero-wrapper::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(99,102,241,0.2) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-wrapper::after {
    content: '';
    position: absolute;
    bottom: -40px; right: -40px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(168,85,247,0.18) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 100px;
    padding: 0.35rem 1rem;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #a5b4fc;
    margin-bottom: 1.6rem;
}
.hero-title {
    font-size: clamp(2.2rem, 5vw, 3.4rem);
    font-weight: 900;
    line-height: 1.15;
    background: linear-gradient(135deg, #e2e8f0 0%, #a5b4fc 45%, #e879f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    max-width: 640px;
    margin: 0 auto 2rem;
    line-height: 1.75;
}
.hero-pills {
    display: flex;
    justify-content: center;
    gap: 0.75rem;
    flex-wrap: wrap;
}
.hero-pill {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 100px;
    padding: 0.4rem 1.1rem;
    font-size: 0.8rem;
    font-weight: 500;
    color: #cbd5e1;
}

/* ── KPI cards ───────────────────────────────────── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.kpi-card {
    background: rgba(8,6,30,0.65);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px;
    padding: 1.8rem 1.2rem;
    text-align: center;
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    cursor: default;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(99,102,241,0.06), rgba(168,85,247,0.04));
    opacity: 0;
    transition: opacity 0.25s ease;
}
.kpi-card:hover { transform: translateY(-5px); border-color: rgba(99,102,241,0.4); box-shadow: 0 20px 40px rgba(0,0,0,0.3), 0 0 0 1px rgba(99,102,241,0.12); }
.kpi-card:hover::before { opacity: 1; }
.kpi-card:nth-child(1) { --card-accent: #a5b4fc; }
.kpi-card:nth-child(2) { --card-accent: #34d399; }
.kpi-card:nth-child(3) { --card-accent: #f472b6; }
.kpi-card:nth-child(4) { --card-accent: #fbbf24; }
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 12%; right: 12%; height: 1px;
    background: linear-gradient(90deg, transparent, var(--card-accent, #818cf8), transparent);
    opacity: 0.5;
}
.kpi-icon-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 42px; height: 42px;
    border-radius: 12px;
    margin: 0 auto 0.85rem;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    color: var(--card-accent, #a5b4fc);
}
.kpi-icon-wrap svg {
    width: 20px; height: 20px;
    stroke: currentColor; stroke-width: 1.75;
    stroke-linecap: round; stroke-linejoin: round; fill: none;
}
.kpi-card:nth-child(1) .kpi-icon-wrap { background: rgba(165,180,252,0.08); border-color: rgba(165,180,252,0.18); }
.kpi-card:nth-child(2) .kpi-icon-wrap { background: rgba(52,211,153,0.08); border-color: rgba(52,211,153,0.18); }
.kpi-card:nth-child(3) .kpi-icon-wrap { background: rgba(244,114,182,0.08); border-color: rgba(244,114,182,0.18); }
.kpi-card:nth-child(4) .kpi-icon-wrap { background: rgba(251,191,36,0.08);  border-color: rgba(251,191,36,0.18); }
.kpi-val {
    font-size: 2.1rem;
    font-weight: 900;
    background: linear-gradient(135deg, #c7d2fe, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block;
    line-height: 1.1;
    margin-bottom: 0.4rem;
    letter-spacing: -0.02em;
}
.kpi-card:nth-child(2) .kpi-val { background: linear-gradient(135deg, #34d399, #6ee7b7); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.kpi-card:nth-child(3) .kpi-val { background: linear-gradient(135deg, #f472b6, #fb7185); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.kpi-card:nth-child(4) .kpi-val { background: linear-gradient(135deg, #fbbf24, #fb923c); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
}

/* ── Content panels ──────────────────────────────── */
.panel {
    background: rgba(8,6,30,0.6);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.25rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.2s;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}
.panel:hover { border-color: rgba(99,102,241,0.2); }
.panel-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.panel-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.3), transparent);
}
.panel p { color: #94a3b8; font-size: 0.93rem; line-height: 1.8; }
.panel strong { color: #c7d2fe; }

/* ── Pipeline steps ──────────────────────────────── */
.steps-list { display: flex; flex-direction: column; gap: 0; }
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 0.9rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    transition: background 0.15s;
}
.step-row:last-child { border-bottom: none; }
.step-row:hover { background: rgba(99,102,241,0.04); border-radius: 8px; }
.step-num {
    width: 30px; height: 30px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 800; color: white;
    flex-shrink: 0; margin-top: 2px;
    box-shadow: 0 4px 12px rgba(99,102,241,0.3);
}
/* .step-title flex layout defined in icon-system block below */
.step-desc  { color: #64748b; font-size: 0.8rem; margin-top: 0.15rem; line-height: 1.5; }

/* ── Author card ─────────────────────────────────── */
.author-panel {
    background: linear-gradient(160deg, rgba(79,70,229,0.10) 0%, rgba(15,10,40,0.55) 60%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}
.author-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid rgba(99,102,241,0.15);
}
.author-avatar {
    width: 54px; height: 54px;
    background: linear-gradient(135deg, #4f46e5, #9333ea);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    flex-shrink: 0;
    box-shadow: 0 6px 20px rgba(99,102,241,0.35);
}
.author-name { color: #f1f5f9; font-size: 1rem; font-weight: 700; }
.author-role { color: #818cf8; font-size: 0.78rem; font-weight: 500; margin-top: 0.2rem; }
.detail-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    gap: 1rem;
}
.detail-row:last-child { border-bottom: none; }
.detail-key { color: #475569; font-size: 0.78rem; font-weight: 500; flex-shrink: 0; }
.detail-val { color: #cbd5e1; font-size: 0.84rem; font-weight: 600; text-align: right; }

/* ── Dataset badges ──────────────────────────────── */
.dataset-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.9rem 1rem;
    border-radius: 12px;
    margin-bottom: 0.6rem;
    border: 1px solid rgba(255,255,255,0.05);
    background: rgba(255,255,255,0.02);
    transition: border-color 0.2s, background 0.2s;
}
.dataset-row:hover { border-color: rgba(99,102,241,0.25); background: rgba(99,102,241,0.05); }
.ds-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.ds-name { color: #e2e8f0; font-weight: 600; font-size: 0.88rem; }
.ds-meta { color: #64748b; font-size: 0.76rem; margin-top: 0.15rem; }

/* ── Nav bar ─────────────────────────────────────── */
.nav-bar {
    display: flex;
    gap: 1rem;
    padding: 1.2rem 1.8rem;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    align-items: center;
    justify-content: center;
    margin-top: 1rem;
}
.nav-bar span { color: #94a3b8; font-size: 0.88rem; }
.nav-bar strong { color: #a5b4fc; }
.nav-sep { color: rgba(255,255,255,0.15); }
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

/* Hero tag icon */
.hero-tag svg {
    width: 13px; height: 13px;
    stroke: currentColor; stroke-width: 2.25;
    stroke-linecap: round; stroke-linejoin: round; fill: none;
    flex-shrink: 0;
}

/* Hero pill icon */
.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
}
.hero-pill svg {
    width: 13px; height: 13px;
    stroke: #a5b4fc; stroke-width: 2;
    stroke-linecap: round; stroke-linejoin: round; fill: none;
    flex-shrink: 0;
}

/* Panel title icon */
.panel-title svg, .panel-title .section-icon svg {
    width: 13px; height: 13px;
    stroke: #818cf8; stroke-width: 2;
    stroke-linecap: round; stroke-linejoin: round; fill: none;
}
.section-icon { display: inline-flex; align-items: center; }

/* Step icon (replaces emoji inside step-title) */
.step-title {
    color: #e2e8f0; font-weight: 600; font-size: 0.9rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.step-icon-wrap {
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px;
    border-radius: 6px;
    background: rgba(99,102,241,0.12);
    color: #a5b4fc;
    flex-shrink: 0;
}
.step-icon-wrap svg {
    width: 12px; height: 12px;
    stroke: currentColor; stroke-width: 2;
    stroke-linecap: round; stroke-linejoin: round; fill: none;
}

/* Author avatar icon (replaces emoji avatar) */
.author-avatar svg {
    width: 24px; height: 24px;
    stroke: #fff; stroke-width: 1.75;
    stroke-linecap: round; stroke-linejoin: round; fill: none;
}

/* Nav bar icon */
.nav-bar { gap: 0.85rem; }
.nav-icon {
    display: inline-flex; align-items: center; gap: 0.4rem;
    vertical-align: middle;
}
.nav-icon svg {
    width: 13px; height: 13px;
    stroke: currentColor; stroke-width: 2;
    stroke-linecap: round; stroke-linejoin: round; fill: none;
    flex-shrink: 0;
}

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
.legend-swatch { width: 8px; height: 8px; border-radius: 2px; flex-shrink: 0; }

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

/* ── Streamlit widget overrides ─────────────────────────── */
.stSelectbox label, .stSlider label, .stMultiSelect label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
.stProgress > div > div { background: linear-gradient(90deg, #4f46e5, #a855f7) !important; }
.stDataFrame { border-radius: 12px; overflow: hidden; }
.stSlider > div > div > div {
    background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-tag"><svg viewBox='0 0 24 24'><path d='M22 10L12 4 2 10l10 6 10-6z'/><path d='M6 12v5c0 1.5 3 3 6 3s6-1.5 6-3v-5'/></svg>Final Year Project · B.Tech Computer Science</div>
    <div class="hero-title">EduPredict<br>Analytics Platform</div>
    <p class="hero-sub">
        <strong style="color:#a5b4fc;">Student performance prediction using XGBoost</strong>
        enhanced with <strong style="color:#e879f9;">SHAP</strong> (SHapley Additive exPlanations).
        Classification stacks diverse tree learners with a <strong>tuned XGBoost meta-learner</strong>;
        SHAP explains every prediction across benchmark educational datasets.
    </p>
    <div class="hero-pills">
        <span class="hero-pill"><svg viewBox='0 0 24 24'><rect x='4' y='4' width='16' height='16' rx='2'/><rect x='9' y='9' width='6' height='6'/><line x1='9' y1='2' x2='9' y2='4'/><line x1='15' y1='2' x2='15' y2='4'/><line x1='9' y1='20' x2='9' y2='22'/><line x1='15' y1='20' x2='15' y2='22'/><line x1='20' y1='9' x2='22' y2='9'/><line x1='20' y1='14' x2='22' y2='14'/><line x1='2' y1='9' x2='4' y2='9'/><line x1='2' y1='14' x2='4' y2='14'/></svg>XGBoost + SHAP</span>
        <span class="hero-pill"><svg viewBox='0 0 24 24'><circle cx='11' cy='11' r='8'/><line x1='21' y1='21' x2='16.65' y2='16.65'/></svg>SHAP Interpretability</span>
        <span class="hero-pill"><svg viewBox='0 0 24 24'><line x1='3' y1='6' x2='21' y2='6'/><path d='M6 6l-3 7a3 3 0 0 0 6 0z'/><path d='M18 6l-3 7a3 3 0 0 0 6 0z'/><line x1='12' y1='3' x2='12' y2='21'/><line x1='8' y1='21' x2='16' y2='21'/></svg>Stratified 5-Fold CV</span>
        <span class="hero-pill"><svg viewBox='0 0 24 24'><line x1='18' y1='20' x2='18' y2='10'/><line x1='12' y1='20' x2='12' y2='4'/><line x1='6' y1='20' x2='6' y2='14'/></svg>3 Datasets</span>
        <span class="hero-pill"><svg viewBox='0 0 24 24'><path d='M10 2v6.5a2 2 0 0 1-.6 1.4L4 15a3 3 0 0 0 2 5h12a3 3 0 0 0 2-5l-5.4-5.1a2 2 0 0 1-.6-1.4V2'/><path d='M6.5 12h11'/></svg>SMOTE Balancing</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-icon-wrap"><svg viewBox='0 0 24 24'><line x1='18' y1='20' x2='18' y2='10'/><line x1='12' y1='20' x2='12' y2='4'/><line x1='6' y1='20' x2='6' y2='14'/></svg></div>
        <span class="kpi-val">3</span>
        <div class="kpi-label">Datasets Evaluated</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon-wrap"><svg viewBox='0 0 24 24'><rect x='4' y='4' width='16' height='16' rx='2'/><rect x='9' y='9' width='6' height='6'/><line x1='9' y1='2' x2='9' y2='4'/><line x1='15' y1='2' x2='15' y2='4'/><line x1='9' y1='20' x2='9' y2='22'/><line x1='15' y1='20' x2='15' y2='22'/><line x1='20' y1='9' x2='22' y2='9'/><line x1='20' y1='14' x2='22' y2='14'/><line x1='2' y1='9' x2='4' y2='9'/><line x1='2' y1='14' x2='4' y2='14'/></svg></div>
        <span class="kpi-val">XGBoost</span>
        <div class="kpi-label">Meta-learner (classification)</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon-wrap"><svg viewBox='0 0 24 24'><path d='M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.94-3.04A2.5 2.5 0 0 1 9.5 2z'/><path d='M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.94-3.04A2.5 2.5 0 0 0 14.5 2z'/></svg></div>
        <span class="kpi-val">SHAP</span>
        <div class="kpi-label">XAI Integration</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon-wrap"><svg viewBox='0 0 24 24'><circle cx='12' cy='12' r='10'/><circle cx='12' cy='12' r='6'/><circle cx='12' cy='12' r='2'/></svg></div>
        <span class="kpi-val">CV+Test</span>
        <div class="kpi-label">Reported metrics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Main layout (Tabs) ────────────────────────────────────────────────────────
tab_predict, tab_about = st.tabs(["🔮 Interactive Prediction Engine", "📚 Project Background & Methodology"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: INTERACTIVE PREDICTION ENGINE
# ──────────────────────────────────────────────────────────────────────────────
with tab_predict:
    MODELS_DIR = Path("models")

    @st.cache_resource
    def load_model_artifacts(dataset_name):
        with open(MODELS_DIR / f"{dataset_name}_model.pkl",    "rb") as f: model    = pickle.load(f)
        with open(MODELS_DIR / f"{dataset_name}_explainer.pkl","rb") as f: explainer= pickle.load(f)
        with open(MODELS_DIR / f"{dataset_name}_pipeline.pkl", "rb") as f: pipeline = pickle.load(f)
        return model, explainer, pipeline

    # ── Dataset selector ──────────────────────────────────────────────────────
    col_sel, _ = st.columns([2, 3])
    with col_sel:
        dataset_choice = st.selectbox(
            "**Analytical Context**",
            ("UCI (High School Grades)", "xAPI (K-12 Online Learning)", "UCI Dropout & Academic Success"),
            help="Each dataset uses a separately trained stacking ensemble.",
            key="predict_ds_selector"
        )

    dataset_map = {
        "UCI (High School Grades)":          ("uci",     "regression",     "pill-uci",     "REG · Final Grade G3"),
        "xAPI (K-12 Online Learning)":       ("xapi",    "classification", "pill-xapi",    "CLF · Low / Medium / High"),
        "UCI Dropout & Academic Success":    ("dropout", "classification", "pill-oulad",   "CLF · Dropout / Enrolled / Graduate"),
    }
    dataset_key, task_type, pill_cls, pill_label = dataset_map[dataset_choice]
    st.markdown(f'<span class="ds-pill {pill_cls}">{pill_label}</span>', unsafe_allow_html=True)

    # ── Load artifacts ────────────────────────────────────────────────────────
    try:
        model, explainer, pipeline = load_model_artifacts(dataset_key)
    except FileNotFoundError:
        st.error("Model artifacts not found. Run: `python src/train.py`")
        st.stop()

    raw_features = pipeline['raw_features']
    cat_cols     = pipeline['cat_cols']
    num_cols     = pipeline['num_cols']
    cat_classes  = pipeline.get('cat_classes', {})

    # ── Profile Builder (Main Page Columns) ───────────────────────────────────
    def _max_val(f):
        fl = f.lower()
        if any(k in fl for k in ["absences","score","hands","visited","discussion","view","mean","median","max_score","min_score","std","submit"]): return 100.0
        if "credits" in fl: return 300.0
        if "attempts" in fl: return 10.0
        return 20.0

    user_input = {}
    half = len(raw_features) // 2

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-label'><span class='section-icon'><svg viewBox='0 0 24 24'><path d='M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2'/><circle cx='12' cy='7' r='4'/></svg></span>Configure Student Profile</div>", unsafe_allow_html=True)

    col_acad, col_beh = st.columns(2, gap="large")

    with col_acad:
        st.markdown("<h4 style='font-size:0.92rem;color:#818cf8 !important;margin-bottom:1rem;text-transform:uppercase;letter-spacing:0.08em;'>Academic & Demographics</h4>", unsafe_allow_html=True)
        for feature in raw_features[:half]:
            if feature in num_cols:
                mv = _max_val(feature)
                user_input[feature] = st.slider(feature.replace("_"," ").title(), 0.0, mv, mv/2, key=feature+"_home")
            elif feature in cat_cols:
                classes = cat_classes.get(feature, ["Unknown"])
                user_input[feature] = st.selectbox(feature.replace("_"," ").title(), classes, key=feature+"_home")

    with col_beh:
        st.markdown("<h4 style='font-size:0.92rem;color:#818cf8 !important;margin-bottom:1rem;text-transform:uppercase;letter-spacing:0.08em;'>Behavioural & Engagement</h4>", unsafe_allow_html=True)
        for feature in raw_features[half:]:
            if feature in num_cols:
                mv = _max_val(feature)
                user_input[feature] = st.slider(feature.replace("_"," ").title(), 0.0, mv, mv/2, key=feature+"_home_b")
            elif feature in cat_cols:
                classes = cat_classes.get(feature, ["Unknown"])
                user_input[feature] = st.selectbox(feature.replace("_"," ").title(), classes, key=feature+"_home_b")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    input_df = pd.DataFrame([user_input])
    if num_cols:
        input_df[num_cols] = pipeline['num_imputer'].transform(input_df[num_cols])
    if cat_cols:
        input_df[cat_cols] = pipeline['cat_imputer'].transform(input_df[cat_cols])
        input_df[cat_cols] = pipeline['encoder'].transform(input_df[cat_cols].astype(str))
    input_df_scaled   = pd.DataFrame(pipeline['scaler'].transform(input_df), columns=input_df.columns)
    input_df_selected = input_df_scaled[pipeline['selected_features']]

    # ── Results layout ────────────────────────────────────────────────────────
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-label'><span class='section-icon'><svg viewBox='0 0 24 24'><polyline points='22 12 18 12 15 21 9 3 6 12 2 12'/></svg></span>Model Predictions & SHAP Explanations</div>", unsafe_allow_html=True)
    res_col, shap_col = st.columns([1, 2], gap="large")

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

            # Confidence bars
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
            try:
                shap_values = explainer.shap_values(input_df_selected)
                if task_type == 'regression':
                    val     = shap_values[0] if isinstance(shap_values, list) else shap_values
                    exp_val = explainer.expected_value
                else:
                    if hasattr(explainer.expected_value, '__len__') and len(explainer.expected_value) > int(pred_class):
                        exp_val = explainer.expected_value[int(pred_class)]
                    else:
                        exp_val = explainer.expected_value
                    
                    if isinstance(shap_values, list):
                        val = shap_values[int(pred_class)]
                    elif isinstance(shap_values, np.ndarray):
                        if shap_values.ndim == 3:
                            s0, s1, s2 = shap_values.shape
                            num_classes = len(explainer.expected_value) if hasattr(explainer.expected_value, '__len__') else 3
                            if s0 == num_classes:
                                val = shap_values[int(pred_class)]
                            elif s2 == num_classes:
                                val = shap_values[:, :, int(pred_class)]
                            else:
                                val = shap_values[int(pred_class)]
                        else:
                            val = shap_values
                    else:
                        val = shap_values

                if hasattr(val, 'ndim') and val.ndim == 1:
                    val = np.expand_dims(val, axis=0)
                elif hasattr(val, 'ndim') and val.ndim == 3:
                    val = val[0]
                
                if hasattr(val, 'shape') and val.shape != input_df_selected.shape:
                    val = val.reshape(input_df_selected.shape)

                st_shap(shap.force_plot(float(exp_val), val, input_df_selected), height=180)
            except Exception as e:
                st.warning(f"Force plot unavailable: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

        # Feature contributions table
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        try:
            if task_type == 'regression':
                sv = shap_values[0] if isinstance(shap_values, list) else shap_values
            else:
                if isinstance(shap_values, list):
                    sv = shap_values[int(pred_class)]
                elif isinstance(shap_values, np.ndarray):
                    if shap_values.ndim == 3:
                        s0, s1, s2 = shap_values.shape
                        num_classes = len(explainer.expected_value) if hasattr(explainer.expected_value, '__len__') else 3
                        if s0 == num_classes:
                            sv = shap_values[int(pred_class)]
                        elif s2 == num_classes:
                            sv = shap_values[:, :, int(pred_class)]
                        else:
                            sv = shap_values[int(pred_class)]
                    else:
                        sv = shap_values
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

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: PROJECT BACKGROUND & METHODOLOGY
# ──────────────────────────────────────────────────────────────────────────────
with tab_about:
    left, right = st.columns([3, 2], gap="large")

    with left:
        # Research aim
        st.markdown("""
        <div class="panel">
            <div class="panel-title"><span class="section-icon"><svg viewBox='0 0 24 24'><path d='M9 2v6.5L4.5 18a2 2 0 0 0 1.8 3h11.4a2 2 0 0 0 1.8-3L15 8.5V2'/><line x1='9' y1='2' x2='15' y2='2'/><line x1='9' y1='13' x2='15' y2='13'/></svg></span>Research Aim</div>
            <p>
                To design and implement a student performance prediction system
                <strong>using XGBoost enhanced with SHAP</strong> (SHapley Additive exPlanations).
                The classifier uses a stacked ensemble where <strong>XGBoost is the tuned meta-learner</strong>;
                SHAP provides global and local explanations of the trained predictor for transparent,
                educator-facing decision support across benchmark educational datasets.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Pipeline
        steps = [
            ("<svg viewBox='0 0 24 24'><path d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/><polyline points='7 10 12 15 17 10'/><line x1='12' y1='15' x2='12' y2='3'/></svg>", "Data Ingestion",      "UCI · xAPI · Dropout — automatic download & caching"),
            ("<svg viewBox='0 0 24 24'><path d='M3 6h18'/><path d='M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6'/><path d='M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2'/></svg>", "Preprocessing",       "Mean/mode imputation · OrdinalEncoding · StandardScaler"),
            ("<svg viewBox='0 0 24 24'><path d='M21.3 8.7L8.7 21.3a1 1 0 0 1-1.4 0l-4.6-4.6a1 1 0 0 1 0-1.4L15.3 2.7a1 1 0 0 1 1.4 0l4.6 4.6a1 1 0 0 1 0 1.4z'/><path d='M7.5 10.5l2 2M10.5 7.5l2 2M13.5 4.5l2 2M4.5 13.5l2 2'/></svg>", "Feature Selection",   "Recursive Feature Elimination (RFE) on training fold only"),
            ("<svg viewBox='0 0 24 24'><line x1='3' y1='6' x2='21' y2='6'/><path d='M6 6l-3 7a3 3 0 0 0 6 0z'/><path d='M18 6l-3 7a3 3 0 0 0 6 0z'/><line x1='12' y1='3' x2='12' y2='21'/><line x1='8' y1='21' x2='16' y2='21'/></svg>", "Class Balancing",     "SMOTEENN on training data after split (no test leakage)"),
            ("<svg viewBox='0 0 24 24'><path d='M3 21h18'/><path d='M5 21V7l8-4v18'/><path d='M19 21V11l-6-4'/><line x1='9' y1='9' x2='9' y2='9.01'/><line x1='9' y1='12' x2='9' y2='12.01'/><line x1='9' y1='15' x2='9' y2='15.01'/><line x1='9' y1='18' x2='9' y2='18.01'/></svg>", "XGBoost-centred",     "Stacked bases (RF, GB, ET, LGBM) → tuned XGBoost meta-learner"),
            ("<svg viewBox='0 0 24 24'><line x1='18' y1='20' x2='18' y2='10'/><line x1='12' y1='20' x2='12' y2='4'/><line x1='6' y1='20' x2='6' y2='14'/></svg>", "Rigorous Evaluation", "Stratified 5-Fold CV · Held-out test set · Macro F1"),
            ("<svg viewBox='0 0 24 24'><circle cx='11' cy='11' r='8'/><line x1='21' y1='21' x2='16.65' y2='16.65'/></svg>", "SHAP Explainability", "KernelExplainer · Force plots · Global feature importance"),
        ]
        rows = "".join([
            f"""<div class="step-row">
                  <div class="step-num">{i+1}</div>
                  <div>
                    <div class="step-title"><span class="step-icon-wrap">{icon}</span>{title}</div>
                    <div class="step-desc">{desc}</div>
                  </div>
                </div>"""
            for i, (icon, title, desc) in enumerate(steps)
        ])
        st.markdown(f"""
        <div class="panel">
            <div class="panel-title"><span class="section-icon"><svg viewBox='0 0 24 24'><circle cx='12' cy='12' r='3'/><path d='M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z'/></svg></span>ML Pipeline</div>
            <div class="steps-list">{rows}</div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        # Author card
        st.markdown("""
        <div class="author-panel">
            <div class="panel-title"><span class="section-icon"><svg viewBox='0 0 24 24'><path d='M22 10L12 4 2 10l10 6 10-6z'/><path d='M6 12v5c0 1.5 3 3 6 3s6-1.5 6-3v-5'/></svg></span>Project Details</div>
            <div class="author-header">
                <div class="author-avatar"><svg viewBox='0 0 24 24'><path d='M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2'/><circle cx='12' cy='7' r='4'/></svg></div>
                <div>
                    <div class="author-name">Amromawhe Osemudiamen</div>
                    <div class="author-role">Final Year Computer Science Student</div>
                </div>
            </div>
            <div class="detail-row"><span class="detail-key">Matric No</span><span class="detail-val">2021/10248</span></div>
            <div class="detail-row"><span class="detail-key">Institution</span><span class="detail-val">Bells University of Technology</span></div>
            <div class="detail-row"><span class="detail-key">Supervisor</span><span class="detail-val">Dr O.J. Olaleye</span></div>
            <div class="detail-row"><span class="detail-key">Department</span><span class="detail-val">Computer Science</span></div>
            <div class="detail-row"><span class="detail-key">Degree</span><span class="detail-val">B.Tech Computer Science</span></div>
            <div class="detail-row"><span class="detail-key">Year</span><span class="detail-val">December 2025</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Datasets
        datasets = [
            ("#34d399", "UCI Student Performance",          "395 students · Regression · Final Grade G3 (0–20)"),
            ("#60a5fa", "xAPI Educational Data",             "480 students · 3-class · Low / Medium / High"),
            ("#fb923c", "UCI Dropout & Academic Success",   "4,424 students · 3-class · Dropout / Enrolled / Graduate"),
        ]
        rows_ds = "".join([
            f"""<div class="dataset-row">
                  <div class="ds-dot" style="background:{c};box-shadow:0 0 8px {c}60;"></div>
                  <div>
                    <div class="ds-name">{name}</div>
                    <div class="ds-meta">{meta}</div>
                  </div>
                </div>"""
            for c, name, meta in datasets
        ])
        st.markdown(f"""
        <div class="panel">
            <div class="panel-title"><span class="section-icon"><svg viewBox='0 0 24 24'><line x1='18' y1='20' x2='18' y2='10'/><line x1='12' y1='20' x2='12' y2='4'/><line x1='6' y1='20' x2='6' y2='14'/></svg></span>Datasets</div>
            {rows_ds}
        </div>
        """, unsafe_allow_html=True)

# ── Bottom nav ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
    <span class="nav-icon"><svg viewBox='0 0 24 24'><line x1='19' y1='12' x2='5' y2='12'/><polyline points='12 19 5 12 12 5'/></svg>Navigate using the sidebar —</span>
    <span><strong class="nav-icon"><svg viewBox='0 0 24 24'><line x1='18' y1='20' x2='18' y2='10'/><line x1='12' y1='20' x2='12' y2='4'/><line x1='6' y1='20' x2='6' y2='14'/></svg>Model Analytics</strong> to explore evaluation metrics &amp; SHAP</span>
</div>
""", unsafe_allow_html=True)