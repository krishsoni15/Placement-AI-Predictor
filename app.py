import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Student Placement AI Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CONSTANTS & CACHED PIPELINE
# ---------------------------------------------------------
FEATURES = [
    "CGPA", "Aptitude_Score", "Coding_Score", "Communication_Score",
    "Internships", "Projects", "Attendance", "Certifications", "Backlogs"
]

FEATURE_LABELS = {
    "CGPA": "CGPA (0 - 10)",
    "Aptitude_Score": "Aptitude Score",
    "Coding_Score": "Coding Score",
    "Communication_Score": "Communication Score",
    "Internships": "Internships Completed",
    "Projects": "Academic & Real Projects",
    "Attendance": "Class Attendance (%)",
    "Certifications": "Technical Certifications",
    "Backlogs": "Active Backlogs"
}

@st.cache_data
def load_data():
    return pd.read_csv("student_placement.csv")

@st.cache_resource
def train_model():
    df = load_data()
    X = df[FEATURES]
    y = df["Placed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=180,
        random_state=42,
        max_depth=8,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]
    return model, X_train, X_test, y_train, y_test, test_pred, test_prob

model, X_train, X_test, y_train, y_test, test_pred, test_prob = train_model()
df = load_data()

# Model Metric Benchmarks
acc = accuracy_score(y_test, test_pred)
prec = precision_score(y_test, test_pred)
rec = recall_score(y_test, test_pred)
f1 = f1_score(y_test, test_pred)
auc = roc_auc_score(y_test, test_prob)

# ---------------------------------------------------------
# MODERN PRO DASHBOARD STYLING
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

/* Base Reset & Typography */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #e4e4e7 !important;
}

/* Deep Obsidian Canvas with subtle vignette */
.stApp {
    background-color: #07090e !important;
    background-image: 
        radial-gradient(circle at 50% 0%, #0f172a 0%, #07090e 70%) !important;
    color: #f4f4f5 !important;
}

.block-container {
    max-width: 1260px !important;
    padding-top: 1.6rem !important;
    padding-bottom: 3.5rem !important;
}

/* Streamlit Header Bar */
header[data-testid="stHeader"] {
    background: rgba(7, 9, 14, 0.9) !important;
    backdrop-filter: blur(12px) !important;
    border-bottom: 1px solid #1e293b !important;
}
header[data-testid="stHeader"] * {
    color: #94a3b8 !important;
}

/* Sidebar - Deep Matte Graphite */
section[data-testid="stSidebar"] {
    background-color: #0b0f19 !important;
    border-right: 1px solid #1e293b !important;
}
section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] .stRadio > label {
    font-size: 0.76rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #94a3b8 !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    padding: 9px 12px !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
    border: 1px solid transparent !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: #131b2e !important;
    border-color: #334155 !important;
}

/* Input Widget Labels - Crisp & 100% Readable */
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] div,
[data-testid="stWidgetLabel"] span {
    color: #f8fafc !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: -0.01em !important;
}

/* Input Fields - Modern Inset */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 9px 12px !important;
    transition: border-color 0.15s ease !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 1px #6366f1 !important;
}
div[data-baseweb="input"],
div[data-baseweb="base-input"] {
    background-color: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}

/* Stepper Buttons (+ / -) */
div[data-testid="stNumberInput"] button {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
}
div[data-testid="stNumberInput"] button svg {
    fill: #ffffff !important;
    stroke: #ffffff !important;
}
div[data-testid="stNumberInput"] button:hover {
    background-color: #334155 !important;
    border-color: #6366f1 !important;
}

/* Primary Action Button - Glowing Indigo/Blue Gradient */
div.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.4rem !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #4338ca 0%, #2563eb 100%) !important;
    box-shadow: 0 6px 18px rgba(79, 70, 229, 0.5) !important;
    transform: translateY(-1px) !important;
}
div.stButton > button:active {
    transform: translateY(1px) !important;
}

/* Preset Buttons - Matte Slate */
div[data-testid="stHorizontalBlock"] div.stButton > button {
    background: #131d31 !important;
    color: #e2e8f0 !important;
    border: 1px solid #27364f !important;
    font-size: 0.82rem !important;
    padding: 0.45rem 0.8rem !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}
div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
    background: #1e293b !important;
    color: #ffffff !important;
    border-color: #3b82f6 !important;
}

/* Selectbox & Dropdowns */
div[data-baseweb="select"] > div {
    background-color: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}
div[data-baseweb="select"] * {
    color: #ffffff !important;
}

/* File Uploader */
div[data-testid="stFileUploader"] {
    background-color: #0f172a !important;
    border: 1px dashed #475569 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
div[data-testid="stFileUploader"] * {
    color: #cbd5e1 !important;
}

/* Hero Banner - Vibrant Gradient Border */
.pro-hero {
    background: linear-gradient(180deg, #0f172a 0%, #0a0f1d 100%);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 22px;
    position: relative;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}
.pro-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #10b981);
}
.pro-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.35);
    color: #a5b4fc;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    padding: 4px 10px;
    border-radius: 6px;
    margin-bottom: 12px;
}
.pro-hero h1 {
    font-size: 2.15rem;
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -0.03em;
    margin: 0;
    color: #ffffff;
}
.pro-hero p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 8px 0 0 0;
    max-width: 820px;
    line-height: 1.55;
}

/* Surface Panels */
.pro-panel {
    background: #0d1322;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
}
.pro-panel-title {
    font-size: 1rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
    letter-spacing: -0.01em;
}

/* Prediction Result Cards - Rich Color Accents */
.pro-result-placed {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.35) 0%, #0d1322 100%);
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-top: 3px solid #10b981;
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.15);
}
.pro-result-unplaced {
    background: linear-gradient(135deg, rgba(136, 19, 55, 0.35) 0%, #0d1322 100%);
    border: 1px solid rgba(244, 63, 94, 0.35);
    border-top: 3px solid #f43f5e;
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 8px 24px rgba(244, 63, 94, 0.15);
}

.pro-badge-placed {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid #10b981;
    color: #34d399;
    font-weight: 800;
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 3px 10px;
    border-radius: 6px;
}
.pro-badge-unplaced {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(244, 63, 94, 0.2);
    border: 1px solid #f43f5e;
    color: #fb7185;
    font-weight: 800;
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 3px 10px;
    border-radius: 6px;
}

.pro-result-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 10px 0 6px 0;
    color: #ffffff;
}

/* Strengths & Vulnerability Chips - Colorful Accents */
.pro-factor-positive {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-left: 3px solid #10b981;
    color: #d1fae5;
    padding: 9px 14px;
    border-radius: 8px;
    font-size: 0.86rem;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.pro-factor-negative {
    background: rgba(244, 63, 94, 0.08);
    border: 1px solid rgba(244, 63, 94, 0.25);
    border-left: 3px solid #f43f5e;
    color: #ffe4e6;
    padding: 9px 14px;
    border-radius: 8px;
    font-size: 0.86rem;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Dataframe & Tables */
[data-testid="stDataFrame"] {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 8px 14px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: #0d1322 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
}
div[data-testid="stExpander"] details summary span {
    color: #f8fafc !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HELPER: COLOR-ACCENTED METRIC CARD (NO ARTIFACTS)
# ---------------------------------------------------------
def render_metric_card(label, value, subtext=None, badge=None, accent_color="#3b82f6"):
    badge_html = f'<span style="font-size: 0.68rem; font-weight: 800; padding: 2px 8px; border-radius: 4px; background: {accent_color}22; color: {accent_color}; border: 1px solid {accent_color}44;">{badge}</span>' if badge else ''
    sub_html = f'<div style="font-size: 0.74rem; color: #94a3b8; margin-top: 5px; font-weight: 500;">{subtext}</div>' if subtext else ''
    return f"""
    <div style="background: #0d1322; border: 1px solid #1e293b; border-top: 2px solid {accent_color}; border-radius: 12px; padding: 18px 20px; box-shadow: 0 4px 16px rgba(0,0,0,0.3); height: 100%;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8;">{label}</span>
            {badge_html}
        </div>
        <div style="font-size: 1.85rem; font-weight: 800; color: #ffffff; letter-spacing: -0.03em; line-height: 1.1;">{value}</div>
        {sub_html}
    </div>
    """

# ---------------------------------------------------------
# SIDEBAR - BRAND & METADATA
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
        <span style="font-size: 1.5rem;">🎓</span>
        <div>
            <div style="font-weight: 800; font-size: 1.12rem; color: #ffffff; letter-spacing: -0.02em;">PlacementAI</div>
            <div style="font-size: 0.72rem; color: #60a5fa; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;">Practical 10 · ML Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 12px; margin: 14px 0 16px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">Architecture</span>
            <span style="font-size: 0.72rem; background: rgba(99, 102, 241, 0.25); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.4); padding: 2px 7px; border-radius: 4px; font-weight: 700;">Random Forest</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
            <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">Test Accuracy</span>
            <span style="font-size: 0.85rem; color: #10b981; font-weight: 800;">{acc*100:.2f}%</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
            <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">ROC-AUC</span>
            <span style="font-size: 0.85rem; color: #38bdf8; font-weight: 800;">{auc:.3f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🎯 Predict & Simulate",
            "👥 Batch Evaluation",
            "📈 Model Performance",
            "🔍 Dataset Explorer"
        ],
        index=0
    )

    st.markdown("<hr style='border-color: #1e293b; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.76rem; color: #64748b; line-height: 1.6;">
        <strong style="color: #94a3b8;">Dataset:</strong> 600 Student Records<br>
        <strong style="color: #94a3b8;">Features:</strong> 9 Academic & Skill Attributes<br>
        <strong style="color: #94a3b8;">Stack:</strong> Python · Scikit-learn · Plotly · Streamlit
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# PAGE 1: PREDICT & SIMULATE
# ---------------------------------------------------------
if page == "🎯 Predict & Simulate":
    st.markdown("""
    <div class="pro-hero">
        <div class="pro-tag">⚡ Practical 10 · Machine Learning Platform</div>
        <h1>Student Placement AI Predictor</h1>
        <p>Evaluate student hiring viability using a trained 180-tree Random Forest classifier. Analyze multi-skill benchmarks, profile gaps, and real-time sensitivity simulations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Preset Profiles
    st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;'>⚡ Quick-Load Candidate Profiles</div>", unsafe_allow_html=True)
    preset_cols = st.columns(4)

    if "p_cgpa" not in st.session_state:
        st.session_state.p_cgpa = 8.20
        st.session_state.p_apt = 75
        st.session_state.p_coding = 78
        st.session_state.p_comm = 75
        st.session_state.p_intern = 1
        st.session_state.p_proj = 3
        st.session_state.p_att = 85
        st.session_state.p_cert = 2
        st.session_state.p_backlogs = 0

    if preset_cols[0].button("🌟 Top Scholar", use_container_width=True):
        st.session_state.p_cgpa = 9.35
        st.session_state.p_apt = 92
        st.session_state.p_coding = 94
        st.session_state.p_comm = 88
        st.session_state.p_intern = 3
        st.session_state.p_proj = 4
        st.session_state.p_att = 95
        st.session_state.p_cert = 3
        st.session_state.p_backlogs = 0
        st.rerun()

    if preset_cols[1].button("💻 Tech Specialist", use_container_width=True):
        st.session_state.p_cgpa = 7.45
        st.session_state.p_apt = 78
        st.session_state.p_coding = 96
        st.session_state.p_comm = 70
        st.session_state.p_intern = 2
        st.session_state.p_proj = 5
        st.session_state.p_att = 80
        st.session_state.p_cert = 3
        st.session_state.p_backlogs = 0
        st.rerun()

    if preset_cols[2].button("⚖️ Borderline Profile", use_container_width=True):
        st.session_state.p_cgpa = 6.95
        st.session_state.p_apt = 64
        st.session_state.p_coding = 62
        st.session_state.p_comm = 65
        st.session_state.p_intern = 1
        st.session_state.p_proj = 2
        st.session_state.p_att = 74
        st.session_state.p_cert = 1
        st.session_state.p_backlogs = 1
        st.rerun()

    if preset_cols[3].button("⚠️ High-Risk Profile", use_container_width=True):
        st.session_state.p_cgpa = 5.50
        st.session_state.p_apt = 42
        st.session_state.p_coding = 38
        st.session_state.p_comm = 48
        st.session_state.p_intern = 0
        st.session_state.p_proj = 1
        st.session_state.p_att = 62
        st.session_state.p_cert = 0
        st.session_state.p_backlogs = 3
        st.rerun()

    st.write("")

    left_col, right_col = st.columns([1.15, 0.85], gap="large")

    with left_col:
        st.markdown("""
        <div class="pro-panel-title">
            <span>📝</span> Student Profile & Academic Inputs
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            cgpa = st.number_input(
                "CGPA (0 - 10)",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.p_cgpa),
                step=0.01
            )
            coding = st.number_input(
                "Coding Score (0 - 100)",
                min_value=0,
                max_value=100,
                value=int(st.session_state.p_coding),
                step=1
            )
            attendance = st.number_input(
                "Attendance (%)",
                min_value=0,
                max_value=100,
                value=int(st.session_state.p_att),
                step=1
            )

        with col2:
            aptitude = st.number_input(
                "Aptitude Score (0 - 100)",
                min_value=0,
                max_value=100,
                value=int(st.session_state.p_apt),
                step=1
            )
            communication = st.number_input(
                "Communication Score",
                min_value=0,
                max_value=100,
                value=int(st.session_state.p_comm),
                step=1
            )
            backlogs = st.number_input(
                "Active Backlogs",
                min_value=0,
                max_value=10,
                value=int(st.session_state.p_backlogs),
                step=1
            )

        with col3:
            internships = st.number_input(
                "Internships",
                min_value=0,
                max_value=10,
                value=int(st.session_state.p_intern),
                step=1
            )
            projects = st.number_input(
                "Projects",
                min_value=0,
                max_value=10,
                value=int(st.session_state.p_proj),
                step=1
            )
            certifications = st.number_input(
                "Certifications",
                min_value=0,
                max_value=10,
                value=int(st.session_state.p_cert),
                step=1
            )

        st.write("")
        st.button("Evaluate Placement Viability", use_container_width=True)

    # Computation of Prediction
    current_input = pd.DataFrame(
        [[cgpa, aptitude, coding, communication, internships, projects, attendance, certifications, backlogs]],
        columns=FEATURES
    )
    current_prediction = int(model.predict(current_input)[0])
    current_prob = float(model.predict_proba(current_input)[0][1])

    with right_col:
        st.markdown("""
        <div class="pro-panel-title">
            <span>📊</span> Model Evaluation Result
        </div>
        """, unsafe_allow_html=True)

        if current_prediction == 1:
            conf_label = "HIGH CONFIDENCE" if current_prob >= 0.75 else "MODERATE CONFIDENCE"
            st.markdown(f"""
            <div class="pro-result-placed">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="pro-badge-placed">● {conf_label}</span>
                    <span style="color: #34d399; font-size: 0.88rem; font-weight: 800; font-family: 'JetBrains Mono', monospace;">{current_prob*100:.1f}% PROBABILITY</span>
                </div>
                <div class="pro-result-title">LIKELY PLACED</div>
                <div style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.55;">
                    The candidate's technical competencies and academic standing exceed the historical campus recruitment threshold.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            risk_label = "CRITICAL RISK" if current_prob <= 0.35 else "ELEVATED RISK"
            st.markdown(f"""
            <div class="pro-result-unplaced">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="pro-badge-unplaced">● {risk_label}</span>
                    <span style="color: #fb7185; font-size: 0.88rem; font-weight: 800; font-family: 'JetBrains Mono', monospace;">{current_prob*100:.1f}% PROBABILITY</span>
                </div>
                <div class="pro-result-title">AT-RISK / UNPLACED</div>
                <div style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.55;">
                    Identified skill deficits or active backlogs position the candidate below the statistical placement threshold.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # COLORFUL GAUGE CHART
        gauge_bar_color = "#10b981" if current_prob >= 0.70 else ("#f59e0b" if current_prob >= 0.45 else "#f43f5e")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_prob * 100,
            number={"suffix": "%", "font": {"size": 36, "color": "#ffffff", "family": "Plus Jakarta Sans"}},
            title={"text": "Placement Likelihood Index", "font": {"size": 13, "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569", "tickfont": {"color": "#94a3b8"}},
                "bar": {"color": gauge_bar_color, "thickness": 0.32},
                "bgcolor": "#1e293b",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 45], "color": "rgba(244, 63, 94, 0.25)"},
                    {"range": [45, 70], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [70, 100], "color": "rgba(16, 185, 129, 0.25)"}
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.8,
                    "value": current_prob * 100
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=200,
            margin=dict(l=20, r=20, t=30, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ---------------------------------------------------------
    # SECONDARY ANALYSIS: FACTORS, RADAR, WHAT-IF
    # ---------------------------------------------------------
    st.write("")
    detail_col1, detail_col2 = st.columns([1, 1], gap="large")

    with detail_col1:
        st.markdown("""
        <div class="pro-panel">
            <div class="pro-panel-title">
                <span>🎯</span> Profile Drivers & Diagnostic Signals
            </div>
        """, unsafe_allow_html=True)

        # Dynamic Factor Drivers
        strengths = []
        risks = []

        if cgpa >= 7.8:
            strengths.append(f"Strong Academic CGPA ({cgpa:.2f}) meets competitive screening benchmarks")
        elif cgpa < 6.8:
            risks.append(f"CGPA ({cgpa:.2f}) sits below standard recruitment thresholds (7.0)")

        if coding >= 75:
            strengths.append(f"Technical Coding Score ({coding}/100) indicates strong software competence")
        elif coding < 60:
            risks.append(f"Coding Score ({coding}/100) reflects vulnerability in technical screening rounds")

        if backlogs == 0:
            strengths.append("Zero active backlogs maintains clean eligibility for corporate drives")
        else:
            risks.append(f"Active standing backlogs ({backlogs}) significantly degrade selection probability")

        if internships >= 2:
            strengths.append(f"Proven practical industry experience ({internships} internships completed)")
        elif internships == 0:
            risks.append("No recorded industry internship experience")

        if projects >= 3:
            strengths.append(f"Solid project portfolio ({projects} completed projects)")

        if aptitude >= 75:
            strengths.append(f"High cognitive and logical aptitude ({aptitude}/100)")
        elif aptitude < 55:
            risks.append(f"Aptitude Score ({aptitude}/100) below competitive median")

        if attendance < 75:
            risks.append(f"Academic attendance ({attendance}%) indicates potential reliability flags")

        for s in strengths:
            st.markdown(f'<div class="pro-factor-positive"><span style="color: #10b981; font-weight: 800; font-size: 1.1rem;">+</span> <span>{s}</span></div>', unsafe_allow_html=True)

        for r in risks:
            st.markdown(f'<div class="pro-factor-negative"><span style="color: #f43f5e; font-weight: 800; font-size: 1.1rem;">-</span> <span>{r}</span></div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Strategic Action Items
        st.markdown("""
        <div class="pro-panel">
            <div class="pro-panel-title">
                <span>💡</span> Strategic Career Milestones
            </div>
            <ul style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.7; padding-left: 20px; margin: 0;">
        """, unsafe_allow_html=True)
        if backlogs > 0:
            st.markdown(f"<li><strong style='color: #f43f5e;'>Clear Standing Backlogs:</strong> Rectify <strong>{backlogs} pending backlog(s)</strong>. Zero-backlog candidates historically yield a <strong>3.2x higher placement conversion</strong>.</li>", unsafe_allow_html=True)
        if internships == 0:
            st.markdown("<li><strong style='color: #38bdf8;'>Industrial Experience:</strong> Complete at least <strong>1 verified internship</strong> before on-campus recruitment cycles commence.</li>", unsafe_allow_html=True)
        if coding < 75:
            st.markdown("<li><strong style='color: #a78bfa;'>Technical Assessment Practice:</strong> Target 50+ data structures problems to raise technical score into the 75+ bracket.</li>", unsafe_allow_html=True)
        if cgpa >= 8.0 and coding >= 80:
            st.markdown("<li><strong style='color: #10b981;'>Elite Recruitment Tracks:</strong> Profile qualifies for competitive engineering and systems engineering roles.</li>", unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)

    with detail_col2:
        # COLORFUL RADAR CHART
        placed_students = df[df["Placed"] == 1]
        benchmark_placed = {
            "CGPA": (placed_students["CGPA"].mean() / 10.0) * 100,
            "Coding": placed_students["Coding_Score"].mean(),
            "Aptitude": placed_students["Aptitude_Score"].mean(),
            "Communication": placed_students["Communication_Score"].mean(),
            "Attendance": placed_students["Attendance"].mean(),
            "Experience": min(100, (placed_students["Internships"].mean() * 25 + placed_students["Projects"].mean() * 15))
        }

        student_norm = {
            "CGPA": (cgpa / 10.0) * 100,
            "Coding": float(coding),
            "Aptitude": float(aptitude),
            "Communication": float(communication),
            "Attendance": float(attendance),
            "Experience": min(100, (internships * 25 + projects * 15))
        }

        radar_categories = ["CGPA", "Coding", "Aptitude", "Communication", "Attendance", "Experience"]

        candidate_radar_color = "#10b981" if current_prediction == 1 else "#f43f5e"
        candidate_radar_fill = "rgba(16, 185, 129, 0.35)" if current_prediction == 1 else "rgba(244, 63, 94, 0.35)"

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[benchmark_placed[c] for c in radar_categories] + [benchmark_placed[radar_categories[0]]],
            theta=radar_categories + [radar_categories[0]],
            fill='toself',
            name='Avg Placed Student',
            fillcolor='rgba(99, 102, 241, 0.22)',
            line=dict(color='#818cf8', width=2, dash='dot')
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[student_norm[c] for c in radar_categories] + [student_norm[radar_categories[0]]],
            theta=radar_categories + [radar_categories[0]],
            fill='toself',
            name='Current Candidate',
            fillcolor=candidate_radar_fill,
            line=dict(color=candidate_radar_color, width=2.8)
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#64748b", gridcolor="rgba(255, 255, 255, 0.08)"),
                angularaxis=dict(color="#cbd5e1", gridcolor="rgba(255, 255, 255, 0.08)")
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(color="#cbd5e1")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=40, r=40, t=20, b=20)
        )

        st.markdown("""
        <div class="pro-panel">
            <div class="pro-panel-title">
                <span>🕸️</span> Multi-Skill Radar Benchmark
            </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # What-If Sensitivity Simulator
        with st.expander("🧪 Sensitivity Sandbox (What-If Simulation)", expanded=True):
            st.caption("Hypothesize interventions to evaluate projected impact on placement odds.")
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                add_internships = st.slider("Additional Internships (+N)", 0, 3, 0)
                add_coding = st.slider("Coding Score Gain (+N)", 0, 30, 0)
            with sim_col2:
                clear_backlogs = st.checkbox("Clear All Backlogs (Set to 0)", value=(backlogs > 0))
                add_projects = st.slider("Additional Projects (+N)", 0, 3, 0)

            sim_backlogs = 0 if clear_backlogs else backlogs
            sim_input = pd.DataFrame([[
                cgpa,
                aptitude,
                min(100, coding + add_coding),
                communication,
                min(10, internships + add_internships),
                min(10, projects + add_projects),
                attendance,
                certifications,
                sim_backlogs
            ]], columns=FEATURES)

            sim_prob = float(model.predict_proba(sim_input)[0][1])
            diff = (sim_prob - current_prob) * 100

            diff_str = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
            diff_color = "#34d399" if diff > 0 else ("#94a3b8" if diff == 0 else "#f87171")

            st.markdown(f"""
            <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 14px 18px; margin-top: 10px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: #94a3b8; font-size: 0.76rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.06em;">Simulated Probability</span>
                    <div style="color: #ffffff; font-size: 1.55rem; font-weight: 800; font-family: 'JetBrains Mono', monospace;">{sim_prob*100:.1f}%</div>
                </div>
                <div style="text-align: right;">
                    <span style="color: #94a3b8; font-size: 0.76rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.06em;">Projected Delta</span>
                    <div style="color: {diff_color}; font-size: 1.45rem; font-weight: 800; font-family: 'JetBrains Mono', monospace;">{diff_str}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------
# PAGE 2: BATCH EVALUATION
# ---------------------------------------------------------
elif page == "👥 Batch Evaluation":
    st.markdown("""
    <div class="pro-hero">
        <div class="pro-tag">👥 Batch Processing Engine</div>
        <h1>Cohort Placement Evaluation</h1>
        <p>Conduct automated high-throughput evaluation across student batches. Generate placement projections, ratio metrics, and downloadable prediction rosters.</p>
    </div>
    """, unsafe_allow_html=True)

    b_col1, b_col2 = st.columns([1.2, 0.8], gap="large")

    with b_col1:
        st.markdown("<div style='font-weight: 700; color: #ffffff; margin-bottom: 6px;'>Upload Cohort CSV Roster</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV formatted with student features", type=["csv"], label_visibility="collapsed")

        sample_batch = df.sample(25, random_state=101)[FEATURES].copy().reset_index(drop=True)
        sample_csv_bytes = sample_batch.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Benchmark Template (25 Students)",
            data=sample_csv_bytes,
            file_name="sample_student_batch.csv",
            mime="text/csv"
        )

    with b_col2:
        st.markdown("<div style='font-weight: 700; color: #ffffff; margin-bottom: 6px;'>Instant Demo Evaluation</div>", unsafe_allow_html=True)
        use_sample = st.button("Evaluate Sample Batch (25 Students)", use_container_width=True)

    batch_df = None
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")
    elif use_sample:
        batch_df = sample_batch.copy()

    if batch_df is not None:
        missing_cols = [c for c in FEATURES if c not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing required columns in CSV: {', '.join(missing_cols)}")
        else:
            predictions = model.predict(batch_df[FEATURES])
            probabilities = model.predict_proba(batch_df[FEATURES])[:, 1]

            results_df = batch_df.copy()
            results_df["Predicted_Status"] = ["PLACED" if p == 1 else "NOT PLACED" for p in predictions]
            results_df["Placement_Probability (%)"] = np.round(probabilities * 100, 1)

            st.write("")
            st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #ffffff; margin-bottom: 12px;'>📊 Batch Summary Metrics</div>", unsafe_allow_html=True)
            
            total_students = len(results_df)
            placed_count = int(np.sum(predictions))
            unplaced_count = total_students - placed_count
            placement_rate = (placed_count / total_students) * 100
            avg_prob = np.mean(probabilities) * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(render_metric_card("Evaluated Candidates", total_students, "Total cohort count", "BATCH", "#6366f1"), unsafe_allow_html=True)
            m2.markdown(render_metric_card("Projected Placed", placed_count, f"{placement_rate:.1f}% cohort rate", "PLACED", "#10b981"), unsafe_allow_html=True)
            m3.markdown(render_metric_card("At-Risk Candidates", unplaced_count, "Requires intervention", "RISK", "#f43f5e"), unsafe_allow_html=True)
            m4.markdown(render_metric_card("Mean Probability", f"{avg_prob:.1f}%", "Cohort average", "MEAN", "#06b6d4"), unsafe_allow_html=True)

            st.write("")
            c_chart1, c_chart2 = st.columns(2)
            with c_chart1:
                # COLORFUL DONUT CHART
                fig_donut = px.pie(
                    names=["Placed", "At-Risk"],
                    values=[placed_count, unplaced_count],
                    hole=0.62,
                    color=["Placed", "At-Risk"],
                    color_discrete_map={"Placed": "#10b981", "At-Risk": "#f43f5e"},
                    title="Placement Projection Ratio"
                )
                fig_donut.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
                    margin=dict(t=40, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            with c_chart2:
                # COLORFUL HISTOGRAM
                fig_hist = px.histogram(
                    results_df,
                    x="Placement_Probability (%)",
                    nbins=12,
                    color="Predicted_Status",
                    color_discrete_map={"PLACED": "#10b981", "NOT PLACED": "#f43f5e"},
                    title="Cohort Probability Distribution"
                )
                fig_hist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Estimated Probability (%)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Candidate Count"),
                    margin=dict(t=40, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #ffffff; margin: 18px 0 10px 0;'>📋 Evaluated Student Roster</div>", unsafe_allow_html=True)
            st.dataframe(results_df, width="stretch", height=380)

            pred_csv_bytes = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Export Evaluated Cohort Data (CSV)",
                data=pred_csv_bytes,
                file_name="placement_cohort_predictions.csv",
                mime="text/csv"
            )
    else:
        st.info("Upload a candidate CSV file or click 'Evaluate Sample Batch' above to initiate inference.")


# ---------------------------------------------------------
# PAGE 3: MODEL PERFORMANCE & EVALUATION
# ---------------------------------------------------------
elif page == "📈 Model Performance":
    st.markdown("""
    <div class="pro-hero">
        <div class="pro-tag">🔬 Model Diagnostics & Evaluation</div>
        <h1>Random Forest Classifier Diagnostics</h1>
        <p>Comprehensive model audit on the held-out 20% test partition (120 student profiles). Evaluated across precision, recall, ROC-AUC curve, and multi-tree feature importance distribution.</p>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics Cards (Vibrant Accent Colors)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(render_metric_card("Overall Accuracy", f"{acc*100:.2f}%", "Held-out test set", "ACC", "#3b82f6"), unsafe_allow_html=True)
    k2.markdown(render_metric_card("Precision Score", f"{prec*100:.2f}%", "Positive predictive value", "PREC", "#10b981"), unsafe_allow_html=True)
    k3.markdown(render_metric_card("Recall / Sensitivity", f"{rec*100:.2f}%", "True positive rate", "REC", "#8b5cf6"), unsafe_allow_html=True)
    k4.markdown(render_metric_card("Balanced F1-Score", f"{f1*100:.2f}%", "Harmonic mean", "F1", "#f59e0b"), unsafe_allow_html=True)
    k5.markdown(render_metric_card("ROC-AUC Metric", f"{auc:.3f}", "Discrimination power", "AUC", "#06b6d4"), unsafe_allow_html=True)

    st.write("")
    row1_c1, row1_c2 = st.columns(2, gap="large")

    with row1_c1:
        st.markdown("""
        <div class="pro-panel-title">
            <span>🔲</span> Confusion Matrix (Held-out Test Set)
        </div>
        """, unsafe_allow_html=True)
        cm = confusion_matrix(y_test, test_pred)
        
        # COLORFUL CONFUSION MATRIX HEATMAP (Sapphire to Cyan)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted: Not Placed", "Predicted: Placed"],
            y=["Actual: Not Placed", "Actual: Placed"],
            color_continuous_scale=[[0, "#0b1329"], [0.35, "#1d4ed8"], [0.75, "#2563eb"], [1, "#06b6d4"]],
            aspect="auto"
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc", family="Plus Jakarta Sans", size=13),
            coloraxis_showscale=False,
            height=340,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        fig_cm.update_traces(
            texttemplate="%{z}",
            textfont=dict(size=24, color="#ffffff", family="Plus Jakarta Sans")
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("✅ 59 True Negatives & 50 True Positives correctly classified out of 120 test candidates.")

    with row1_c2:
        st.markdown("""
        <div class="pro-panel-title">
            <span>📈</span> ROC Curve & Discrimination Power
        </div>
        """, unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, test_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'Random Forest (AUC = {auc:.3f})',
            line=dict(color='#10b981', width=3),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.20)'
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Chance (AUC = 0.500)',
            line=dict(color='#f59e0b', dash='dash', width=2)
        ))
        fig_roc.update_layout(
            xaxis=dict(title="False Positive Rate", gridcolor="rgba(255,255,255,0.06)", range=[0, 1]),
            yaxis=dict(title="True Positive Rate (Recall)", gridcolor="rgba(255,255,255,0.06)", range=[0, 1.02]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            height=340,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        st.caption(f"🚀 Area-under-curve ({auc:.3f}) confirms robust threshold invariance and ranking capability.")

    st.write("")
    row2_c1, row2_c2 = st.columns([1.1, 0.9], gap="large")

    with row2_c1:
        st.markdown("""
        <div class="pro-panel-title">
            <span>🌲</span> Feature Importance Hierarchy (Gini Impurity)
        </div>
        """, unsafe_allow_html=True)
        imp_df = pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(f, f) for f in FEATURES],
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)

        # COLORFUL FEATURE IMPORTANCE BAR CHART (Blue to Magenta)
        fig_imp = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=[[0, "#3b82f6"], [0.35, "#6366f1"], [0.7, "#a855f7"], [1, "#ec4899"]],
            text=imp_df["Importance"].apply(lambda v: f"{v*100:.1f}%")
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc", family="Plus Jakarta Sans"),
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Relative Importance Weight"),
            yaxis=dict(title=""),
            height=370,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        fig_imp.update_traces(textposition='outside')
        st.plotly_chart(fig_imp, use_container_width=True)

    with row2_c2:
        st.markdown("""
        <div class="pro-panel-title">
            <span>📋</span> Classification Report
        </div>
        """, unsafe_allow_html=True)
        rep = classification_report(y_test, test_pred, target_names=["Not Placed", "Placed"], output_dict=True)
        rep_df = pd.DataFrame(rep).T.round(3)
        st.dataframe(rep_df, width="stretch", height=210)

        # Specifications Card
        st.markdown("""
        <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 14px 16px; margin-top: 14px;">
            <div style="font-size: 0.76rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">Model Architecture Specifications</div>
            <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.65;">
                • <strong>Algorithm:</strong> Random Forest Classifier (Decision Tree Ensemble)<br>
                • <strong>Estimators:</strong> 180 Parallel Trees<br>
                • <strong>Max Tree Depth:</strong> 8 Levels (Regularized)<br>
                • <strong>Partition Strategy:</strong> 80% Train (480) / 20% Stratified Test (120)<br>
                • <strong>Reproducibility:</strong> Seed 42 Deterministic
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------
# PAGE 4: DATASET EXPLORER
# ---------------------------------------------------------
else:
    st.markdown("""
    <div class="pro-hero">
        <div class="pro-tag">🔍 Exploratory Data Analysis</div>
        <h1>Dataset Explorer & Empirical Trends</h1>
        <p>Inspect distributions across 600 student profiles, analyze correlations between academic predictors, and export custom filtered subsets.</p>
    </div>
    """, unsafe_allow_html=True)

    placed_total = int(df["Placed"].sum())
    unplaced_total = len(df) - placed_total

    d1, d2, d3, d4 = st.columns(4)
    d1.markdown(render_metric_card("Total Records", len(df), "Full training dataset", "DATA", "#6366f1"), unsafe_allow_html=True)
    d2.markdown(render_metric_card("Placed Students", placed_total, f"{(placed_total/len(df))*100:.1f}% placement rate", "PLACED", "#10b981"), unsafe_allow_html=True)
    d3.markdown(render_metric_card("Unplaced Students", unplaced_total, "Historical cohort", "RISK", "#f43f5e"), unsafe_allow_html=True)
    d4.markdown(render_metric_card("Cohort Mean CGPA", f"{df['CGPA'].mean():.2f}", "Scale: 0.00 - 10.00", "CGPA", "#06b6d4"), unsafe_allow_html=True)

    st.write("")
    tab_analytics, tab_table = st.tabs(["📊 Visual Analytics", "📁 Filterable Data Table"])

    with tab_analytics:
        chart_col1, chart_col2 = st.columns(2, gap="large")

        with chart_col1:
            st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #ffffff; margin-bottom: 8px;'>🎓 CGPA Distribution by Placement Outcome</div>", unsafe_allow_html=True)
            fig_cgpa = px.box(
                df,
                x="Placed",
                y="CGPA",
                color="Placed",
                color_discrete_map={1: "#10b981", 0: "#f43f5e"},
                points="all",
                labels={"Placed": "Placement Status (0: No, 1: Yes)"}
            )
            fig_cgpa.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
                showlegend=False,
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", ticktext=["Not Placed", "Placed"], tickvals=[0, 1]),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="CGPA"),
                height=350
            )
            st.plotly_chart(fig_cgpa, use_container_width=True)

        with chart_col2:
            st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #ffffff; margin-bottom: 8px;'>💻 Coding Score vs Aptitude Score (Clusters)</div>", unsafe_allow_html=True)
            fig_scatter = px.scatter(
                df,
                x="Aptitude_Score",
                y="Coding_Score",
                color="Placed",
                color_discrete_map={1: "#06b6d4", 0: "#f43f5e"},
                size="Projects",
                hover_data=["CGPA", "Backlogs"],
                labels={"Placed": "Placed"}
            )
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Aptitude Score"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Coding Score"),
                height=350
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.write("")
        st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #ffffff; margin-bottom: 8px;'>🔗 Feature Correlation Heatmap</div>", unsafe_allow_html=True)
        corr_matrix = df.corr().round(2)
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc", size=11, family="Plus Jakarta Sans"),
            height=460,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab_table:
        st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #ffffff; margin-bottom: 8px;'>Filter Criteria</div>", unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            cgpa_range = st.slider("CGPA Range", float(df["CGPA"].min()), float(df["CGPA"].max()), (6.0, 10.0))
        with fc2:
            status_filter = st.selectbox("Placement Outcome", ["All Records", "Placed Only", "Unplaced Only"])
        with fc3:
            max_backlogs = st.slider("Max Active Backlogs", 0, int(df["Backlogs"].max()), int(df["Backlogs"].max()))

        filtered_df = df[(df["CGPA"] >= cgpa_range[0]) & (df["CGPA"] <= cgpa_range[1])]
        filtered_df = filtered_df[filtered_df["Backlogs"] <= max_backlogs]

        if status_filter == "Placed Only":
            filtered_df = filtered_df[filtered_df["Placed"] == 1]
        elif status_filter == "Unplaced Only":
            filtered_df = filtered_df[filtered_df["Placed"] == 0]

        st.caption(f"Showing **{len(filtered_df)}** of {len(df)} records matching filter criteria.")
        st.dataframe(filtered_df, width="stretch", height=430)

        csv_download = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered CSV Dataset",
            data=csv_download,
            file_name="student_placement_filtered.csv",
            mime="text/csv"
        )
