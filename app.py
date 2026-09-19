"""
Streamlit app: Diabetes Risk Predictor (High-Tech Dark UI)

Run locally:
    streamlit run app.py

Requires model.pkl and scaler.pkl to already exist
(run `python train_model.py` first).
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom styling — dark, high-tech theme with high-contrast text
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 20% 0%, #111827 0%, #0a0f1a 55%, #05070c 100%);
        color: #e5e7eb;
    }

    section[data-testid="stSidebar"] {
        background: #0d1420;
        border-right: 1px solid #1f2937;
    }

    .main-header {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #22d3ee, #6366f1, #22d3ee);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.3rem;
        margin-bottom: 1.8rem;
    }

    .metric-card {
        background: linear-gradient(145deg, #111827, #0d1420);
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        text-align: center;
        box-shadow: 0 0 0 1px rgba(99,102,241,0.05), 0 8px 20px rgba(0,0,0,0.35);
    }
    .metric-card h3 {
        margin: 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-card p {
        margin: 0.3rem 0 0 0;
        font-size: 1.6rem;
        font-weight: 700;
        color: #f1f5f9;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Single-column result panel, high contrast on dark backgrounds */
    .result-panel {
        border-radius: 20px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.2rem;
        border: 1px solid;
    }
    .result-panel h2 {
        margin: 0 0 0.4rem 0;
        font-size: 1.9rem;
        font-weight: 700;
    }
    .result-panel p {
        margin: 0;
        font-size: 1.15rem;
        color: #e2e8f0;
    }
    .result-panel .prob {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
    }
    .result-low {
        background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(16,185,129,0.03));
        border-color: #10b981;
    }
    .result-low h2 { color: #34d399; }
    .result-medium {
        background: linear-gradient(135deg, rgba(245,158,11,0.14), rgba(245,158,11,0.03));
        border-color: #f59e0b;
    }
    .result-medium h2 { color: #fbbf24; }
    .result-high {
        background: linear-gradient(135deg, rgba(239,68,68,0.14), rgba(239,68,68,0.03));
        border-color: #ef4444;
    }
    .result-high h2 { color: #f87171; }

    .section-card {
        background: #0d1420;
        border: 1px solid #1f2937;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
    }
    .section-card h3 {
        margin-top: 0;
        color: #f1f5f9;
        font-size: 1.15rem;
    }
    .section-card ul {
        margin: 0.5rem 0 0 0;
        padding-left: 1.2rem;
        color: #cbd5e1;
    }
    .section-card li {
        margin-bottom: 0.45rem;
        line-height: 1.5;
    }

    .disclaimer {
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 0.8rem;
    }

    .stButton>button {
        background: linear-gradient(90deg, #6366f1, #22d3ee);
        color: #05070c;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.4rem;
        width: 100%;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.03em;
    }
    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(34,211,238,0.4);
    }

    /* Sidebar text contrast */
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #cbd5e1 !important;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 {
        color: #f1f5f9 !important;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run `python train_model.py` first.")
    st.stop()

def synced_input(label, min_val, max_val, default, step=1, help=None, key_prefix=""):
    """A slider and a number box that stay in sync with each other."""
    slider_key = f"{key_prefix}_slider"
    number_key = f"{key_prefix}_number"

    if slider_key not in st.session_state:
        st.session_state[slider_key] = default
    if number_key not in st.session_state:
        st.session_state[number_key] = default

    def _from_slider():
        st.session_state[number_key] = st.session_state[slider_key]

    def _from_number():
        st.session_state[slider_key] = st.session_state[number_key]

    st.markdown(f"**{label}**")
    col_slider, col_number = st.columns([2.2, 1])
    with col_slider:
        st.slider(
            label, min_val, max_val, step=step, key=slider_key,
            on_change=_from_slider, help=help, label_visibility="collapsed",
        )
    with col_number:
        st.number_input(
            label, min_val, max_val, step=step, key=number_key,
            on_change=_from_number, label_visibility="collapsed",
        )

    return st.session_state[slider_key]

FEATURE_LABELS = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose (mg/dL)",
    "BloodPressure": "Blood Pressure (mm Hg)",
    "Insulin": "Insulin (mu U/mL)",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Family History",
    "Age": "Age",
}

def get_risk_tier(probability):
    """Returns (tier_name, icon, css_class, recommendations) for a given probability."""
    if probability < 0.33:
        return (
            "Low Risk", "✅", "result-low",
            [
                "Keep up regular physical activity — aim for at least 150 minutes a week.",
                "Maintain a balanced diet with controlled sugar and refined-carb intake.",
                "Get a routine health checkup once a year, even without symptoms.",
                "Stay at a healthy weight for your height.",
            ],
        )
    elif probability < 0.66:
        return (
            "Moderate Risk", "⚠️", "result-medium",
            [
                "Consider getting an actual fasting blood glucose or HbA1c test soon.",
                "Cut down on sugary drinks, refined carbs, and processed snacks.",
                "Add 30 minutes of moderate exercise (walking, cycling) most days.",
                "Track your weight and blood pressure regularly.",
                "Talk to a doctor about your family history and personal risk factors.",
            ],
        )
    else:
        return (
            "High Risk", "🚨", "result-high",
            [
                "Schedule an appointment with a doctor for proper diabetes screening soon.",
                "Ask about a fasting glucose test and HbA1c test specifically.",
                "Start reducing sugar, refined carbs, and processed food intake now.",
                "Increase physical activity gradually, under medical guidance if needed.",
                "Monitor blood pressure and weight closely.",
                "Don't self-diagnose or self-medicate — this tool only flags risk, a doctor confirms it.",
            ],
        )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">🩺 DIABETES RISK PREDICTOR</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">AI-powered risk assessment from basic health metrics. '
    'Educational demo — not a medical diagnosis.</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📋 Patient Details")
    st.caption("Drag the slider or type the exact number — they stay in sync.")

    pregnancies = synced_input("Pregnancies", 0, 17, 1, key_prefix="preg")
    glucose = synced_input("Glucose level (mg/dL)", 0, 250, 110, key_prefix="gluc")
    blood_pressure = synced_input("Blood pressure (mm Hg)", 0, 150, 70, key_prefix="bp")
    insulin = synced_input("Insulin level (mu U/mL)", 0, 850, 80, key_prefix="ins")
    bmi = synced_input("BMI", 0.0, 67.0, 25.0, step=0.1, key_prefix="bmi")

    family_history = st.selectbox(
        "Family history of diabetes",
        options=[
            "No close relatives with diabetes",
            "One parent or sibling with diabetes",
            "More than one close relative with diabetes",
        ],
        index=0,
        help="Close relatives = parents or siblings. This estimates your genetic/family risk factor.",
    )
    dpf_map = {
        "No close relatives with diabetes": 0.2,
        "One parent or sibling with diabetes": 0.5,
        "More than one close relative with diabetes": 0.9,
    }
    dpf = dpf_map[family_history]

    age = synced_input("Age", 1, 100, 30, key_prefix="age")

    st.divider()
    predict_clicked = st.button("🔍 PREDICT RISK", use_container_width=True)

# ---------------------------------------------------------------------------
# Main area: summary cards of entered values
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
for col, label, value in zip(
    [c1, c2, c3, c4],
    ["Glucose", "BMI", "Blood Pressure", "Age"],
    [f"{glucose} mg/dL", f"{bmi}", f"{blood_pressure} mmHg", f"{age} yrs"],
):
    with col:
        st.markdown(
            f'<div class="metric-card"><h3>{label}</h3><p>{value}</p></div>',
            unsafe_allow_html=True,
        )

st.write("")

# ---------------------------------------------------------------------------
# Prediction — single-column layout
# ---------------------------------------------------------------------------
if predict_clicked:
    features = np.array([[
        pregnancies, glucose, blood_pressure,
        insulin, bmi, dpf, age,
    ]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    tier_name, icon, box_class, recommendations = get_risk_tier(probability)

    # --- Result panel (high contrast, single column) ---
    st.markdown(f"""
    <div class="result-panel {box_class}">
        <h2>{icon} {tier_name}</h2>
        <p>Estimated probability of diabetes: <span class="prob">{probability:.1%}</span></p>
        <p class="disclaimer">
            This is a machine learning estimate from sample-trained data, not a medical
            diagnosis. Please consult a doctor for an actual assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Gauge chart ---
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"color": "#f1f5f9", "family": "JetBrains Mono"}},
        title={"text": "RISK PROBABILITY", "font": {"color": "#94a3b8", "size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569"},
            "bar": {"color": "#22d3ee"},
            "bgcolor": "#0d1420",
            "borderwidth": 1,
            "bordercolor": "#1f2937",
            "steps": [
                {"range": [0, 33], "color": "rgba(16,185,129,0.25)"},
                {"range": [33, 66], "color": "rgba(245,158,11,0.25)"},
                {"range": [66, 100], "color": "rgba(239,68,68,0.25)"},
            ],
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(t=50, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Recommendations ---
    tips_html = "".join(f"<li>{tip}</li>" for tip in recommendations)
    st.markdown(f"""
    <div class="section-card">
        <h3>{icon} What you can do — {tier_name}</h3>
        <ul>{tips_html}</ul>
    </div>
    """, unsafe_allow_html=True)

    # --- Feature importance chart ---
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": [FEATURE_LABELS[f] for f in FEATURE_LABELS],
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    fig2 = go.Figure(go.Bar(
        x=imp_df["Importance"],
        y=imp_df["Feature"],
        orientation="h",
        marker=dict(
            color=imp_df["Importance"],
            colorscale=[[0, "#6366f1"], [1, "#22d3ee"]],
        ),
    ))
    fig2.update_layout(
        title={"text": "📊 WHAT INFLUENCED THIS PREDICTION", "font": {"color": "#94a3b8", "size": 13}},
        height=320,
        margin=dict(t=50, b=10, l=10, r=20),
        xaxis_title="Relative importance",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1"},
        xaxis=dict(gridcolor="#1f2937"),
        yaxis=dict(gridcolor="#1f2937"),
    )
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.markdown("""
    <div class="section-card">
        <p style="color:#94a3b8; margin:0;">
            👈 Enter patient details in the sidebar and click <b style="color:#f1f5f9;">PREDICT RISK</b> to see results.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("Built with scikit-learn + Streamlit + Plotly · Model: Random Forest Classifier")
