import time
import joblib
import pandas as pd
import streamlit as st

# ----------------------------
# Page Configuration & Layout
# ----------------------------
st.set_page_config(
    page_title="Box Office Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Custom CSS (Formal Cool Palette)
# ----------------------------
st.markdown(
    """
<style>
    /* Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Root Variables - Cool Tone Palette */
    :root {
        --bg-primary: #0f172a;       /* Dark slate background */
        --card-bg: #1e293b;          /* Darker blue-gray card background */
        --card-border: #334155;      /* Subtle border contrast */
        --accent-blue: #2563eb;       /* Primary enterprise blue */
        --accent-cyan: #0891b2;       /* Secondary cool highlight */
        --text-primary: #f8fafc;      /* Clean off-white */
        --text-secondary: #94a3b8;    /* Muted gray text */
        --status-success: #059669;   /* Cool emerald green */
        --status-warning: #d97706;   /* Cool amber */
        --status-danger: #dc2626;    /* Cool muted red */
    }

    /* Base App Styling */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }
    
    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Hero Section Header */
    .hero-container {
        background-color: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 0.95rem;
        max-width: 800px;
        margin: 0;
        line-height: 1.5;
    }

    /* Dashboard Metric Cards */
    .metric-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 8px;
        padding: 1.25rem;
        height: 100%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }

    .metric-container {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }

    .metric-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--text-secondary);
        font-weight: 600;
    }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: var(--text-primary);
        margin: 0.25rem 0;
    }

    .metric-sub {
        font-size: 0.8rem;
        color: var(--text-secondary);
    }

    /* Indicator Status Borders */
    .border-neutral { border-left: 4px solid var(--accent-blue); }
    .border-success { border-left: 4px solid var(--status-success); }
    .border-danger { border-left: 4px solid var(--status-danger); }
    .border-warning { border-left: 4px solid var(--status-warning); }

    /* Custom Input Controls Styling */
    [data-testid="stSidebar"] {
        background-color: #0b1120;
        border-right: 1px solid var(--card-border);
    }

    .sidebar-header {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--card-border);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Button Styling */
    .stButton > button {
        width: 100%;
        background-color: var(--accent-blue);
        color: white;
        border: none;
        padding: 0.65rem 1.25rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: background-color 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--card-border);
        border-radius: 3px;
    }

    /* Component Overrides */
    .stAlert {
        border-radius: 6px;
    }

    .custom-hr {
        height: 1px;
        background: var(--card-border);
        margin: 1.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Data & Model Caching
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("Model/best_catboost_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("data/Input_data.csv")


try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(
        f"System Error: Unable to load dataset or prediction model. Details: {e}"
    )
    st.stop()


# ----------------------------
# Header Section
# ----------------------------
st.markdown(
    """
    <div class="hero-container">
        <h1 class="hero-title">Box Office Prediction Model</h1>
        <p class="hero-subtitle">
            An enterprise machine learning model designed to estimate commercial box office performance, compute projected yield, and assess overall investment risk for film releases.
        </p>
    </div>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Sidebar Inputs
# ----------------------------
with st.sidebar:
    st.markdown(
        '<div class="sidebar-header">Model Parameters</div>',
        unsafe_allow_html=True,
    )

    vote_count = st.number_input(
        "Interesed in BMS",
        min_value=0,
        max_value=int(df["vote_count"].max()),
        value=1000,
        step=100,
        help="Total number of user votes received."
    )

    st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
    st.caption("Production & Talent")

    budget = st.number_input(
        "Budget (INR Crores)",
        min_value=1,
        value=100,
        step=5,
        help="Total budget allocated for production and marketing.",
    )

    language = st.selectbox(
        "Original Language",
        [
            "Hindi",
            "Tamil",
            "Telugu",
            "Malayalam",
            "Kannada",
            "Bengali",
            "English",
            "Marathi",
            "Punjabi",
            "Gujarati",
            "Oriya",
            "Tibetan",
            "Assamese",
            "Urdu",
        ],
    )

    genres_list = sorted(df["genres"].dropna().unique().tolist())
    genre = st.selectbox("Primary Genre", genres_list)

    directors_list = sorted(df["directors"].dropna().unique().tolist())
    director = st.selectbox("Director", directors_list)

    cast_list = sorted(df["cast"].dropna().unique().tolist())
    cast = st.selectbox("Lead Actor", cast_list)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Calculate Prediction")


# ----------------------------
# Main Dashboard Output
# ----------------------------
if predict_btn:
    with st.spinner("Processing inference pipeline..."):
        time.sleep(0.3)

        # Prepare feature payload
        input_df = pd.DataFrame(
            {
                "vote_count": [vote_count],
                "original_language": [language],
                # "popularity": [popularity],
                "genres": [genre],
                "cast": [cast],
                "budget": [budget],
                "directors": [director],
            }
        )

        prediction = float(model.predict(input_df)[0])
        profit = prediction - budget
        roi = (prediction / budget) * 100

        # Risk Classification
        if roi >= 200:
            risk_label = "Low Risk (High Return)"
            risk_border = "border-success"
        elif roi >= 100:
            risk_label = "Moderate Risk (Profitable)"
            risk_border = "border-neutral"
        elif roi >= 50:
            risk_label = "Elevated Risk (Underperforming)"
            risk_border = "border-warning"
        else:
            risk_label = "High Risk (Deficit)"
            risk_border = "border-danger"

    st.markdown("### Projected Analysis")
    st.markdown("<br>", unsafe_allow_html=True)

    # Metric Cards Layout
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card border-neutral">
                <div class="metric-container">
                    <div class="metric-header">
                        <span class="metric-label">Estimated Gross</span>
                    </div>
                    <div class="metric-value">₹ {prediction:.2f} Cr</div>
                    <div class="metric-sub">Gross Revenue Projection</div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with c2:
        is_profit = profit > 0
        p_border = "border-success" if is_profit else "border-danger"
        p_text = f"₹ {profit:.2f} Cr" if is_profit else f"- ₹ {abs(profit):.2f} Cr"
        p_sub = "Projected Net Profit" if is_profit else "Projected Net Deficit"

        st.markdown(
            f"""
            <div class="metric-card {p_border}">
                <div class="metric-container">
                    <div class="metric-header">
                        <span class="metric-label">Net Profit / Loss</span>
                    </div>
                    <div class="metric-value">{p_text}</div>
                    <div class="metric-sub">{p_sub}</div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card border-neutral">
                <div class="metric-container">
                    <div class="metric-header">
                        <span class="metric-label">Return on Investment</span>
                    </div>
                    <div class="metric-value">{roi:.1f}%</div>
                    <div class="metric-sub">Capital Return Multiplier</div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card {risk_border}">
                <div class="metric-container">
                    <div class="metric-header">
                        <span class="metric-label">Risk Category</span>
                    </div>
                    <div class="metric-value" style="font-size: 1.1rem; line-height: 2rem;">{risk_label}</div>
                    <div class="metric-sub">Commercial Risk Profile</div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)

else:
    # Default State
    st.markdown(
        """
        <div class="metric-card" style="text-align: center; padding: 3rem 1.5rem;">
            <h3 style="color: var(--text-primary); font-weight: 600; margin-bottom: 0.5rem;">Awaiting Inputs</h3>
            <p style="color: var(--text-secondary); max-width: 500px; margin: 0 auto;">
                Select the parameters in the sidebar panel and execute <strong>"Calculate Prediction"</strong> to view revenue forecasts and financial risk metrics.
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)

# ----------------------------
# Documentation Section
# ----------------------------
tab1, tab2 = st.tabs(["Feature Methodology", "Model Disclaimer"])

with tab1:
    st.info(
        """
    #### Feature Variables Used in Model Evaluation:
    * **Interested count from BookMyShow:** Quantifies early audience interest and historical engagement levels.
    * **Budget (INR Crores):** Financial expenditure dedicated to production and marketing.
    * **Original Language:** Identifies target market demographic and regional distribution footprint.
    * **Genre, Director & Lead Actor:** Tracks historical commercial performance patterns corresponding to talent combinations.
    """
    )

with tab2:
    st.warning(
        """
    #### Operational Limitations:
    * **Statistical Scope:** Projections are output by a XGBoost regression model trained on historical theatrical release data.
    * **Unaccounted Variables:** Projections do not factor in external disruptions, sudden critical review shifts, competitive release scheduling, or distribution channel constraints.
    * **Purpose:** This utility is provided strictly for exploratory analysis and decision-support modeling.
    """
    )

# Footer
st.markdown(
    """
    <div style="text-align: center; color: var(--text-secondary); font-size: 0.8rem; padding: 2rem 0 1rem 0;">
        Made By : Jal Shah
    </div>
""",
    unsafe_allow_html=True,
)