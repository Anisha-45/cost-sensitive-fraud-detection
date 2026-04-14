# =============================================================================
# app.py — Streamlit Fraud Detection Web Application
# =============================================================================
# Run:  streamlit run app.py
#
# Features:
#   • Dark financial-grade UI
#   • Manual transaction input with preset quick-fills
#   • Real-time fraud probability + risk-level badge
#   • Adjustable detection threshold (sidebar)
#   • Session transaction history table
#   • Info tabs: How it works / Cost model / Model details
#
# Pre-requisites:
#   Run main.py first to generate outputs/best_model.pkl
# =============================================================================

import os
import sys
import time
import warnings
import math

import numpy as np
import pandas as pd
import joblib
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title = "Fraud Detection System",
    page_icon  = "🛡️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# =============================================================================
# GLOBAL CSS  (dark finance theme)
# =============================================================================
st.markdown("""
<style>
/* ── App background ─────────────────────────────────────────────────────── */
.stApp { background-color: #0f1117; color: #c9d1d9; font-family: 'Inter', sans-serif; }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #13151f;
    border-right: 1px solid #3a3d52;
}

/* ── Cards ──────────────────────────────────────────────────────────────── */
.metric-card {
    background: #1a1d2e;
    border: 1px solid #3a3d52;
    border-radius: 14px;
    padding: 22px 26px;
    margin: 8px 0;
}
.fraud-result {
    background: linear-gradient(135deg, #1a1d2e 0%, #2d1a1a 100%);
    border: 2px solid #ff4757;
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
}
.safe-result {
    background: linear-gradient(135deg, #1a1d2e 0%, #1a2d1a 100%);
    border: 2px solid #2ed573;
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
}
.warn-result {
    background: linear-gradient(135deg, #1a1d2e 0%, #2d2a1a 100%);
    border: 2px solid #ffa502;
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
}

/* ── Gauge bar ──────────────────────────────────────────────────────────── */
.gauge-wrap { background: #21262d; border-radius: 8px; height: 14px;
              margin: 6px 0 2px 0; }
.gauge-fill { height: 14px; border-radius: 8px; }
.gauge-labels { display: flex; justify-content: space-between;
                font-size: 10px; color: #8b949e; }

/* ── Analyse button ─────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1e90ff, #a29bfe);
    color: white; border: none; border-radius: 10px;
    padding: 13px 30px; font-size: 16px; font-weight: 700;
    width: 100%; letter-spacing: 0.5px;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: 0.88; }

/* ── Section headings ───────────────────────────────────────────────────── */
h1, h2, h3 { color: #e6edf3 !important; }

/* ── Input labels ───────────────────────────────────────────────────────── */
.stSlider label, .stNumberInput label,
.stSelectbox label, .stTextInput label {
    color: #8b949e !important; font-size: 13px;
}

/* ── Divider ────────────────────────────────────────────────────────────── */
hr { border-color: #3a3d52; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CATEGORY MAPPING  (matches the 11 categories in the dataset)
# =============================================================================
CATEGORIES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net",
    "grocery_pos", "health_fitness", "home", "kids_pets",
    "misc_net", "misc_pos", "personal_care", "shopping_net",
    "shopping_pos", "travel",
]

CATEGORY_ENC = {cat: i for i, cat in enumerate(sorted(CATEGORIES))}


# =============================================================================
# MODEL LOADING
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Load serialised model bundle. Returns (model, threshold, feature_list)."""
    model_path = "outputs/best_model.pkl"
    if not os.path.exists(model_path):
        return None, None, None
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["threshold"], bundle["features"]


# =============================================================================
# PREDICTION
# =============================================================================
def predict(model, features: list, inp: dict, threshold: float):
    """
    Build a feature vector from user inputs and return (probability, label).
    """
    row = np.array([inp.get(f, 0.0) for f in features], dtype=np.float64)
    prob = float(model.predict_proba(row.reshape(1, -1))[0, 1])
    pred = int(prob >= threshold)
    return prob, pred


def risk_info(prob: float):
    """Return (label, colour, card_class) based on fraud probability."""
    if prob < 0.20:
        return "🟢  Very Low Risk",  "#2ed573", "safe-result"
    if prob < 0.40:
        return "🟡  Low Risk",       "#ffa502", "warn-result"
    if prob < 0.60:
        return "🟠  Moderate Risk",  "#ff6b35", "warn-result"
    if prob < 0.80:
        return "🔴  High Risk",      "#ff4757", "fraud-result"
    return     "🚨  Critical Risk",  "#ff0033", "fraud-result"


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar(default_threshold: float) -> float:
    with st.sidebar:
        st.markdown("## 🛡️ Fraud Detection")
        st.markdown("*Real-time transaction screening*")
        st.divider()

        st.markdown("### ⚙️ Detection Settings")
        threshold = st.slider(
            "Decision Threshold",
            min_value = 0.05, max_value = 0.80,
            value = float(default_threshold) if default_threshold is not None else 0.5,
            step      = 0.01,
            help      = "Lower → more sensitive (catches more fraud, "
                        "may flag more legitimate transactions).",
        )

        st.divider()
        st.markdown("### 📊 Risk Levels")
        levels = [
            ("🟢  Very Low",  "0 – 20%"),
            ("🟡  Low",       "20 – 40%"),
            ("🟠  Moderate",  "40 – 60%"),
            ("🔴  High",      "60 – 80%"),
            ("🚨  Critical",  "80 – 100%"),
        ]
        for label, rng in levels:
            st.markdown(f"**{label}** &ensp; `{rng}`")

        st.divider()
        st.caption("Built with Gradient Boosting + Streamlit\n"
                   "Dataset: Kaggle Credit-Card Fraud")

    return threshold


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # ── Load model ────────────────────────────────────────────────────────────
    model, default_threshold, feat_names = load_model()

    if model is None:
        st.error("❌ **Model not found.**  "
                 "Please run `python main.py` first to train the model "
                 "and generate `outputs/best_model.pkl`.")
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    threshold = render_sidebar(default_threshold)

    # ── Session history ───────────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("# 🛡️ Credit Card Fraud Detection")
    st.markdown(
        "*Enter transaction details and click **Analyse** to receive "
        "an instant fraud-risk assessment.*"
    )
    st.divider()

    # ── Two-column layout ─────────────────────────────────────────────────────
    col_in, col_out = st.columns([1.1, 0.9], gap="large")

    # ==========================================================================
    # LEFT — INPUT FORM
    # ==========================================================================
    with col_in:
        st.markdown("### 📝 Transaction Details")

        # Quick preset
        preset = st.selectbox(
            "Quick Preset  *(populates fields below)*",
            ["— Custom Input —",
             "Normal grocery purchase",
             "Suspicious late-night transaction",
             "High-value travel booking (risk)"],
        )

        PRESETS = {
            "Normal grocery purchase": dict(
                amt=42.50, hour=14, dow=2, month=6,
                is_night=0, is_weekend=0, age=38,
                distance_km=4.2, city_pop=85000,
                category="grocery_pos", gender="F",
                lat=40.71, lon=-74.01, m_lat=40.72, m_lon=-74.00,
            ),
            "Suspicious late-night transaction": dict(
                amt=389.00, hour=2, dow=5, month=11,
                is_night=1, is_weekend=1, age=27,
                distance_km=312.0, city_pop=3200,
                category="misc_net", gender="M",
                lat=34.05, lon=-118.24, m_lat=36.17, m_lon=-115.14,
            ),
            "High-value travel booking (risk)": dict(
                amt=1250.00, hour=23, dow=4, month=12,
                is_night=1, is_weekend=0, age=54,
                distance_km=890.0, city_pop=520,
                category="travel", gender="M",
                lat=47.61, lon=-122.33, m_lat=25.77, m_lon=-80.19,
            ),
        }

        p = PRESETS.get(preset, {})

        # ── Row 1 ─────────────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        amt         = c1.number_input("💵 Amount ($)",       0.01, 20000.0,
                                       float(p.get("amt", 100.0)), step=0.01)
        hour        = c2.slider("🕐 Hour of day",       0, 23, int(p.get("hour", 12)))
        age         = c3.number_input("🎂 Cardholder age",   18, 100,
                                       int(p.get("age", 35)))

        # ── Row 2 ─────────────────────────────────────────────────────────────
        c4, c5 = st.columns(2)
        category = c4.selectbox("🏪 Merchant category",
                                 sorted(CATEGORIES),
                                 index=sorted(CATEGORIES).index(
                                     p.get("category", "misc_pos")))
        gender   = c5.selectbox("👤 Gender",
                                 ["F", "M"],
                                 index=["F","M"].index(p.get("gender","F")))

        # ── Row 3 ─────────────────────────────────────────────────────────────
        c6, c7, c8 = st.columns(3)
        dow         = c6.slider("📅 Day of week  (0=Mon)",  0, 6, int(p.get("dow", 2)))
        month       = c7.slider("📆 Month",                 1, 12, int(p.get("month", 6)))
        city_pop    = c8.number_input("🏙️ City population",  100, 3_000_000,
                                       int(p.get("city_pop", 50000)), step=1000)

        # ── Row 4 ─────────────────────────────────────────────────────────────
        st.markdown("**📍 Location** *(cardholder vs merchant)*")
        lc1, lc2, lc3, lc4 = st.columns(4)
        lat   = lc1.number_input("Cardholder Lat",   -90.0, 90.0,  float(p.get("lat",  40.71)), 0.001, format="%.4f")
        lon   = lc2.number_input("Cardholder Lon",  -180.0,180.0, float(p.get("lon", -74.01)), 0.001, format="%.4f")
        m_lat = lc3.number_input("Merchant Lat",     -90.0, 90.0,  float(p.get("m_lat", 40.72)), 0.001, format="%.4f")
        m_lon = lc4.number_input("Merchant Lon",    -180.0,180.0, float(p.get("m_lon",-74.00)), 0.001, format="%.4f")

        # ── Derived fields (auto-calculated) ──────────────────────────────────
        is_night   = int(hour >= 22 or hour <= 5)
        is_weekend = int(dow >= 5)

        # Haversine distance
        R    = 6371.0
        rlat1, rlon1 = math.radians(lat),   math.radians(lon)
        rlat2, rlon2 = math.radians(m_lat), math.radians(m_lon)
        dlat = rlat2 - rlat1;  dlon = rlon2 - rlon1
        a    = math.sin(dlat/2)**2 + math.cos(rlat1)*math.cos(rlat2)*math.sin(dlon/2)**2
        distance_km = 2 * R * math.asin(math.sqrt(a))

        amt_log     = math.log1p(amt)
        amt_zscore  = (amt - 70.0) / 160.0   # rough global normalisation
        category_enc = CATEGORY_ENC.get(category, 0)
        gender_enc   = int(gender == "M")
        city_pop_log = math.log1p(city_pop)

        # Map to the exact feature order the model was trained on
        inp = {
            "amt":          amt,
            "amt_log":      amt_log,
            "amt_zscore":   amt_zscore,
            "trans_hour":   hour,
            "trans_dow":    dow,
            "trans_month":  month,
            "is_weekend":   is_weekend,
            "is_night":     is_night,
            "age":          age,
            "distance_km":  distance_km,
            "city_pop_log": city_pop_log,
            "category_enc": category_enc,
            "gender_enc":   gender_enc,
            "lat":          lat,
            "long":         lon,
            "merch_lat":    m_lat,
            "merch_long":   m_lon,
        }

        st.markdown(
            f"<small style='color:#8b949e;'>"
            f"Auto-derived: is_night={is_night} | is_weekend={is_weekend} | "
            f"distance={distance_km:.1f} km</small>",
            unsafe_allow_html=True,
        )

        analyse_btn = st.button("🔍  Analyse Transaction", use_container_width=True)

    # ==========================================================================
    # RIGHT — RESULT PANEL
    # ==========================================================================
    with col_out:
        st.markdown("### 🎯 Analysis Result")

        if not analyse_btn:
            st.markdown(
                """<div class="metric-card" style="text-align:center;padding:50px 20px">
                    <div style="font-size:52px">🔎</div>
                    <div style="color:#8b949e;margin-top:14px;font-size:15px">
                        Fill in the transaction details<br>and click
                        <b style="color:#c9d1d9">Analyse Transaction</b>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Analysing …"):
                time.sleep(0.3)
                prob, pred = predict(model, feat_names, inp, threshold)

            risk_label, risk_color, card_class = risk_info(prob)
            verdict = "⚠️  FRAUD DETECTED" if pred == 1 else "✅  TRANSACTION SAFE"
            verdict_color = "#ff4757" if pred == 1 else "#2ed573"

            st.markdown(
                f"""<div class="{card_class}">
                    <div style="font-size:28px;font-weight:900;color:{verdict_color};">
                        {verdict}
                    </div>
                    <div style="margin:18px 0 6px 0;">
                        <span style="font-size:54px;font-weight:900;color:{risk_color};">
                            {prob*100:.1f}%
                        </span>
                    </div>
                    <div style="color:#8b949e;margin-bottom:14px;">Fraud Probability</div>
                    <span style="background:{risk_color}22;color:{risk_color};
                          border:1px solid {risk_color};border-radius:20px;
                          padding:6px 20px;font-weight:700;font-size:14px;
                          letter-spacing:0.5px;">
                        {risk_label}
                    </span>
                </div>""",
                unsafe_allow_html=True,
            )

            # Gauge bar
            st.markdown("**Risk Gauge**")
            st.markdown(
                f"""<div class="gauge-wrap">
                      <div class="gauge-fill"
                           style="width:{prob*100:.1f}%;background:{risk_color};"></div>
                    </div>
                    <div class="gauge-labels">
                      <span>0%</span><span>25%</span>
                      <span>50%</span><span>75%</span><span>100%</span>
                    </div>""",
                unsafe_allow_html=True,
            )

            # Quick metrics
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Amount",     f"${amt:,.2f}")
            m2.metric("Distance",   f"{distance_km:.0f} km")
            m3.metric("Threshold",  f"{threshold:.2f}")

            # Action recommendation
            if pred == 1:
                st.error(
                    "🚨 **Action:** Block this transaction and alert the "
                    "cardholder immediately. Escalate to the fraud team."
                )
            elif prob > 0.35:
                st.warning(
                    "⚡ **Action:** Flag for manual review before processing. "
                    "Consider sending a verification SMS to the cardholder."
                )
            else:
                st.success(
                    "✅ **Action:** Approve. Transaction appears legitimate "
                    "based on current risk profile."
                )

            # Save to history
            st.session_state["history"].insert(0, {
                "Amount":     f"${amt:,.2f}",
                "Category":   category,
                "Hour":       f"{hour:02d}:00",
                "Distance":   f"{distance_km:.0f} km",
                "Probability":f"{prob*100:.1f}%",
                "Risk":       risk_label.split("  ")[1],
                "Verdict":    "FRAUD" if pred == 1 else "SAFE",
            })

    # ==========================================================================
    # TRANSACTION HISTORY
    # ==========================================================================
    if st.session_state["history"]:
        st.divider()
        st.markdown("### 📋 Session Transaction History")

        hist_df = pd.DataFrame(st.session_state["history"][:15])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        if st.button("🗑️  Clear History"):
            st.session_state["history"] = []
            st.rerun()

    # ==========================================================================
    # INFO TABS
    # ==========================================================================
    st.divider()
    tab1, tab2, tab3 = st.tabs(
        ["📖  How It Works", "💰  Cost Model", "🔬  Model Details"]
    )

    with tab1:
        st.markdown("""
#### How the System Works

1. **Feature Extraction**
   The transaction is described by 17 engineered features including transaction
   amount (raw + log + per-category z-score), time signals (hour, day, month,
   night/weekend flags), cardholder age, Haversine distance from home to merchant,
   city population, merchant category, and geographic coordinates.

2. **Risk Scoring**
   A Gradient Boosting classifier trained on 1.3 million real transactions
   outputs a fraud probability (0–100%).

3. **Threshold Decision**
   The decision boundary is tuned to **minimise total financial cost**,
   not just classification error. The optimal threshold is typically below
   0.50 to prioritise catching fraud over avoiding false alarms.

4. **Cost-Sensitive Logic**
   Missing a fraud (False Negative) costs the full transaction amount.
   A false alarm (False Positive) costs ~$10 in analyst review time.
   The model is therefore biased toward high recall.
        """)

    with tab2:
        st.markdown("""
#### Cost Matrix

| &nbsp; | Predicted: Legitimate | Predicted: Fraud |
|---|---|---|
| **Actually Legitimate** | ✅ No cost | ❌ ~\\$10 review cost |
| **Actually Fraud** | 🚨 Full amount **lost** | ✅ Amount **recovered** |

**Optimisation objective:**

$$\\text{Minimise} \\quad C_{FP} \\times n_{FP} + \\bar{\\text{amt}}_{fraud} \\times n_{FN}$$

where $C_{FP}$ = \\$10 (analyst review) and $\\bar{\\text{amt}}_{fraud}$ is the
mean fraudulent transaction amount.

**Key insight:** Lowering the threshold from 0.50 → optimal value dramatically
reduces missed fraud at the expense of a modest increase in false alarms.
The net financial benefit can be millions of dollars at scale.
        """)

    with tab3:
        model_name = type(model).__name__
        st.markdown(f"""
#### Model Configuration

| Parameter | Value |
|---|---|
| Algorithm | {model_name} |
| Training rows | ~1,296,675 |
| Test rows | ~555,719 |
| Fraud rate (train) | ~0.58% |
| Class imbalance handled by | SMOTE (20% strategy) + class weights |
| Tuning method | RandomizedSearchCV (AUC-PR objective) |
| CV folds | 5-fold Stratified |
| Decision threshold | `{default_threshold:.2f}` (cost-optimised) |
| Features used | {len(feat_names)} |

**Feature list:** {', '.join(feat_names)}

**Why Gradient Boosting?**
Sequential boosting naturally handles non-linear interactions between
features (e.g. high amount × late night × large distance).
The estimator is robust to the mild class imbalance remaining after SMOTE,
and its `feature_importances_` attribute enables direct explainability.
        """)


# =============================================================================
if __name__ == "__main__":
    main()
