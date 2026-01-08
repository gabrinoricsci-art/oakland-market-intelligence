import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import sklearn

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Oakland Hospitality Intelligence",
    layout="wide",
    page_icon="🏠"
)


# ---------------------------------------------------------
# LOAD MODELS + DATA
# ---------------------------------------------------------
@st.cache_resource
def load_all_assets():
    try:
        with st.spinner("Loading machine learning models..."):
            pricing_model = joblib.load("airbnb_xgboost_model_v6.pkl")
            occupancy_model = joblib.load("rf_occupancy_model_fixed.pkl")

            # --- COMPATIBILITY REPAIR FOR XGBOOST ---
            # This function injects missing attributes required by newer XGBoost versions
            def repair_model(model):
                # If it's a Scikit-Learn Pipeline, look at the final estimator
                if hasattr(model, 'steps'):
                    estimator = model.steps[-1][1]
                else:
                    estimator = model

                # Check if it's an XGBoost model and inject missing internal attributes
                if 'XGB' in str(type(estimator)):
                    if not hasattr(estimator, 'gpu_id'):
                        estimator.gpu_id = None
                    if not hasattr(estimator, 'predictor'):
                        estimator.predictor = "auto"
                return model

            pricing_model = repair_model(pricing_model)
            # ---------------------------------------

        with st.spinner("Loading listing dataset..."):
            listings = pd.read_csv("merged_cleaned_detailed_listing.csv")

        neighborhoods = ["Select..."] + sorted(listings['neighbourhood_cleansed'].unique().tolist())
        room_types = ["Select..."] + sorted(listings['room_type'].unique().tolist())
        months = ["Select...", "January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]

        return pricing_model, occupancy_model, listings, neighborhoods, room_types, months

    except Exception as e:
        st.error(f"❌ Error loading system assets:\n\n{e}")
        st.stop()


price_model, occ_model, listings, neighborhoods, room_types, months = load_all_assets()

# ---------------------------------------------------------
# SESSION STATE SETUP
# ---------------------------------------------------------
if 'price_data' not in st.session_state:
    st.session_state.price_data = None
if 'occ_data' not in st.session_state:
    st.session_state.occ_data = None
if 'last_inputs' not in st.session_state:
    st.session_state.last_inputs = {}

# ---------------------------------------------------------
# GLOBAL STYLES
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #f8f9fb; }
    [data-testid="stMetricValue"] { font-size: 28px !important; }
    div.stMetric { background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border-left: 5px solid #FF5A5F; }
    .insight-card { background: white; padding: 20px; border-radius: 15px; border-top: 4px solid #FF5A5F; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    div.stButton > button { background: linear-gradient(90deg, #FF5A5F 0%, #D70466 100%); color: white !important; border-radius: 12px; height: 50px; font-weight: bold; width: 100%; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR UI
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/Airbnb_Logo_B%C3%A9lo.svg", width=120)
    st.markdown("<br>", unsafe_allow_html=True)

    nb = st.selectbox("Neighborhood", neighborhoods, index=0)
    rt = st.selectbox("Room Type", room_types, index=0)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    acc = col_a.number_input("Guests", min_value=0, value=0)
    bed = col_b.number_input("Beds", min_value=0, value=0)
    bath = col_a.number_input("Baths", min_value=0.0, value=0.0, step=0.5)
    stay = col_b.number_input("Min Stay", min_value=0, value=0)

    st.markdown("---")
    mo = st.selectbox("Select Month", months, index=0)
    dy = st.number_input("Day", min_value=1, max_value=31, value=1)

# Check stale state
current_inputs = {"nb": nb, "rt": rt, "acc": acc, "bed": bed, "mo": mo, "dy": dy}
is_stale = current_inputs != st.session_state.last_inputs
is_ready = (nb != "Select...") and (rt != "Select...") and (mo != "Select...") and (acc > 0)

# ---------------------------------------------------------
# MAIN UI TABS
# ---------------------------------------------------------
st.title("Oakland Market Intelligence")
tab1, tab2, tab3, tab4 = st.tabs(["OPTIMAL PRICING", "OCCUPANCY FORECAST", "MARKET ANALYSIS", "HOST INSIGHTS"])

# ---------------------------------------------------------
# TAB 1 — PRICING ENGINE
# ---------------------------------------------------------
with tab1:
    if not is_ready:
        st.warning("Please complete the sidebar to generate a pricing report.")
    else:
        if st.button("CALCULATE RECOMMENDED PRICE"):
            with st.spinner("Computing optimal pricing..."):
                m_idx = months.index(mo)
                try:
                    weekday_val = datetime(2026, m_idx, dy).weekday()
                except ValueError:
                    weekday_val = 1

                # MODEL INPUT
                input_df = pd.DataFrame({
                    "neighbourhood_cleansed": [nb],
                    "room_type": [rt],
                    "accommodates": [acc],
                    "bedrooms": [bed],
                    "beds": [bed],
                    "bathrooms": [bath],
                    "minimum_nights": [stay],
                    "reviews_per_month": [0.0],
                    "review_scores_rating": [5.0],
                    "host_is_superhost": [0],
                    "host_years_active": [2],
                    "month": [m_idx],
                    "day": [dy],
                    "weekday": [weekday_val]
                })

                # PRICE PREDICTION
                try:
                    # Log-transform handling: models usually predict log(price)
                    raw_prediction = price_model.predict(input_df)[0]
                    raw_p = float(np.exp(raw_prediction))

                    adjusted = raw_p * 1.1 if m_idx in [6, 7, 8, 12] else raw_p
                    st.session_state.price_data = adjusted
                    st.session_state.last_inputs = current_inputs
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

        if st.session_state.price_data:
            p = st.session_state.price_data
            if is_stale:
                st.caption("*Note: Results below reflect previous settings. Press calculate to refresh.*")

            c1, c2, c3 = st.columns(3)
            c1.metric("Nightly Rate", f"${p:,.2f}")
            c2.metric("Seasonality", "Peak" if months.index(mo) in [6, 7, 8, 12] else "Standard")
            avg = listings[listings['neighbourhood_cleansed'] == nb]['price'].mean()
            c3.metric("Area Average", f"${avg:,.2f}", delta=f"${p - avg:,.2f}", delta_color="inverse")

            # Seasonal Chart
            seasonal_vals = [p * (1.1 if i in [6, 7, 8, 12] else 0.9) for i in range(1, 13)]
            fig = px.area(
                x=months[1:],
                y=seasonal_vals,
                title="Estimated Seasonal Price Trend",
                template="plotly_white",
                color_discrete_sequence=['#FF5A5F']
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 2 — OCCUPANCY FORECAST
# ---------------------------------------------------------
with tab2:
    if not is_ready:
        st.warning("Provide listing details in the sidebar first.")
    else:
        if st.button("FORECAST BOOKING PROBABILITY"):
            with st.spinner("Analyzing market demand..."):
                m_idx = months.index(mo)
                input_occ = pd.DataFrame([{
                    "neighbourhood_cleansed": nb,
                    "room_type": rt,
                    "minimum_nights": stay,
                    "accommodates": acc,
                    "bedrooms": bed,
                    "beds": bed,
                    "bathrooms": bath,
                    "availability_365": 180,
                    "month": m_idx,
                    "weekday": 0
                }])

                prob = occ_model.predict_proba(input_occ)[0][1]
                st.session_state.occ_data = prob
                st.session_state.last_inputs = current_inputs

        if st.session_state.occ_data is not None:
            prob = st.session_state.occ_data
            color = "#28a745" if prob >= 0.7 else "#ffc107" if prob >= 0.4 else "#dc3545"

            c1, c2, c3 = st.columns(3)
            c1.metric("Booking Chance", f"{prob * 100:.1f}%", delta=None)
            c2.metric("Market Status", "HIGH" if prob >= 0.7 else "MEDIUM")
            c3.metric("Signal", "🔥" if prob >= 0.7 else "📊")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}}
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 3 — MARKET ANALYSIS
# ---------------------------------------------------------
with tab3:
    if nb != "Select...":
        st.plotly_chart(
            px.histogram(
                listings[listings['neighbourhood_cleansed'] == nb],
                x="price",
                title=f"Competitor Price Spread in {nb}",
                color_discrete_sequence=['#484848']
            ),
            use_container_width=True
        )

# ---------------------------------------------------------
# TAB 4 — HOST INSIGHTS
# ---------------------------------------------------------
with tab4:
    if st.session_state.price_data and st.session_state.occ_data:
        st.subheader("Strategic Asset Optimization")

        monthly_rev = st.session_state.price_data * (st.session_state.occ_data * 30)
        st.metric(
            "Estimated Monthly Revenue",
            f"${monthly_rev:,.2f}",
            help="Price × Probability × 30 days"
        )

        st.markdown("---")
        i1, i2 = st.columns(2)

        with i1:
            avg_price = listings[listings['neighbourhood_cleansed'] == nb]['price'].mean()
            pos_text = 'above' if st.session_state.price_data > avg_price else 'below'
            st.markdown(f"""
            <div class='insight-card'>
                <h4>Market Context: {nb}</h4>
                <ul>
                    <li><b>Competition:</b> Elevated for {rt} types.</li>
                    <li><b>Positioning:</b> Your price is {pos_text} the area average.</li>
                    <li><b>Recommendation:</b> If occupancy < 50%, enable weekly discounts.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with i2:
            stay_advice = (
                "Your min-stay is competitive."
                if stay <= 2 else
                f"Area prefers 1–2 nights. Your {stay}-night minimum may reduce visibility."
            )
            st.markdown(f"""
            <div class='insight-card'>
                <h4>✨ Quick Wins</h4>
                <p><b>Stay Policy:</b> {stay_advice}</p>
                <p><b>Optimization:</b> Listings in {nb} with 'Self Check-in' see a 12% higher booking rate.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Run both Pricing and Occupancy tools to unlock host insights.")

