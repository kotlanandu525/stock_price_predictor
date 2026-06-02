import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
from tensorflow.keras.models import load_model

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Stock Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# LOAD MODEL & SCALER
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "stock_price_rnn.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

model = None
scaler = None

try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, compile=False)

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)

except Exception as e:
    st.error(f"Loading Error: {e}")

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>

.stApp{
    background-color:#0B1220;
    color:white;
}

section[data-testid="stSidebar"]{
    background-color:#111827;
}

.hero{
    background:linear-gradient(
        135deg,
        #111827,
        #1E293B
    );
    padding:35px;
    border-radius:20px;
    border:1px solid rgba(255,255,255,0.08);
    text-align:center;
    margin-bottom:25px;
}

.hero h1{
    color:white;
    font-size:46px;
    font-weight:700;
    margin-bottom:10px;
}

.hero p{
    color:#94A3B8;
    font-size:18px;
}

.prediction-card{
    background:linear-gradient(
        135deg,
        #10B981,
        #059669
    );
    padding:35px;
    border-radius:20px;
    text-align:center;
    color:white;
    margin-top:20px;
    margin-bottom:20px;
    box-shadow:0px 8px 20px rgba(0,0,0,0.3);
}

.chart-card{
    background:#111827;
    padding:15px;
    border-radius:18px;
    border:1px solid #1F2937;
}

.footer{
    text-align:center;
    color:#94A3B8;
    padding:20px;
}

div.stButton > button{
    background:#2563EB;
    color:white;
    border:none;
    height:55px;
    font-size:18px;
    font-weight:600;
    border-radius:12px;
    width:100%;
}

div.stButton > button:hover{
    background:#1D4ED8;
}

textarea{
    border-radius:12px !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="hero">
    <h1>📈 AI Stock Forecasting Dashboard</h1>
    <p>Deep Learning Based Next-Day Stock Price Prediction using SimpleRNN</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:

    st.title("📊 Dashboard Info")

    st.markdown("---")

    if model:
        st.success("🟢 Model Loaded")
    else:
        st.error("🔴 Model Missing")

    if scaler:
        st.success("🟢 Scaler Loaded")
    else:
        st.error("🔴 Scaler Missing")

    st.markdown("---")

    st.subheader("Model Details")

    st.info("""
    Dataset : ADANIPORTS

    Model : SimpleRNN

    Input Window : 30 Days

    Output : Next Day Price
    """)

    st.markdown("---")

    st.subheader("Tech Stack")

    st.write("✔ TensorFlow")
    st.write("✔ Streamlit")
    st.write("✔ Plotly")
    st.write("✔ Scikit-Learn")
    st.write("✔ NumPy")

# =====================================================
# KPI SECTION
# =====================================================
c1, c2, c3, c4 = st.columns(4)

c1.metric("Dataset", "ADANIPORTS")
c2.metric("Model", "SimpleRNN")
c3.metric("Window Size", "30 Days")
c4.metric("Framework", "TensorFlow")

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# SAMPLE DATA
# =====================================================
sample_data = (
    "1520.50,1523.80,1518.40,1527.60,1531.20,"
    "1535.90,1532.70,1540.10,1545.80,1542.30,"
    "1548.90,1551.40,1547.20,1555.60,1560.30,"
    "1558.70,1564.20,1568.90,1572.50,1569.80,"
    "1575.40,1580.10,1577.60,1585.20,1590.70,"
    "1588.40,1595.90,1600.30,1605.80,1610.50"
)

# =====================================================
# INPUT SECTION
# =====================================================
st.subheader("📥 Enter Previous 30 Closing Prices")

colA, colB = st.columns([4,1])

with colA:
    user_input = st.text_area(
        "Comma Separated Prices",
        height=150,
        placeholder="100,101,102,103..."
    )

with colB:
    st.write("")
    st.write("")
    if st.button("Load Sample"):
        user_input = sample_data

predict = st.button(
    "🚀 Predict Next Day Price"
)

# =====================================================
# PREDICTION
# =====================================================
if predict:

    if model is None or scaler is None:

        st.error(
            "Model or scaler file is missing. Please upload stock_price_rnn.h5 and scaler.pkl."
        )

    else:

        try:

            prices = [
                float(x.strip())
                for x in user_input.split(",")
                if x.strip()
            ]

            if len(prices) != 30:

                st.error(
                    f"Expected 30 values but got {len(prices)}"
                )

            else:

                arr = np.array(prices).reshape(-1,1)

                scaled = scaler.transform(arr)

                X = scaled.reshape(1,30,1)

                prediction = model.predict(
                    X,
                    verbose=0
                )

                predicted_price = scaler.inverse_transform(
                    prediction
                )[0][0]

                last_price = prices[-1]

                change = predicted_price - last_price

                percent_change = (
                    change / last_price
                ) * 100

                trend = "📈 Bullish" if change > 0 else "📉 Bearish"

                # ===================================
                # METRICS
                # ===================================
                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Last Closing Price",
                    f"₹ {last_price:.2f}"
                )

                m2.metric(
                    "Predicted Price",
                    f"₹ {predicted_price:.2f}"
                )

                m3.metric(
                    "Expected Change",
                    f"{percent_change:.2f}%"
                )

                # ===================================
                # PREDICTION CARD
                # ===================================
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <h2>{trend}</h2>
                        <h1>₹ {predicted_price:.2f}</h1>
                        <h3>Change: ₹ {change:.2f}</h3>
                        <h3>{percent_change:.2f}%</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ===================================
                # CHART
                # ===================================
                st.subheader(
                    "📊 Previous 30-Day Price Trend"
                )

                df = pd.DataFrame({
                    "Day": range(1,31),
                    "Price": prices
                })

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=df["Day"],
                        y=df["Price"],
                        mode="lines+markers",
                        fill="tozeroy",
                        name="Price"
                    )
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#111827",
                    plot_bgcolor="#111827",
                    height=550,
                    margin=dict(
                        l=20,
                        r=20,
                        t=40,
                        b=20
                    ),
                    title="Stock Closing Price Trend",
                    xaxis_title="Day",
                    yaxis_title="Price",
                    font=dict(
                        color="white"
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")

st.markdown("""
<div class="footer">
    <h4>🚀 AI Stock Forecasting Dashboard</h4>
    <p>Powered by TensorFlow • Streamlit • Plotly</p>
</div>
""", unsafe_allow_html=True)
