import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
from tensorflow.keras.models import load_model

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Stock Forecasting",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# LOAD MODEL & SCALER
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "stock_price_rnn.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

model = None
scaler = None

try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, compile=False)
    else:
        st.error("❌ stock_price_rnn.h5 not found")

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    else:
        st.error("❌ scaler.pkl not found")

except Exception as e:
    st.error(f"Loading Error: {e}")

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.hero{
    background:linear-gradient(90deg,#1f77ff,#00d4ff);
    padding:2rem;
    border-radius:20px;
    text-align:center;
    color:white;
    margin-bottom:20px;
}

.prediction-card{
    background:linear-gradient(135deg,#00c853,#64dd17);
    padding:25px;
    border-radius:20px;
    text-align:center;
    color:white;
    font-size:32px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class='hero'>
<h1>📈 AI Stock Price Forecasting System</h1>
<p>SimpleRNN Based Next-Day Stock Price Prediction</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("📊 Project Details")

    st.success("Dataset: ADANIPORTS.csv")

    st.info("""
Model: SimpleRNN

Input:
Previous 30 Days Prices

Output:
Next Day Price
""")

    st.markdown("---")

    st.write("✔ TensorFlow")
    st.write("✔ Streamlit")
    st.write("✔ Plotly")
    st.write("✔ Scikit-Learn")

# -----------------------------
# INPUT
# -----------------------------
st.subheader("Enter Previous 30 Closing Prices")

user_input = st.text_area(
    "Comma Separated Prices",
    height=120,
    placeholder="100,101,102,103,104..."
)

predict = st.button(
    "🚀 Predict Next Day Price",
    use_container_width=True
)

# -----------------------------
# PREDICTION
# -----------------------------
if predict:

    if model is None or scaler is None:
        st.error(
            "Model or scaler file is missing. Upload stock_price_rnn.h5 and scaler.pkl to GitHub."
        )

    else:

        try:

            prices = [float(x.strip()) for x in user_input.split(",")]

            if len(prices) != 30:
                st.error("Please enter exactly 30 stock prices.")

            else:

                arr = np.array(prices).reshape(-1, 1)

                scaled = scaler.transform(arr)

                X = scaled.reshape(1, 30, 1)

                prediction = model.predict(X, verbose=0)

                predicted_price = scaler.inverse_transform(
                    prediction
                )[0][0]

                col1, col2, col3 = st.columns(3)

                col1.metric("Input Days", "30")
                col2.metric("Model", "SimpleRNN")
                col3.metric(
                    "Predicted Price",
                    f"₹ {predicted_price:.2f}"
                )

                st.markdown(
                    f"""
                    <div class='prediction-card'>
                    🎯 Predicted Next Day Price<br>
                    ₹ {predicted_price:.2f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.subheader("📊 Previous 30 Days Trend")

                df = pd.DataFrame({
                    "Day": range(1, 31),
                    "Price": prices
                })

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=df["Day"],
                        y=df["Price"],
                        mode="lines+markers",
                        name="Stock Price"
                    )
                )

                fig.update_layout(
                    height=500,
                    template="plotly_dark",
                    title="Stock Closing Price Trend",
                    xaxis_title="Days",
                    yaxis_title="Price"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.balloons()

        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.markdown(
    """
    <center>
    <h4>🚀 AI Stock Forecasting Dashboard</h4>
    </center>
    """,
    unsafe_allow_html=True
)
