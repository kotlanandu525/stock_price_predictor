import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
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
# LOAD MODEL
# -----------------------------
model = load_model(
    "stock_price_rnn.h5",
    compile=False
)

scaler = joblib.load("scaler.pkl")

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.hero {
    background: linear-gradient(90deg,#1f77ff,#00d4ff);
    padding: 2rem;
    border-radius: 20px;
    text-align:center;
    color:white;
    margin-bottom:20px;
}

.hero h1{
    font-size:3rem;
    margin-bottom:10px;
}

.hero p{
    font-size:1.2rem;
}

.prediction-card{
    background: linear-gradient(135deg,#00c853,#64dd17);
    padding:25px;
    border-radius:20px;
    text-align:center;
    color:white;
    font-size:32px;
    font-weight:bold;
}

.metric-box{
    background-color:#1e222d;
    padding:15px;
    border-radius:15px;
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

    st.success("""
    Dataset:
    ADANIPORTS.csv
    """)

    st.info("""
    Model: SimpleRNN

    Input:
    Previous 30 Days Prices

    Output:
    Next Day Price
    """)

    st.markdown("---")

    st.write("Built using:")
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

    try:

        prices = list(
            map(float, user_input.split(","))
        )

        if len(prices) != 30:
            st.error(
                "Please enter exactly 30 stock prices."
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

            # Metrics
            col1,col2,col3 = st.columns(3)

            col1.metric(
                "Input Days",
                "30"
            )

            col2.metric(
                "Model",
                "SimpleRNN"
            )

            col3.metric(
                "Predicted Price",
                f"₹ {predicted_price:.2f}"
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Prediction Card
            st.markdown(
                f"""
                <div class='prediction-card'>
                🎯 Predicted Next Day Price<br>
                ₹ {predicted_price:.2f}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Trend Chart
            st.subheader("📊 Previous 30 Days Trend")

            df = pd.DataFrame({
                "Day": list(range(1,31)),
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
                xaxis_title="Days",
                yaxis_title="Price",
                title="Stock Closing Price Trend"
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