import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model

# Page Configuration
st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

h1 {
    color: #0f172a;
    text-align: center;
}

h2, h3 {
    color: #2563eb;
}

[data-testid="stSidebar"] {
    background-color: #e0f2fe;
}

.stAlert {
    border-radius: 10px;
}

div[data-testid="metric-container"] {
    background-color: #dbeafe;
    border: 1px solid #93c5fd;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    "<h1> Tesla Stock Price Prediction</h1>",
    unsafe_allow_html=True
)

st.write("Upload Tesla stock CSV file to predict future closing price.")

# Sidebar
st.sidebar.title("Tesla Stock Prediction")

st.sidebar.info("""
Project:
Tesla Stock Price Prediction using
SimpleRNN, LSTM and GRU

Developer:
Priyanshi Solanki

Domain:
Financial Services
""")

# Load Model and Scaler
model = load_model("tesla_gru_model.h5", compile=False)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# File Upload
uploaded_file = st.file_uploader("Upload TSLA CSV File", type=["csv"])

if uploaded_file is not None:

    st.info("""
    🤖 Final Model Used: GRU

    📊 Evaluation Metrics:
    • RMSE
    • MAE
    • MSE
    • R² Score

    📈 Forecast Horizons:
    • 1 Day
    • 5 Days
    • 10 Days
    """)

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Latest Close", f"${round(df['Close'].iloc[-1], 2)}")

    st.subheader("Dataset Statistics")
    st.write(df.describe())

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    st.subheader("Tesla Closing Price Chart")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        df.index,
        df["Close"],
        color="#2563eb",
        linewidth=2
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.set_title("Tesla Closing Price Over Time")
    st.pyplot(fig)

    st.subheader("20-Day Moving Average")

    df["MA20"] = df["Close"].rolling(20).mean()

    fig_ma, ax_ma = plt.subplots(figsize=(10, 5))

    ax_ma.plot(
        df.index,
        df["Close"],
        label="Close Price",
        color="#2563eb"
    )

    ax_ma.plot(
        df.index,
        df["MA20"],
        label="20 Day Moving Average",
        color="#ef4444"
    )

    ax_ma.set_xlabel("Date")
    ax_ma.set_ylabel("Price")
    ax_ma.set_title("Close Price vs 20-Day Moving Average")
    ax_ma.legend()

    st.pyplot(fig_ma)

    close_data = df[["Close"]]
    scaled_data = scaler.transform(close_data)

    last_60_days = scaled_data[-60:]
    X_input = last_60_days.reshape(1, 60, 1)

    predicted_scaled = model.predict(X_input)
    predicted_price = scaler.inverse_transform(predicted_scaled)

    st.subheader("Next Day Prediction")

    st.metric(
        "Predicted Next Day Close Price",
        f"${predicted_price[0][0]:.2f}"
    )

    st.subheader("Next 5 Days Forecast")

    temp_input = scaled_data[-60:].reshape(1, 60, 1)
    future_predictions = []

    for i in range(5):
        pred = model.predict(temp_input)
        future_predictions.append(pred[0][0])

        temp_input = np.append(
            temp_input[:, 1:, :],
            pred.reshape(1, 1, 1),
            axis=1
        )

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_prices = scaler.inverse_transform(future_predictions)

    forecast_df = pd.DataFrame({
        "Day": ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"],
        "Predicted Close Price": future_prices.flatten()
    })

    st.table(forecast_df)

    st.success("Forecast generated successfully using GRU Deep Learning Model.")

else:
    st.info("Please upload CSV file.")

st.markdown("---")

st.subheader("Project Summary")

st.write("""
This project predicts Tesla stock closing prices using Deep Learning models including
SimpleRNN, LSTM, and GRU.

The final deployed model is GRU, which achieved the best forecasting performance.

The application allows users to:

• Upload Tesla stock data  
• Visualize stock trends  
• View moving averages  
• Generate future price forecasts  
• Analyze historical market behavior
""")

st.caption(
    "Built with Streamlit | Tesla Stock Prediction using Deep Learning (GRU)"
)