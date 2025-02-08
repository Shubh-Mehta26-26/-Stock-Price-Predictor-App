import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Set Seaborn Style
sns.set_style("darkgrid")

# Streamlit Title
st.title("📈 Stock Price Predictor App")

# User Input for Stock Symbol
stock = st.text_input("🔍 Enter Stock Symbol (e.g., GOOG, AAPL)", "GOOG")

# Add a loading spinner
with st.spinner("Fetching stock data..."):
    end = datetime.now()
    start = datetime(end.year - 20, end.month, end.day)
    google_data = yf.download(stock, start, end)

# Display Stock Data
st.subheader("📊 Stock Data")
st.write(google_data)

# Compute Moving Averages
google_data["MA_100"] = google_data["Close"].rolling(100).mean()
google_data["MA_200"] = google_data["Close"].rolling(200).mean()
google_data["MA_250"] = google_data["Close"].rolling(250).mean()

# Plot Function
def plot_graph(title, values):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(google_data["Close"], label="Original Close Price", color="blue", alpha=0.7)
    ax.plot(values, label=title, color="orange", linestyle="dashed")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Stock Price", fontsize=12)
    ax.legend()
    st.pyplot(fig)

# Display Moving Averages
st.subheader("📈 Moving Averages")
plot_graph("100-day MA", google_data["MA_100"])
plot_graph("200-day MA", google_data["MA_200"])
plot_graph("250-day MA", google_data["MA_250"])

# Load Model
try:
    model = load_model("Latest_stock_price_model.keras")
    st.success("✅ Model Loaded Successfully!")
except:
    st.error("⚠️ Model file not found! Please ensure 'Latest_stock_price_model.keras' is in the same directory.")
    st.stop()

# Data Preprocessing
splitting_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data["Close"][splitting_len:])

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test)

x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Show progress while making predictions
st.subheader("📉 Predicting Stock Prices...")
progress_bar = st.progress(0)

# Predict Stock Prices
predictions = model.predict(x_data)
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Update progress bar
for percent_complete in range(100):
    progress_bar.progress(percent_complete + 1)

# Plot Predictions
st.subheader("📉 Original vs Predicted Stock Prices")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(google_data["Close"][:splitting_len + 100], label="Train Data (Not Used)", color="gray", alpha=0.6)
ax.plot(google_data.index[splitting_len + 100:], inv_y_test, label="Original Test Data", color="blue")
ax.plot(google_data.index[splitting_len + 100:], inv_predictions, label="Predicted Test Data", color="red", linestyle="dashed")
ax.set_title("Original vs Predicted Stock Prices", fontsize=14)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Stock Price", fontsize=12)
ax.legend()
st.pyplot(fig)

st.success("🎉 Prediction Completed!")
