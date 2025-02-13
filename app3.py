import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Set Seaborn style for better visuals
sns.set_style("darkgrid")

# Streamlit app title with enhanced UI
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>Stock Price Predictor</h1>
""", unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.header("Please enter stock symbol:")
stock = st.sidebar.text_input("Enter Stock Symbol (e.g., GOOG, AAPL)", "GOOG")

# Fetch stock data
with st.spinner("Fetching stock data..."):
    end = datetime.now()
    start = datetime(end.year - 20, end.month, end.day)
    stock_data = yf.download(stock, start, end)

# Display stock data
st.subheader("Stock Data")
st.dataframe(stock_data.style.set_properties(**{'background-color': '#f9f9f9', 'border': '1px solid #ddd'}))

# Calculate moving averages
stock_data["MA_100"] = stock_data["Close"].rolling(100).mean()
stock_data["MA_200"] = stock_data["Close"].rolling(200).mean()
stock_data["MA_250"] = stock_data["Close"].rolling(250).mean()

# Function to plot stock prices with moving averages
def plot_graph(title, values):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data["Close"], label="Original Close Price", color="blue", alpha=0.7)
    ax.plot(values, label=title, color="orange", linestyle="dashed")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Stock Price", fontsize=12)
    ax.legend()
    st.pyplot(fig)

# Display moving averages with expanders
st.subheader("Moving Averages")
with st.expander("View 100-day Moving Average"):
    plot_graph("100-day MA", stock_data["MA_100"])
with st.expander("View 200-day Moving Average"):
    plot_graph("200-day MA", stock_data["MA_200"])
with st.expander("View 250-day Moving Average"):
    plot_graph("250-day MA", stock_data["MA_250"])

# Load trained model for prediction
try:
    model = load_model("Latest_stock_price_model.keras")
    st.success("Model Loaded Successfully!")
except:
    st.error("Model file not found! Please ensure 'Latest_stock_price_model.keras' is in the same directory.")
    st.stop()

# Prepare test data
splitting_len = int(len(stock_data) * 0.7)
x_test = pd.DataFrame(stock_data["Close"][splitting_len:])

# Scale data for model input
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test)

# Create input sequences for model prediction
x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Show prediction progress
st.subheader("Predicting Stock Prices...")
progress_bar = st.progress(0)

# Make predictions
predictions = model.predict(x_data)
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Update progress bar
for percent_complete in range(100):
    progress_bar.progress(percent_complete + 1)

# Plot original vs predicted stock prices
st.subheader("Original vs Predicted Stock Prices")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(stock_data["Close"][:splitting_len + 100], label="Train Data (Not Used)", color="gray", alpha=0.6)
ax.plot(stock_data.index[splitting_len + 100:], inv_y_test, label="Original Test Data", color="blue")
ax.plot(stock_data.index[splitting_len + 100:], inv_predictions, label="Predicted Test Data", color="red", linestyle="dashed")
ax.set_title("Original vs Predicted Stock Prices", fontsize=14)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Stock Price", fontsize=12)
ax.legend()
st.pyplot(fig)

# Prediction completion message
st.success("Prediction Completed!")
