import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import joblib
import os
from datetime import timedelta

# File paths
model_path = "C:/Users/46704/Desktop/Bitcoin/data/bitcoin_model_t1_2var_train_val.h5"
scaler_path = "C:/Users/46704/Desktop/Bitcoin/data/bitcoin_scaler_close.pkl"
data_path = "C:/Users/46704/Desktop/Bitcoin/data/dataset/bitcoin_data_scaled.csv"

# Load the pre-trained model
model = load_model(model_path)

# Load the scaler for the target variable (Close)
scaler = joblib.load(scaler_path)

# Load the CSV file with features and target
data = pd.read_csv(data_path)

# Define the features used for prediction (ensure these match the training features)
features_columns = ['Lagged_Close1', 'Lagged_Open_Close_Range']

# Make sure to only select the feature columns (not the target 'Close')
features = data[features_columns].copy()  # Avoid modifying original data

# Streamlit layout
st.set_page_config(page_title="Bitcoin Price Prediction", page_icon="📈", layout="wide")

# Add custom CSS to change background and text colors
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    .stApp {
        background-color: black;
        color: white;
    }
    .css-1offfwp {
        color: white !important;  /* Style for text */
    }
    .stButton>button {
        color: black; 
        background-color: white; /* Button style */
    }
    .stCheckbox>div [data-baseweb="checkbox"] {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    body {
        background-color: black;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .centered-text {
        text-align: center;
    }
    .small-text {
        font-size: 16px;
    }
    .green-arrow {
        color: green;
        font-size: 30px;
    }
    .red-arrow {
        color: red;
        font-size: 30px;
    }
    .streamlit-expanderHeader {
        color: white;
    }
    .streamlit-button {
        background-color: #000000;
        color: white;
    }
    .stTextInput input, .stTextArea textarea {
        background-color: black;
        color: white;
    }
    .stSelectbox, .stMultiselect {
        background-color: black;
        color: white;
    }
    /* Set Streamlit Sidebar background */
    .css-1d391kg {
        background-color: black;
        color: white;
    }
    /* Style for Streamlit file uploader */
    .css-1n0yug6 {
        background-color: black;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title centered
st.markdown('<h1 class="centered-text">Bitcoin Price Prediction</h1>', unsafe_allow_html=True)

# Ensure the data isn't empty
if data.empty:
    st.write("Error: The data file is empty or could not be read.")
else:
    # Extract the last 365 days (1 year) of data for visualization
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.tail(365)  # Last 365 days of data

    # Calculate the SMAs (10 and 50-day moving averages)
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # Identify bullish and bearish trends
    data['Bullish_Trend'] = data['SMA_10'] > data['SMA_50']
    data['Bearish_Trend'] = data['SMA_10'] < data['SMA_50']

    # Compute support and resistance
    window_size = 50  # Lookback window for support/resistance
    data['Support'] = data['Close'].rolling(window=window_size, min_periods=1).min()
    data['Resistance'] = data['Close'].rolling(window=window_size, min_periods=1).max()

    # Initialize Buy/Sell/Hold signals based on SMA crossovers (Signal 1)
    buy_signal_1 = None
    sell_signal_1 = None

    # Check for SMA crossovers (first plot signal)
    if data['SMA_10'].iloc[-1] > data['SMA_50'].iloc[-1]:
        buy_signal_1 = True
    elif data['SMA_10'].iloc[-1] < data['SMA_50'].iloc[-1]:
        sell_signal_1 = True


    # Determine signal for first plot (SMA crossover)
    if buy_signal_1:
        latest_signal_1 = ('Buy', 'green')
    elif sell_signal_1:
        latest_signal_1 = ('Sell', 'red')
    
    

    # Signal for the second plot (Support/Resistance)
    latest_signal_2 = None
    if data['Close'].iloc[-1] >= data['Resistance'].iloc[-1]:
        if data['Bullish_Trend'].iloc[-1]:  # Market is bullish
            latest_signal_2 = ('Buy', 'green')
        else:  # Market is bearish
            latest_signal_2 = ('Sell', 'red')
    elif data['Close'].iloc[-1] <= data['Support'].iloc[-1]:
        if data['Bullish_Trend'].iloc[-1]:  # Market is bullish
            latest_signal_2 = ('Buy', 'green')
        else:  # Market is bearish
            latest_signal_2 = ('Sell', 'red')
    else:
        latest_signal_2 = ('Hold', 'yellow')

    # Extract the last 10 rows for prediction (for the time_step of 10)
    last_rows = features.iloc[-10:]  # Get the last 10 rows for prediction

    # Ensure the feature order is correct (same as training)
    last_rows = last_rows[features_columns]  # Reorder to match training feature order

    # Check if the columns match the trained features
    if list(last_rows.columns) != features_columns:
        st.write("Error: The feature columns do not match the trained columns!")
    else:
        # Reshape the input to match the model's expected shape (1, 10, 5)
        input_data = last_rows.values.reshape(1, 10, len(features_columns))  # 1 sample, 10 time steps, 5 features

        # Predict Bitcoin price for t+1
        predictions_scaled = model.predict(input_data)

        # Unscale the predictions (target) to bring them back to the original scale
        predictions_unscaled = scaler.inverse_transform(predictions_scaled)

        # Get the prediction for t+1
        predicted_price_t1 = predictions_unscaled[0][0]

        # Date of prediction (which is t+1, one step ahead of the last date in the data)
        prediction_date = data['Date'].iloc[-1] + timedelta(days=1)

        # Get the real closing price of the previous day
        real_price_previous_day = data['Close'].iloc[-1]

        # Check if the predicted price went up or down from the previous day's close
        price_change = "up" if predicted_price_t1 > real_price_previous_day else "down"
        price_arrow = "&#x2191;" if price_change == "up" else "&#x2193;"
        arrow_class = "green-arrow" if price_change == "up" else "red-arrow"

        # Display the prediction and the date for t+1 with smaller text
        st.markdown(f'<h3 class="centered-text small-text">Predicted Bitcoin Price at {prediction_date.date()}: {predicted_price_t1:.2f} <span class="{arrow_class}">{price_arrow}</span></h3>', unsafe_allow_html=True)


        # Display the signal text with appropriate color
        signal_text = "Price is going up!"
        signal_color = "green" if price_change == "up" else "red"
        st.markdown(f'<h4 class="centered-text small-text" style="color:{signal_color};">{signal_text}</h4>', unsafe_allow_html=True)


    # Add centered "Technical Indicators" section header
    st.markdown('<h2 class="centered-text">Technical Indicators</h2>', unsafe_allow_html=True)

    # Create a figure with two subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

    # Set the background color of the figure and axes to black
    fig.patch.set_facecolor('black')
    ax1.set_facecolor('black')
    ax2.set_facecolor('black')

    # Set the color of the titles and labels
    ax1.title.set_color('white')
    ax1.set_xlabel('Date', color='white')
    ax1.set_ylabel('Price', color='white')
    ax1.tick_params(axis='x', rotation=45, colors='white')
    ax1.tick_params(axis='y', colors='white')

    ax2.title.set_color('white')
    ax2.set_xlabel('Date', color='white')
    ax2.set_ylabel('Price', color='white')
    ax2.tick_params(axis='x', rotation=45, colors='white')
    ax2.tick_params(axis='y', colors='white')

    # First plot: Bitcoin price with SMAs and Bullish/Bearish trends
    ax1.plot(data['Date'], data['Close'], label='Close Price', color='gray', alpha=0.5)
    ax1.plot(data['Date'], data['SMA_10'], label='SMA 10', color='blue', linestyle='--')
    ax1.plot(data['Date'], data['SMA_50'], label='SMA 50', color='red', linestyle='--')

    # Highlight Bullish and Bearish Trend Periods with lighter colors
    ax1.fill_between(data['Date'], data['Close'], data['SMA_10'], where=data['Bullish_Trend'], color='lightgreen', alpha=0.5, label='Bullish')
    ax1.fill_between(data['Date'], data['Close'], data['SMA_10'], where=data['Bearish_Trend'], color='lightcoral', alpha=0.5, label='Bearish')

    ax1.legend(loc='upper left')
    
    # Title for the first plot
    ax1.set_title('SMA 10/50 Crossover and Trend Indicators', color='white')

    # Second plot: Bitcoin price with Support and Resistance levels
    ax2.plot(data['Date'], data['Close'], label='Close Price', color='gray', alpha=0.5)  # Add Bitcoin price line here
    ax2.plot(data['Date'], data['Support'], label='Support Level', color='purple', linestyle=':', linewidth=2)
    ax2.plot(data['Date'], data['Resistance'], label='Resistance Level', color='orange', linestyle=':', linewidth=2)

    # Title for the second plot
    ax2.set_title('Support and Resistance', color='white')

    # Display the graphs
    st.pyplot(fig)

    # Display signals below both plots
    st.markdown(f'<div style="display: flex; justify-content: space-between; align-items: center; margin-left:50px; width: 100%;">'
                f'<h3 style="color:{latest_signal_1[1]}; font-size: 14px;">{latest_signal_1[0]} Signal (SMA Crossover)</h3>'
                f'<h3 style="color:{latest_signal_2[1]}; font-size: 14px;">{latest_signal_2[0]} Signal (Support/Resistance)</h3>'
                '</div>', unsafe_allow_html=True)

# Footer disclaimer
st.markdown("""
    <footer style="text-align:center; font-size:12px; color:white;">
        Disclaimer: This model is for educational purposes only. Predictions may vary and should not be used for financial decision-making without further analysis.
    </footer>
""", unsafe_allow_html=True)