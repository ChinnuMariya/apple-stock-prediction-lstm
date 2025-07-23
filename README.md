# apple-stock-prediction-lstm
LSTM-based model to predict Apple stock prices using historical data from Yahoo Finance. Built with TensorFlow, yfinance, and scikit-learn.
# Project Overview
This project uses Long Short-Term Memory (LSTM), a type of recurrent neural network (RNN), to predict Apple Inc. (AAPL) stock prices based on historical stock data. The model is trained to forecast the closing price using past sequences, leveraging deep learning for time-series prediction.

# Motivation
Stock price prediction is a common use case for machine learning and deep learning models. This project helped me understand how sequence models work and how they can be applied to financial data. The aim was to gain hands-on experience with:

# Time series data preprocessing

Building and training LSTM models

Evaluating model performance using RMSE

Visualizing predictions

# Tools & Technologies
Python 

Google Colab 

NumPy & Pandas

Matplotlib & Seaborn

Scikit-learn

TensorFlow / Keras

Yahoo Finance API (yfinance)

# Dataset
Source: Yahoo Finance

Ticker: AAPL

Timeframe: 2015 - 2024

Features Used: Open, High, Low, Close, Volume

# How It Works
Data Loading & Preprocessing

Downloaded historical AAPL data using yfinance

Scaled the closing price between 0 and 1

Created sequences of past 60 days to predict the next day

## Model Architecture

LSTM layers with dropout

Dense output layer for regression

Compiled with mean_squared_error loss and adam optimizer

## Training

Trained on 80% of the data, validated on 20%

25 epochs with early stopping considerations

## Evaluation

Root Mean Squared Error (RMSE) used to evaluate the model

Predicted vs Actual prices plotted for visual comparison

# Results
<b>Final RMSE: ~5.75</b>

The model captured the general trend of the stock but struggles with sudden volatility.


# Future Improvements
Include additional technical indicators like RSI, MACD

Compare LSTM with other models (GRU, Transformer, ARIMA)

Implement multi-feature or multi-stock forecasting
