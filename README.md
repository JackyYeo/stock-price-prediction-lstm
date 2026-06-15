# stock-price-prediction-lstm
LSTM-based deep learning pipeline to predict stock price movements for AAPL, TSLA, AMZN, MSFT using historical OHLCV data.

## Overview
Built and evaluated a multi-stage LSTM classification and regression 
pipeline on 15+ years of OHLCV data. Includes hyperparameter tuning, 
confidence-filtered backtesting, multi-stock generalization, and 
feature explainability analysis.

## Stages
1. Baseline → Enhanced LSTM classification and regression variants
2. Hyperparameter grid search (lookback window × forecast horizon)
3. Backtesting with confidence filtering (Sharpe: 0.55)
4. Multi-stock generalization across AAPL, TSLA, AMZN, MSFT
5. Permutation-based feature importance (top features: EMA10, Close, MACDsig)

## Tech Stack
Python, TensorFlow/Keras, Pandas, NumPy, Scikit-learn, 
yfinance, Matplotlib

## How to Run
pip install -r requirements.txt
jupyter notebook FYP_Project.ipynb
