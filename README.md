# Tesla Stock Price Prediction Using Deep Learning

## Project Overview

This project predicts Tesla stock closing prices using Deep Learning models including:

* SimpleRNN
* LSTM
* GRU

The project performs complete data analysis, feature engineering, model training, hyperparameter tuning, and deployment using Streamlit.

## Domain

Financial Services

## Project Type

Time Series Forecasting (Regression)

## Dataset Features

* Date
* Open
* High
* Low
* Close
* Adj Close
* Volume

## Deep Learning Models

### Model 1: SimpleRNN

Captures short-term temporal dependencies in stock price data.

### Model 2: LSTM

Captures long-term dependencies and handles vanishing gradient problems.

### Model 3: GRU

Provides efficient learning with fewer parameters and achieved the best performance.

## Evaluation Metrics

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* R² Score

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* TensorFlow / Keras
* Streamlit
* Scikit-Learn

## Deployment

The final GRU model was deployed using Streamlit, allowing users to:

* Upload Tesla stock data
* Visualize stock trends
* View moving averages
* Generate next-day forecasts
* Generate future stock price predictions

## Author

Priyanshi Solanki
B.Tech Software Engineering
VIT Bhopal

