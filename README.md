
# Bharti Airtel Stock Price Prediction Using LSTM

This project uses Long Short-Term Memory (LSTM), a type of Recurrent Neural Network (RNN), to predict Bharat Airtel's stock prices based on historical data stored in BhartiAirtel.csv. Implemented in a Jupyter Notebook, the model analyzes historical trends in stock prices and forecasts future values.

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Requirements](#requirements)
4. [Model Architecture](#model-architecture)
5. [Training and Testing](#training-and-testing)
6. [Usage](#usage)
7. [Results](#results)
8. [Conclusion](#conclusion)

## Overview
Stock price prediction is a challenging problem due to various factors influencing the stock market. This project aims to demonstrate how LSTMs, well-suited for sequential data, can be applied to forecast stock prices based on past data for Bharti Airtel. 

## Dataset
The BhartiAirtel.csv file contains historical stock data for Bharti Airtel. The dataset includes the following columns:

- *Date*: The date of each stock price entry.
- *Open*: Opening price of the stock on that date.
- *High*: Highest price of the stock on that date.
- *Low*: Lowest price of the stock on that date.
- *Close*: Closing price of the stock on that date.
- *Volume*: Number of stocks traded on that date.
- etc....

### Loading the Dataset
The dataset can be loaded using Pandas in the notebook:

python
import pandas as pd

# Load the dataset
data = pd.read_csv('BhartiAirtel.csv')
data.head()


## Requirements
The following libraries are required to run this project:
- Python 3.x
- Jupyter Notebook
- Numpy
- Pandas
- Matplotlib
- Scikit-Learn
- TensorFlow or Keras

Install the required packages by running:
bash
pip install numpy pandas matplotlib scikit-learn tensorflow


## Model Architecture
The model uses an LSTM architecture designed for time series forecasting:
- *LSTM Layers*: Learn patterns in stock prices over time.
- *Dense Layer*: Outputs the final stock price prediction.

## Training and Testing
1. *Data Preprocessing*: Normalize stock prices to improve model performance.
2. *Data Splitting*: Divide the data into training and testing sets to evaluate the model.
3. *Model Training*: Train the LSTM model on the training data.
4. *Evaluation*: Use Mean Squared Error (MSE) to assess prediction accuracy.

## Usage
1. Clone this repository or download the Jupyter Notebook and BhartiAirtel.csv file.
2. Open the Jupyter Notebook.
3. Run the notebook cells to load data, preprocess, train, and predict stock prices.

## Results
After training, the model predicts stock prices on test data. Visualization of predictions versus actual stock prices can be done as follows:

python
import matplotlib.pyplot as plt

plt.plot(y_test, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.title('Bharat Airtel Stock Price Prediction')
plt.legend()
plt.show()


## Conclusion
This project showcases LSTM's capability in predicting Bharat Airtel stock prices. While the model captures certain trends, stock price prediction remains challenging due to inherent market volatility.
