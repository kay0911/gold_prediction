import yfinance as yf
import datetime
import math
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class Dataset():
  def __init__(self):
    self.end_date = None
    self.start_date = None
    self.data = None
    self.scaler = None
  
  def load_data(self, ticker,years):
    self.end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    self.start_date = (datetime.datetime.today() - datetime.timedelta(days=math.ceil(years*365.25))).strftime('%Y-%m-%d')
    self.data = yf.Ticker(ticker).history(start=self.start_date, end=self.end_date)
    self.data.reset_index(inplace=True)
    return self.data

  def build_normalized(self, arr):
    self.scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = self.scaler.fit_transform(arr.reshape(-1, 1))
    return scaled_data

  def normalized(self, arr):
    scaled_data = self.scaler.transform(arr.reshape(-1, 1))
    return scaled_data

  def inverse_normalized(self, arr):
    scaled_data = self.scaler.inverse_transform(arr.reshape(-1, 1))
    return scaled_data

  def create_dataset(self, data, window_size, future_size):
    X, y = [], []
    for i in range(len(data) - window_size - future_size + 1):
        X.append(data[i:i + window_size, 0])
        y.append(data[i + window_size:i + window_size + future_size, 0])
    return np.array(X), np.array(y)

  def split_data(self, X, y, train_size: float = 0.8):
    train_size = int(len(X) * train_size)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return (X_train, y_train), (X_test, y_test)
