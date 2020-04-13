# Reccurent Neural Network

# PART 1 - Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv("DL/Recurrent_Neural_Networks/Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values
# to make it an array (and not a vector) we use "from' and "to" in columns selection. So 1:2, instead of only 1.

# Feature Scaling (using Normalization)
from sklearn.preprocessing import MinMaxScaler
# scaler (sc)
cs = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = cs.fit_transform(training_set)

# Creating data structure with 60 timesteps and 1 output
X_train = []
y_train = []
# t-60 so we need to start from 60th observation. Up to 1258 (not including last number 1258), so in fact 1257
# first i = 60
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60: i, 0]) # from 0 to 59 observations, not including upper bound i, so i-1
    y_train.append(training_set_scaled[i, 0]) # 60th observation - t+1 (59+1)
X_train, y_train = np.array(X_train), np.array(y_train) # converting list to an array

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# X_train.shape[0] is number of all observations (number of rows)
# X_train.shape[1] is number of timesteps (number of columns)
# 1 - is number of indicators/predictors (in our case it's only one: the opening price)

# PART 2 - Building the RNN

# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularization
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
# units = 50 is number of neurons in (each) LSTM layer. This number needs to be high.
# return_sequences=True - to make stacked(!) LSTM network
# input_shape is still 3-dimensional. We don't include total number of observations as it is already 'included'
regressor.add(Dropout(0.2))
# dropping 20% of neurons in the layer

# Adding the second LSTM layer and some Dropout regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding the third LSTM layer and some Dropout regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding the forth LSTM layer and some Dropout regularization
regressor.add(LSTM(units=50))
# no "return_sequence" as we don't need to stack no more - this is a last layer
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units=1))
# units=1 - the dimension of the output layer which is one. Just one output: stock price at time T+1

# Compiling the RNN
regressor.compile(optimizer="adam", loss="mean_squared_error")

# Fitting the RNN to the Training set
# Training phase
regressor.fit(X_train, y_train, epochs=100, batch_size=32)


# PART 3 - Making the predictions and visualising the results

# Getting the real stock price of (January) 2017
dataset_test = pd.read_csv("DL/Recurrent_Neural_Networks/Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of (January) 2017
# Concatenate both data sets: training and test. And decide along which axis to combine them by.
# axis=0 - to concatenate vertically (by columns)
dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1) # reshape
inputs = cs.transform(inputs) # scale by using cs method

# getting the test observations we need to predict
X_test = []
for i in range(60, 80): # 60+20(test set)=80 days
    X_test.append(inputs[i-60: i, 0]) # from 0 to 59 observations, not including upper bound i, so i-1. 0 for column
X_test = np.array(X_test)

# to add a 3-D structure
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)

# rescale test data back to original numbers by inversing predicted stock prices
predicted_stock_price = cs.inverse_transform(predicted_stock_price)


# Visualizing the results
plt.plot(real_stock_price, color = "red", label = "Actual Google Stock Price 01.2017")
plt.plot(predicted_stock_price, color = "blue", label = "Predicted Google Stock Price 01.2017")
plt.title("Google Stock Price Prediction Model")
plt.xlabel("Time in days")
plt.ylabel("Price in USD")
plt.legend()
plt.show()
