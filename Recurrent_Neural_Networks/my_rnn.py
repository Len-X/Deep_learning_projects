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
