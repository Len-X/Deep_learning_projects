# Artificial Neural Network
# Installing Theano, Tensorflow, Keras

# # Part 1 - Data Preprocessing
# Importing the Libraries
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv("/Users/olenahoryn/Python_Workspace/Study/DL/Artificial_Neural_Networks/Churn_Modelling.csv")
# Define independent variables (all rows of all columns, except for the last column)
X = dataset.iloc[:, 3:13].values
# Define dependent variable (all rows of the last column)
y = dataset.iloc[:, 13].values

# Encoding categorical data
labelencoder_X_country = LabelEncoder()
X[:, 1] = labelencoder_X_country.fit_transform(X[:, 1])
labelencoder_X_gender = LabelEncoder()
X[:, 2] = labelencoder_X_gender.fit_transform(X[:, 2])
# OneHotEncoder for X
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 1:]

# Splitting the dataset into the Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
# We are scaling only independent variables, no need to scale dependent(what we want to predict) variable
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)  # only fit train data! First fit,then transform
X_test = sc_X.transform(X_test)  # only transform test data (no need to fit)

# # Part 2 - Let's make the ANN!
# Importing the Keras Libraries and Packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and first hidden layer
# "units" = 6 - average number of nodes, hidden plus output layers ( (11+1) / 2 = 6)
# "relu" - rectifier activation function (for hidden layers)
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_shape=(11,)))
classifier.add(Dropout(p = 0.1)) # with dropout rate p = 0.1 (% of neurons to ignore/drop w. each iteration.
# In our case 1 of 10 = 0.1 we override/disable per each iteration)

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
classifier.add(Dropout(p = 0.1)) # with dropout

# Adding the output layer
# "units" = number of nodes in output layer. In our binary sample it is 1.
# For classifications (e.g. 3 classes) "units" = 3
# "sigmoid" activation function for binary (classes) output layer. For more categories use "softmax" function.
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

# Compiling the ANN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# # Part 3  - Making the predictions and evaluating the model
# Predicting the Test set results
# This is where we test our model - we try to predict the results (salary) based on years of experience
y_predicted = classifier.predict(X_test)
y_predicted = (y_predicted > 0.5)

# Making the Confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predicted)

# accuracy
# (1541+147)/2000

# Homework: Predicting a single new observation
'''Use our ANN model to predict if the customer with the following information will leave the bank: 

Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
So should we say goodbye to that customer ?

'''
new_prediction = classifier.predict(sc_X.transform(np.array([[0.0, 0.0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = new_prediction > 0.5

# # Part 4 - Evaluating, Improving and Tuning the ANN
# Evaluating the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_shape=(11,)))
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
    classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv=10, n_jobs=-1)
# cv = number of folds

mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed (see above part 2)

# (Parameter) Tuning the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_shape=(11,)))
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
    classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
    classifier.compile(optimizer = optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier)
parameters = {"batch_size": [25, 32],
              "epochs": [100, 500],
              "optimizer": ["adam", "rmsprop"]}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring="accuracy", cv=10)
# cv = number of folds
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_



