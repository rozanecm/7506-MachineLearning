import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
# import ML packages
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics
from sklearn import neural_network

print "finished importing"
print "start csv loading"

# CSV Loading
trainingSet = pd.read_csv('../CSVs/tempTraining.csv')
print "improve trip train csv loaded."
testingSet = pd.read_csv('../CSVs/tempTesting.csv')
print "improved trip test csv loaded"
testingOriginal = pd.read_csv('../CSVs/trip_test.csv')
print "trip test csv loaded"

print "finished csv loading"

# KERAS SECTION
from keras.models import Sequential
from keras.layers import Dense, Activation

print "creating model..."
model = Sequential()
# add first layer which will receive all 21 cols as input, and will be formed by 32 neurons.
# 
# In this case, we initialize the network weights to a small random number generated from a uniform 
# distribution ('uniform'), in this case between 0 and 0.05 because that is the default uniform weight 
# initialization in Keras.
print "adding layers..."

model.add(Dense(64, input_dim=21, activation="relu"))
model.add(Dense(32, activation="tanh"))
model.add(Dense(18, activation="sigmoid"))
model.add(Dense(9, activation="elu"))
model.add(Dense(1))

# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
                      loss='mean_squared_error')




print "fitting model..."
# Fit the model
model.fit(X.as_matrix(), y.as_matrix(), validation_split=0.3, epochs=5, batch_size=10)
# model.fit(X_train.as_matrix(), y_train.as_matrix(), epochs=10, batch_size=10)


# print "X_test shape: ", X_test.shape
# print "evaluating model..."
# loss_and_metrics = model.evaluate(X_test.as_matrix(), y_test, batch_size=128, verbose=1)
# print "scores: ", model.evaluate(X_test.as_matrix(), y_test.as_matrix())
# print "model.metrics_names: ", model.metrics_names
# print "scores: ", scores
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
