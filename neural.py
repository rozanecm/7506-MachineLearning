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
trainingSet = pd.read_csv('../CSVs/improved_trip_train.csv')
print "improve trip train csv loaded."
#testingSet = pd.read_csv('../CSVs/improved_trip_test.csv')
print "improved trip test csv loaded"
#testingOriginal = pd.read_csv('../CSVs/trip_test.csv')
print "trip test csv loaded"

print "finished csv loading"

trainingSet = trainingSet.loc[trainingSet.duration < 1133,:]
print ("Training set filtrado")
X = trainingSet [['end_day', 'historical', 'bike_id', 'distance', 'start_hour_13', 'viajes', 'end_day_7', 'start_hour_14', 'start_day', 'subscription_type_Subscriber']]
            
y = trainingSet.duration

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 2)
from sklearn.preprocessing import StandardScaler, RobustScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# KERAS SECTION
from keras.models import Sequential
from keras.layers import Dense, Activation

# create model
model = Sequential()
# add first layer which will receive all 194 cols as input, and will be formed by 32 neurons.
# 
# In this case, we initialize the network weights to a small random number generated from a uniform 
# distribution ('uniform'), in this case between 0 and 0.05 because that is the default uniform weight 
# initialization in Keras.
model.add(Dense(30, input_dim=10, activation="relu"))
model.add(Dense(6, activation="relu"))
# model.add(18, Activation('relu'))          no se que onda esto
model.add(Dense(1, activation='sigmoid'))

# For a mean squared error regression problem
model.compile(optimizer='adam',
                      loss='mse')

# Fit the model
model.fit(X_train, y_train, epochs=15, batch_size=1000)



# evaluate the model
#scores = model.evaluate(X_train, y_train)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
