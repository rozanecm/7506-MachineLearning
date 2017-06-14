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
#print "improved trip test csv loaded"
#testingOriginal = pd.read_csv('../CSVs/trip_test.csv')
#print "trip test csv loaded"

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

from sklearn.neural_network import MLPRegressor

# Puedo variar el hidden_layer a mas grande o el random_state a mas grande
MLP = MLPRegressor(
    hidden_layer_sizes=(50,50,50,50) ,activation='relu', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
    random_state=2, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

MLP.fit(X_train,y_train)
prediction = MLP.predict(X_test)
print (prediction)
print ("Error cuadratico medio: %.2f " % np.mean((prediction - y_test)**2))

