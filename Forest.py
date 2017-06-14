
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 3)
from sklearn.preprocessing import StandardScaler, RobustScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print ("Comienza el entrenamiento RF")
RF = RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=True, n_jobs=1, random_state=3, verbose=1, warm_start=False)

RF.fit(X_train,y_train)
print ("FIt terminado")

print "Features sorted by their score:"
n_features = sorted(zip(map(lambda x: round(x, 4), RF.feature_importances_), X), 
             reverse=True)
print (n_features)
prediction = RF.predict(X_test)
print ("Prediction:" ,prediction)
print ("Error cuadratico medio: %.2f " % np.mean((prediction - y_test)**2))


#44202560.95 
