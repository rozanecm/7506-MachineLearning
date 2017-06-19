print ("Starting importing")
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
trainingSet = pd.read_csv('../CSVs/finalTraining.csv')
print "final training csv loaded."
testingSet = pd.read_csv('../CSVs/finalTesting.csv')
print "final testing csv loaded"
dfTrip = pd.read_csv('../CSVs/trip.csv')


trainingSet = trainingSet.loc[trainingSet.duration < 100000,:]


dfTrip = dfTrip.loc[:,['id','duration']]
dfTrip = dfTrip.rename(columns={'duration':'durationPosta'})
dfScore = pd.merge(testingSet,dfTrip,on =['id'],how = 'inner')

X1 = trainingSet [['historical', 'subscription_type_Subscriber','subscription_type_Customer', 'distance','start_minute', 'start_station_id', 'end_station_id',  'wind_dir_degrees', 'max_gust_speed_kmh','precipitation_cm', 'mean_humidity', 'business_day']]
X2 = testingSet [['historical', 'subscription_type_Subscriber','subscription_type_Customer', 'distance','start_minute', 'start_station_id', 'end_station_id',  'wind_dir_degrees', 'max_gust_speed_kmh','precipitation_cm', 'mean_humidity', 'business_day']]
y = trainingSet.duration

X_train = X1
X_test = X2
y_train = y
y_test = dfScore.durationPosta

print ('Scaling data ...')

# min_max_scaler = preprocessing.MinMaxScaler()
# X_train= min_max_scaler.fit_transform(X_train)
# X_test= min_max_scaler.fit_transform(X_test)

from sklearn.preprocessing import StandardScaler, RobustScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print ("---- Start algorithm ---- ")
RF = RandomForestRegressor(n_estimators=50, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=True, n_jobs=1, random_state=3, verbose=1, warm_start=False)

print ("Fitting ...")

RF.fit(X_train,y_train)

print ("Features sorted by their score:")

prediction = RF.predict(X_test)
print ("Prediction:" ,prediction)
print ("Error cuadratico medio: %.2f " % np.mean((prediction - y_test)**2))


dfComparison = pd.DataFrame(prediction)
dfScore = dfScore.loc[:,['id']]
dfScore['duracion'] = dfComparison

print ("Guardando data Frame")
dfScore.to_csv('../CSVs/RandomForestPrediction.csv',index = 'False')
