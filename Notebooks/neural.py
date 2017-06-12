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
testingSet = pd.read_csv('../CSVs/improved_trip_test.csv')
print "improved trip test csv loaded"
testingOriginal = pd.read_csv('../CSVs/trip_test.csv')
print "trip test csv loaded"

print "finished csv loading"

X = trainingSet[['id',
#  'duration',
#  'start_date',
 'start_dayOfWeek',
 'start_week',
 'start_quarter',
#  'start_time',
 'start_hour',
 'start_minute',
#  'start_station_name',
 'start_station_id',
#  'end_date',
 'end_dayOfWeek',
 'end_week',
 'end_quarter',
#  'end_time',
 'end_hour',
 'end_minute',
#  'end_station_name',
 'end_station_id',
 'bike_id',
#  'subscription_type',
#  'zip_code',
 'distance',
 'historical',
 'viajes',
 'max_temperature_c',
 'min_temperature_c',
 'max_humidity',
 'max_sea_level_pressure_cm',
 'precipitation_cm',
 'Fog',
 'Normal',
 'Rain',
 'business_day',
 'holiday',
 'year',
 'month',
 'weekday',
 'start Harry Bridges Plaza (Ferry Building)',
 'start Market at Sansome',
 'start San Francisco Caltrain 2 (330 Townsend)',
 'start Market at 10th',
 'start Townsend at 7th',
 'start Powell at Post (Union Square)',
 'start 2nd at Folsom',
 'start San Francisco Caltrain (Townsend at 4th)',
 'start 2nd at Townsend',
 'start Beale at Market',
 'start Embarcadero at Bryant',
 'start Temporary Transbay Terminal (Howard at Beale)',
 'start Howard at 2nd',
 'start Steuart at Market',
 'start Rengstorff Avenue / California Street',
 'start San Jose Diridon Caltrain Station',
 'start Grant Avenue at Columbus Avenue',
 'start 2nd at South Park',
 'start Embarcadero at Sansome',
 'start Davis at Jackson',
 'start St James Park',
 'start South Van Ness at Market',
 'start Powell Street BART',
 'start Broadway St at Battery St',
 'start Civic Center BART (7th at Market)',
 'start Yerba Buena Center of the Arts (3rd @ Howard)',
 'start Clay at Battery',
 'start San Antonio Caltrain Station',
 'start Commercial at Montgomery',
 'start Cowper at University',
 'start Golden Gate at Polk',
 'start Embarcadero at Folsom',
 'start Embarcadero at Vallejo',
 'start Market at 4th',
 'start Spear at Folsom',
 'start Mechanics Plaza (Market at Battery)',
 'start Palo Alto Caltrain Station',
 'start 5th at Howard',
 'start Paseo de San Antonio',
 'start San Antonio Shopping Center',
 'start Santa Clara at Almaden',
 'start Ryland Park',
 'start San Pedro Square',
 'start Mountain View Caltrain Station',
 'start San Francisco City Hall',
 'start Post at Kearny',
 'start Castro Street and El Camino Real',
 'start Mountain View City Hall',
 'start Redwood City Caltrain Station',
 'start Stanford in Redwood City',
 'start Japantown',
 'start California Ave Caltrain Station',
 'start Evelyn Park and Ride',
 'start Washington at Kearny',
 'start MLK Library',
 'start Redwood City Medical Center',
 'start Mezes Park',
 'start SJSU 4th at San Carlos',
 'start San Jose Civic Center',
 'start Adobe on Almaden',
 'start Franklin at Maple',
 'start SJSU - San Salvador at 9th',
 'start Santa Clara County Civic Center',
 'start San Jose City Hall',
 'start Arena Green / SAP Center',
 'start San Salvador at 1st',
 'start University and Emerson',
 'start Post at Kearney',
 'start Washington at Kearney',
 'start Park at Olive',
 'start San Mateo County Center',
 'start Redwood City Public Library',
 'start Broadway at Main',
 'start San Jose Government Center',
 'end Embarcadero at Sansome',
 'end 2nd at Folsom',
 'end Temporary Transbay Terminal (Howard at Beale)',
 'end Powell Street BART',
 'end San Francisco Caltrain (Townsend at 4th)',
 'end Market at 10th',
 'end Embarcadero at Folsom',
 'end 2nd at Townsend',
 'end Harry Bridges Plaza (Ferry Building)',
 'end South Van Ness at Market',
 'end 2nd at South Park',
 'end Townsend at 7th',
 'end Commercial at Montgomery',
 'end Mountain View Caltrain Station',
 'end San Pedro Square',
 'end Market at Sansome',
 'end Civic Center BART (7th at Market)',
 'end Embarcadero at Bryant',
 'end Market at 4th',
 'end Steuart at Market',
 'end Yerba Buena Center of the Arts (3rd @ Howard)',
 'end Beale at Market',
 'end Broadway St at Battery St',
 'end San Jose Diridon Caltrain Station',
 'end 5th at Howard',
 'end Howard at 2nd',
 'end Embarcadero at Vallejo',
 'end San Francisco Caltrain 2 (330 Townsend)',
 'end MLK Library',
 'end San Antonio Shopping Center',
 'end Palo Alto Caltrain Station',
 'end Spear at Folsom',
 'end Powell at Post (Union Square)',
 'end Clay at Battery',
 'end Davis at Jackson',
 'end Washington at Kearny',
 'end Post at Kearny',
 'end Golden Gate at Polk',
 'end Cowper at University',
 'end San Antonio Caltrain Station',
 'end Paseo de San Antonio',
 'end Santa Clara County Civic Center',
 'end California Ave Caltrain Station',
 'end Mountain View City Hall',
 'end Mechanics Plaza (Market at Battery)',
 'end Santa Clara at Almaden',
 'end Grant Avenue at Columbus Avenue',
 'end Stanford in Redwood City',
 'end Redwood City Caltrain Station',
 'end San Francisco City Hall',
 'end Arena Green / SAP Center',
 'end Franklin at Maple',
 'end Castro Street and El Camino Real',
 'end San Mateo County Center',
 'end SJSU - San Salvador at 9th',
 'end Redwood City Medical Center',
 'end Japantown',
 'end Evelyn Park and Ride',
 'end SJSU 4th at San Carlos',
 'end St James Park',
 'end Ryland Park',
 'end Rengstorff Avenue / California Street',
 'end Adobe on Almaden',
 'end Park at Olive',
 'end San Jose City Hall',
 'end San Salvador at 1st',
 'end San Jose Civic Center',
 'end University and Emerson',
 'end Redwood City Public Library',
 'end Mezes Park',
 'end Washington at Kearney',
 'end Post at Kearney',
 'end Broadway at Main',
 'end San Jose Government Center',
 'start_dayOfWeek_id0',
 'start_dayOfWeek_id1',
 'start_dayOfWeek_id2',
 'start_dayOfWeek_id3',
 'start_dayOfWeek_id4',
 'start_dayOfWeek_id5',
 'start_dayOfWeek_id6',
 'end_dayOfWeek_id0',
 'end_dayOfWeek_id1',
 'end_dayOfWeek_id2',
 'end_dayOfWeek_id3',
 'end_dayOfWeek_id4',
 'end_dayOfWeek_id5',
 'end_dayOfWeek_id6',
 'subscription_type_Subscriber',
 'subscription_type_Customer']]                    
    

y = trainingSet.duration

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 2)


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
model.add(Dense(32, input_dim=194, activation="relu"))
model.add(Dense(18, activation="relu"))
# model.add(18, Activation('relu'))          no se que onda esto
model.add(Dense(1))

# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
                      loss='mse')

# Fit the model
model.fit(X_train.as_matrix(), y_train, epochs=1, batch_size=200)



# evaluate the model
scores = model.evaluate(X_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
