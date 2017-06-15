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
# testingSet = pd.read_csv('../CSVs/tempTesting.csv')
# print "improved trip test csv loaded"
# testingOriginal = pd.read_csv('../CSVs/trip_test.csv')
# print "trip test csv loaded"

print "finished csv loading"

X = trainingSet[['distance',
 'start_week',
 'start_minute',
 'year',
 'month',
 'weekday',
 'historical',
 'viajes',
 'max_temperature_c',
 'mean_temperature_c',
 'min_temperature_c',
 'max_dew_point_c',
 'mean_dew_point_c',
 'min_dew_point_c',
 'max_humidity',
 'mean_humidity',
 'min_humidity',
 'max_sea_level_pressure_cm',
 'mean_sea_level_pressure_cm',
 'min_sea_level_pressure_cm',
 'max_visibility_km',
 'mean_visibility_km',
 'min_visibility_km',
 'max_wind_Speed_kmh',
 'mean_wind_speed_kmh',
 'max_gust_speed_kmh',
 'precipitation_cm',
 'cloud_cover',
 'wind_dir_degrees',
 'Fog',
 'Fog-Rain',
 'Normal',
 'Rain',
 'Rain-Thunderstorm',
 'business_day',
 'holiday',
 'start 2nd at Folsom',
 'start 2nd at South Park',
 'start 2nd at Townsend',
 'start 5th at Howard',
 'start Adobe on Almaden',
 'start Arena Green / SAP Center',
 'start Beale at Market',
 'start Broadway St at Battery St',
 'start Broadway at Main',
 'start California Ave Caltrain Station',
 'start Castro Street and El Camino Real',
 'start Civic Center BART (7th at Market)',
 'start Clay at Battery',
 'start Commercial at Montgomery',
 'start Cowper at University',
 'start Davis at Jackson',
 'start Embarcadero at Bryant',
 'start Embarcadero at Folsom',
 'start Embarcadero at Sansome',
 'start Embarcadero at Vallejo',
 'start Evelyn Park and Ride',
 'start Franklin at Maple',
 'start Golden Gate at Polk',
 'start Grant Avenue at Columbus Avenue',
 'start Harry Bridges Plaza (Ferry Building)',
 'start Howard at 2nd',
 'start Japantown',
 'start MLK Library',
 'start Market at 10th',
 'start Market at 4th',
 'start Market at Sansome',
 'start Mechanics Plaza (Market at Battery)',
 'start Mezes Park',
 'start Mountain View Caltrain Station',
 'start Mountain View City Hall',
 'start Palo Alto Caltrain Station',
 'start Park at Olive',
 'start Paseo de San Antonio',
 'start Post at Kearney',
 'start Post at Kearny',
 'start Powell Street BART',
 'start Powell at Post (Union Square)',
 'start Redwood City Caltrain Station',
 'start Redwood City Medical Center',
 'start Redwood City Public Library',
 'start Rengstorff Avenue / California Street',
 'start Ryland Park',
 'start SJSU - San Salvador at 9th',
 'start SJSU 4th at San Carlos',
 'start San Antonio Caltrain Station',
 'start San Antonio Shopping Center',
 'start San Francisco Caltrain (Townsend at 4th)',
 'start San Francisco Caltrain 2 (330 Townsend)',
 'start San Francisco City Hall',
 'start San Jose City Hall',
 'start San Jose Civic Center',
 'start San Jose Diridon Caltrain Station',
 'start San Jose Government Center',
 'start San Mateo County Center',
 'start San Pedro Square',
 'start San Salvador at 1st',
 'start Santa Clara County Civic Center',
 'start Santa Clara at Almaden',
 'start South Van Ness at Market',
 'start Spear at Folsom',
 'start St James Park',
 'start Stanford in Redwood City',
 'start Steuart at Market',
 'start Temporary Transbay Terminal (Howard at Beale)',
 'start Townsend at 7th',
 'start University and Emerson',
 'start Washington at Kearney',
 'start Washington at Kearny',
 'start Yerba Buena Center of the Arts (3rd @ Howard)',
 'end 2nd at Folsom',
 'end 2nd at South Park',
 'end 2nd at Townsend',
 'end 5th at Howard',
 'end Adobe on Almaden',
 'end Arena Green / SAP Center',
 'end Beale at Market',
 'end Broadway St at Battery St',
 'end Broadway at Main',
 'end California Ave Caltrain Station',
 'end Castro Street and El Camino Real',
 'end Civic Center BART (7th at Market)',
 'end Clay at Battery',
 'end Commercial at Montgomery',
 'end Cowper at University',
 'end Davis at Jackson',
 'end Embarcadero at Bryant',
 'end Embarcadero at Folsom',
 'end Embarcadero at Sansome',
 'end Embarcadero at Vallejo',
 'end Evelyn Park and Ride',
 'end Franklin at Maple',
 'end Golden Gate at Polk',
 'end Grant Avenue at Columbus Avenue',
 'end Harry Bridges Plaza (Ferry Building)',
 'end Howard at 2nd',
 'end Japantown',
 'end MLK Library',
 'end Market at 10th',
 'end Market at 4th',
 'end Market at Sansome',
 'end Mechanics Plaza (Market at Battery)',
 'end Mezes Park',
 'end Mountain View Caltrain Station',
 'end Mountain View City Hall',
 'end Palo Alto Caltrain Station',
 'end Park at Olive',
 'end Paseo de San Antonio',
 'end Post at Kearney',
 'end Post at Kearny',
 'end Powell Street BART',
 'end Powell at Post (Union Square)',
 'end Redwood City Caltrain Station',
 'end Redwood City Medical Center',
 'end Redwood City Public Library',
 'end Rengstorff Avenue / California Street',
 'end Ryland Park',
 'end SJSU - San Salvador at 9th',
 'end SJSU 4th at San Carlos',
 'end San Antonio Caltrain Station',
 'end San Antonio Shopping Center',
 'end San Francisco Caltrain (Townsend at 4th)',
 'end San Francisco Caltrain 2 (330 Townsend)',
 'end San Francisco City Hall',
 'end San Jose City Hall',
 'end San Jose Civic Center',
 'end San Jose Diridon Caltrain Station',
 'end San Jose Government Center',
 'end San Mateo County Center',
 'end San Pedro Square',
 'end San Salvador at 1st',
 'end Santa Clara County Civic Center',
 'end Santa Clara at Almaden',
 'end South Van Ness at Market',
 'end Spear at Folsom',
 'end St James Park',
 'end Stanford in Redwood City',
 'end Steuart at Market',
 'end Temporary Transbay Terminal (Howard at Beale)',
 'end Townsend at 7th',
 'end University and Emerson',
 'end Washington at Kearney',
 'end Washington at Kearny',
 'end Yerba Buena Center of the Arts (3rd @ Howard)',
 'start_dayOfWeek_id0',
 'start_dayOfWeek_id1',
 'start_dayOfWeek_id2',
 'start_dayOfWeek_id3',
 'start_dayOfWeek_id4',
 'start_dayOfWeek_id5',
 'start_dayOfWeek_id6',
 'subscription_type_Customer',
 'subscription_type_Subscriber',
 'start_year_2013',
 'start_year_2014',
 'start_year_2015',
 'start_month_1',
 'start_month_2',
 'start_month_3',
 'start_month_4',
 'start_month_5',
 'start_month_6',
 'start_month_7',
 'start_month_8',
 'start_month_9',
 'start_month_10',
 'start_month_11',
 'start_month_12',
 'start_quarter_1',
 'start_quarter_2',
 'start_quarter_3',
 'start_quarter_4',
 'start_hour_0',
 'start_hour_1',
 'start_hour_2',
 'start_hour_3',
 'start_hour_4',
 'start_hour_5',
 'start_hour_6',
 'start_hour_7',
 'start_hour_8',
 'start_hour_9',
 'start_hour_10',
 'start_hour_11',
 'start_hour_12',
 'start_hour_13',
 'start_hour_14',
 'start_hour_15',
 'start_hour_16',
 'start_hour_17',
 'start_hour_18',
 'start_hour_19',
 'start_hour_20',
 'start_hour_21',
 'start_hour_22',
 'start_hour_23']]                    
    

y = trainingSet.duration

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 2)

print "X_train shape: ", X_train.shape
print "X_test shape: ", X_test.shape
print "y_train shape: ", y_train.shape
print "y_test shape: ", y_test.shape
print "y shape: ", y.shape



# KERAS SECTION
from keras.models import Sequential
from keras.layers import Dense, Activation

print "creating model..."
model = Sequential()
# add first layer which will receive all 236 cols as input, and will be formed by 32 neurons.
# 
# In this case, we initialize the network weights to a small random number generated from a uniform 
# distribution ('uniform'), in this case between 0 and 0.05 because that is the default uniform weight 
# initialization in Keras.
print "adding layers..."

model.add(Dense(32, input_dim=236, activation="relu"))
model.add(Dense(18, activation="relu"))
model.add(Dense(1))

# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
                      loss='mean_squared_error')




print "fitting model..."
# Fit the model
model.fit(X.as_matrix(), y, validation_split=0.3, epochs=15, batch_size=10)
# model.fit(X_train.as_matrix(), y_train.as_matrix(), epochs=10, batch_size=10)


# print "X_test shape: ", X_test.shape
# print "evaluating model..."
# loss_and_metrics = model.evaluate(X_test.as_matrix(), y_test, batch_size=128, verbose=1)
# print "scores: ", model.evaluate(X_test.as_matrix(), y_test.as_matrix())
# print "model.metrics_names: ", model.metrics_names
# print "scores: ", scores
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
