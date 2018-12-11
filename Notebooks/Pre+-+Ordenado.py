
# coding: utf-8

# # TP2: Machine Learning

# ### Imports

# In[ ]:

import pandas as pd
from datetime import datetime
import scipy.spatial
from sklearn import preprocessing


# ### Data loading

# In[ ]:

print "*****Data loading*****"
print "loading station csv"
stationDF = pd.read_csv('../CSVs/station.csv')
print "loading trip train csv"
trainingSet = pd.read_csv('../CSVs/trip_train.csv')
print "loading trip test csv"
testingSet = pd.read_csv('../CSVs/trip_test.csv')

# GLORIOSO DF DEL TP1
# (0 = Monday, 1 = Tuesday...)
print "loading dfSF_Bay csv"
dfSF_Bay = pd.read_csv('../CSVs/dfSF_Bay.csv')


# ## Basic data analysis

# In[ ]:

print "stationDF.shape: ", stationDF.shape
print "trainingSet.shape: ", trainingSet.shape
print "testingSet.shape: ", testingSet.shape
print "dfSF_Bay.shape: ", dfSF_Bay.shape


# ### Distancias entre estaciones

# In[ ]:

print "*****Working on station distances*****"
# Create new temporary dataframe with distances
distancesDF = pd.DataFrame(columns=["start_station_id", "end_station_id", "distance"])

# Calculate distances between stations
for station, lat, lon in zip(stationDF.id, stationDF.lat, stationDF.long):
    for station2, lat2, lon2 in zip(stationDF.id, stationDF.lat, stationDF.long):
        distancesDF = distancesDF.append({
            "start_station_id": station,
            "end_station_id": station2,
            "distance": scipy.spatial.distance.cityblock([lat, lon], [lat2, lon2])
        }, ignore_index=True)

distancesDF['start_station_id'] = distancesDF.start_station_id.astype(int)
distancesDF['end_station_id'] = distancesDF.end_station_id.astype(int)

# Merge this new data to training and testing sets
trainingSet = pd.merge(trainingSet,distancesDF,on =['start_station_id','end_station_id'],how = 'inner')
testingSet = pd.merge(testingSet,distancesDF,on =['start_station_id','end_station_id'],how = 'inner')

# delete auxiliary distances df
del distancesDF


# ### Process date & time data

# In[ ]:

print "*****Converting necessary data to dateTime*****"
# Convert necessary data to dateTime
dfSF_Bay['date'] = pd.to_datetime(dfSF_Bay.date)

trainingSet['start_date'] = pd.to_datetime(trainingSet.start_date)
trainingSet['end_date'] = pd.to_datetime(trainingSet.end_date)

testingSet['start_date'] = pd.to_datetime(testingSet.start_date)
testingSet['end_date'] = pd.to_datetime(testingSet.end_date)

# Create new features related to date & time based on the unique 'date' feature
# Work with training set
trainingSet['start_dayOfWeek'] = trainingSet.start_date.dt.dayofweek
trainingSet['start_week'] = trainingSet.start_date.dt.week
trainingSet['start_quarter'] = trainingSet.start_date.dt.quarter
trainingSet['start_time'] = trainingSet.start_date.dt.time
trainingSet['start_hour'] = trainingSet.start_date.dt.hour
trainingSet['start_minute'] = trainingSet.start_date.dt.minute
trainingSet['start_year'] = trainingSet.start_date.dt.year
trainingSet['start_month'] = trainingSet.start_date.dt.month
trainingSet['start_day'] = trainingSet.start_date.dt.day
trainingSet['start_date'] = trainingSet.start_date.dt.date

trainingSet['end_dayOfWeek'] = trainingSet.end_date.dt.dayofweek
trainingSet['end_week'] = trainingSet.end_date.dt.week
trainingSet['end_quarter'] = trainingSet.end_date.dt.quarter
trainingSet['end_time'] = trainingSet.end_date.dt.time
trainingSet['end_hour'] = trainingSet.end_date.dt.hour
trainingSet['end_minute'] = trainingSet.end_date.dt.minute
trainingSet['end_year'] = trainingSet.end_date.dt.year
trainingSet['end_month'] = trainingSet.end_date.dt.month
trainingSet['end_day'] = trainingSet.end_date.dt.day
trainingSet['end_date'] = trainingSet.end_date.dt.date

trainingSet['year'] = pd.to_datetime(trainingSet['start_date']).dt.year
trainingSet['month'] = pd.to_datetime(trainingSet['start_date']).dt.month
trainingSet['weekday'] = pd.to_datetime(trainingSet['start_date']).dt.weekday

# Work with testing set
testingSet['start_dayOfWeek'] = testingSet.start_date.dt.dayofweek
testingSet['start_week'] = testingSet.start_date.dt.week
testingSet['start_quarter'] = testingSet.start_date.dt.quarter
testingSet['start_time'] = testingSet.start_date.dt.time
testingSet['start_hour'] = testingSet.start_date.dt.hour
testingSet['start_minute'] = testingSet.start_date.dt.minute
testingSet['start_year'] = testingSet.start_date.dt.year
testingSet['start_month'] = testingSet.start_date.dt.month
testingSet['start_day'] = testingSet.start_date.dt.day
testingSet['start_date'] = testingSet.start_date.dt.date

testingSet['end_dayOfWeek'] = testingSet.end_date.dt.dayofweek
testingSet['end_week'] = testingSet.end_date.dt.week
testingSet['end_quarter'] =testingSet.end_date.dt.quarter
testingSet['end_time'] = testingSet.end_date.dt.time
testingSet['end_hour'] = testingSet.end_date.dt.hour
testingSet['end_minute'] = testingSet.end_date.dt.minute
testingSet['end_year'] = testingSet.end_date.dt.year
testingSet['end_month'] = testingSet.end_date.dt.month
testingSet['end_day'] = testingSet.end_date.dt.day
testingSet['end_date'] = testingSet.end_date.dt.date

testingSet['year'] = pd.to_datetime(testingSet['start_date']).dt.year
testingSet['month'] = pd.to_datetime(testingSet['start_date']).dt.month
testingSet['weekday'] = pd.to_datetime(testingSet['start_date']).dt.weekday


# In[ ]:

print "trainingSet cols values", list(trainingSet.columns.values)


# In[ ]:

print "testingSet cols values", list(testingSet.columns.values)


# ## Feature Historico

# In[ ]:

print "*****Working on historic feature*****"
print "***Calculating historic feature***"
import math
listaStart = []
listaEnd = []
for i in list(trainingSet.start_station_id.values):
    if i not in listaStart:
        listaStart.append(i)
for i in list(trainingSet.end_station_id.values):
    if i not in listaEnd:
        listaEnd.append(i)
listaHistorico = []
for i in listaStart:
    for j in listaEnd:
        df = trainingSet[(trainingSet['start_station_id'] == i) & (trainingSet['end_station_id'] == j)]
        historico = df.duration.mean()
        if (not(math.isnan(historico))):
            listaHistorico.append([i,j,historico])
        
listaHistorico


# In[ ]:

starStationId = []
endStationId = []
historical = []
for x in listaHistorico:
    starStationId.append(x[0])
    endStationId.append(x[1])
    historical.append(x[2])

data = {
    'start_station_id' : starStationId,
    'end_station_id' : endStationId,
    'historical' : historical,
}

dfData = pd.DataFrame(data,columns = ['start_station_id','end_station_id','historical'])
dfData


# In[ ]:

print "**Merging historic feature**"
# Merge this new data to training and testing dfs
# Training
trainingSet = pd.merge(trainingSet,dfData,on=['start_station_id', 'end_station_id'],how = 'inner') 

trainingSet['historical'] = trainingSet.historical.astype(int)

# Testing
testingSet = pd.merge(testingSet, dfData, on=['start_station_id', 'end_station_id'], how='inner')

testingSet['historical'] = testingSet.historical.astype(int)

# delete auxiliar dataframe
del dfData


# In[ ]:

print "trainingSet.shape: ", trainingSet.shape
print "testingSet.shape: ", testingSet.shape


# The difference in the shapes is due to the duration feature used in the training set, which was used to calculate the historical feature.

# ### Trabajamos con dfSF_Bay

# In[ ]:

# Convert necessary data to dateTime
dfSF_Bay['date'] = pd.to_datetime(dfSF_Bay.date)

trainingSet['start_date'] = pd.to_datetime(trainingSet.start_date)
trainingSet['end_date'] = pd.to_datetime(trainingSet.end_date)

testingSet['start_date'] = pd.to_datetime(testingSet.start_date)
testingSet['end_date'] = pd.to_datetime(testingSet.end_date)


# In[ ]:

print "***Merging dfSF_Bay data***"
# Merge trainingSet with new data

testingSet = pd.merge(testingSet,dfSF_Bay,left_on ='start_date',right_on='date',how = 'inner')
trainingSet = pd.merge(trainingSet,dfSF_Bay,left_on ='start_date',right_on='date',how = 'inner')


# In[ ]:

testingSet


# In[ ]:

trainingSet


# # Discretizacion y Normalizacion

# ## Discretizacion

# In[ ]:

(trainingSet.dtypes)


# In[ ]:

list(testingSet.columns.values)


# In[ ]:

print "*****Discretizacion y Normalizacion*****"
print "*****Discretizacion*****"


# In[ ]:

def crearLista (listadoCompleto):
    listaReducida = []
    for i in listadoCompleto:
        if i not in listaReducida:
            listaReducida.append(i)
    listaReducida.sort()
    return listaReducida


# In[ ]:

def discretizar(columna,nombre, df):
    listaReducida = crearLista(columna)
    v = list(range(len(columna)))
    listaCompleta = list(columna)
    for i in listaReducida:
        for j in range(len(listaCompleta)):
            if(listaCompleta[j] == i):
                v[j] = 1
            else:
                v[j] = 0
        df[nombre+str(i)] = v


# In[ ]:

print "Discretizando start_station_name..."
discretizar(trainingSet.start_station_name,'start ', trainingSet)
discretizar(testingSet.start_station_name,'start ', testingSet)

print "Discretizando end_station_name..."
discretizar(trainingSet.end_station_name,'end ', trainingSet)
discretizar(testingSet.end_station_name,'end ', testingSet)

print "Discretizando start_dayOfWeek..."
discretizar(trainingSet.start_dayOfWeek,'start_dayOfWeek_id', trainingSet)
discretizar(testingSet.start_dayOfWeek,'start_dayOfWeek_id', testingSet)

print "Discretizando end_dayOfWeek..."
discretizar(trainingSet.end_dayOfWeek,'end_dayOfWeek_id', trainingSet)
discretizar(testingSet.end_dayOfWeek,'end_dayOfWeek_id', testingSet)

print "Discretizando subscription_type_..."
discretizar(trainingSet.subscription_type,'subscription_type_', trainingSet)
discretizar(testingSet.subscription_type,'subscription_type_', testingSet)

print "Discretizando start_year..."
discretizar(trainingSet.start_year,'start_year_', trainingSet)
discretizar(testingSet.start_year,'start_year_', testingSet)

print "Discretizando end_year_..."
discretizar(trainingSet.end_year,'end_year_', trainingSet)
discretizar(testingSet.end_year,'end_year_', testingSet)

print "Discretizando start_month..."
discretizar(trainingSet.start_month,'start_month_', trainingSet)
discretizar(testingSet.start_month,'start_month_', testingSet)

print "Discretizando end_month..."
discretizar(trainingSet.end_month,'end_month_', trainingSet)
discretizar(testingSet.end_month,'end_month_', testingSet)

print "Discretizando start_day..."
discretizar(trainingSet.start_day,'start_day_', trainingSet)
discretizar(testingSet.start_day,'start_day_', testingSet)

print "Discretizando end_day..."
discretizar(trainingSet.end_day,'end_day_', trainingSet)
discretizar(testingSet.end_day,'end_day_', testingSet)

print "Discretizando start_quarter..."
discretizar(trainingSet.start_quarter,'start_quarter_', trainingSet)
discretizar(testingSet.start_quarter,'start_quarter_', testingSet)

print "Discretizando end_quarter..."
discretizar(trainingSet.end_quarter,'end_quarter_', trainingSet)
discretizar(testingSet.end_quarter,'end_quarter_', testingSet)

print "Discretizando start_hour..."
discretizar(trainingSet.start_hour,'start_hour_', trainingSet)
discretizar(testingSet.start_hour,'start_hour_', testingSet)

print "Discretizando end_hour..."
discretizar(trainingSet.end_hour,'end_hour', trainingSet)
discretizar(testingSet.end_hour,'end_hour', testingSet)


# In[ ]:

print "Dropping trash columns..."
trainingSet = trainingSet.drop(labels = ['start_date', 
                                         'end_station_name',
                                         'start_station_name',
                                         'end_date',
                                         'subscription_type',
                                         'zip_code',
                                         'start_time',
                                         'end_time',
                                         'start_dayOfWeek',
                                         'end_dayOfWeek',
                                         'start_year',
                                         'end_year',
                                         'start_month',
                                         'end_month',
                                         'start_day',
                                         'end_day',
                                         'start_quarter',
                                         'end_quarter',
                                         'start_hour',
                                         'end_hour'
                                        ],axis = 1)

testingSet = testingSet.drop(labels = ['start_date', 
                                         'end_station_name',
                                         'start_station_name',
                                         'end_date',
                                         'subscription_type',
                                         'zip_code',
                                         'start_time',
                                         'end_time',
                                         'start_dayOfWeek',
                                         'end_dayOfWeek',
                                         'start_year',
                                         'end_year',
                                         'start_month',
                                         'end_month',
                                         'start_day',
                                         'end_day',
                                         'start_quarter',
                                         'end_quarter',
                                         'start_hour',
                                         'end_hour'
                                        ],axis = 1)


# In[ ]:

print "trainingSet.shape: ", trainingSet.shape
print "testingSet.shape: ", testingSet.shape


# ## Normalizacion

# In[ ]:

print "*****Normalizacion*****"


# In[ ]:

print "Saving temp csvs..."
trainingSet.to_csv('../CSVs/tempTraining.csv')
testingSet.to_csv('../CSVs/tempTesting.csv')


# In[ ]:

# trainingSet = pd.read_csv('../CSVs/tempTraining.csv')
# testingSet = pd.read_csv('../CSVs/tempTesting.csv')


# In[ ]:

print "Normalizando data..."
durationNormalize = preprocessing.normalize(trainingSet.duration)
trainingSet['duration'] = durationNormalize[0]
maxTemperatureNormalize = preprocessing.normalize(trainingSet.max_temperature_c)
trainingSet['max_temperature_c'] = maxTemperatureNormalize[0]
minTemperatureNormalize= preprocessing.normalize(trainingSet.min_temperature_c)
trainingSet['min_temperature_c'] = minTemperatureNormalize[0]
maxHumidityNormalize = preprocessing.normalize(trainingSet.max_humidity)
trainingSet['max_humidity'] = maxHumidityNormalize[0]
maxSeaLevelPressureNormalize = preprocessing.normalize(trainingSet.max_sea_level_pressure_cm)
trainingSet['max_sea_level_pressure_cm'] = maxSeaLevelPressureNormalize[0]
precipitationNormalize = preprocessing.normalize(trainingSet.precipitation_cm)
trainingSet['precipitation_cm'] = precipitationNormalize[0]


# In[ ]:

print "filtrando por duracion..."
trainingSet = trainingSet.loc[trainingSet.duration < 1000000,:]


# In[ ]:

trainingSet


# In[ ]:

print "Saving temp csvs..."
trainingSet.to_csv('../CSVs/tempTraining.csv')
testingSet.to_csv('../CSVs/tempTesting.csv')


# In[ ]:

print "Saving to new csvs..."
print "Saving trainingSet to ../CSVs/finalTraining.csv..."
trainingSet.to_csv('../CSVs/finalTraining.csv')
print "Saving testingSet to ../CSVs/finalTesting.csv..."
testingSet.to_csv('../CSVs/finalTesting.csv')

