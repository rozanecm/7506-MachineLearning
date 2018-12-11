
# coding: utf-8

# # TP2: Machine Learning

# ### Imports

# In[1]:

import pandas as pd
from datetime import datetime
import scipy.spatial
from sklearn import preprocessing


# ### Data loading

# In[2]:

stationDF = pd.read_csv('../CSVs/station.csv')
trainingSet = pd.read_csv('../CSVs/trip_train.csv')
testingSet = pd.read_csv('../CSVs/trip_test.csv')


# ## Basic data analysis

# In[3]:

stationDF


# In[ ]:

print "stationDF.shape: ", stationDF.shape


# In[4]:

distancesDF = pd.DataFrame(columns=["start_station_id", "end_station_id", "distance"])


# In[5]:

# scipy.spatial.distance_matrix([scipy.spatial.distance.cityblock(stationDF.lat, stationDF.long), 70], 
#                               [scipy.spatial.distance.cityblock(stationDF.lat, stationDF.long), 70])


# In[ ]:

for station, lat, lon in zip(stationDF.id, stationDF.lat, stationDF.long):
    for station2, lat2, lon2 in zip(stationDF.id, stationDF.lat, stationDF.long):
        distancesDF = distancesDF.append({
            "start_station_id": station,
            "end_station_id": station2,
            "distance": scipy.spatial.distance.cityblock([lat, lon], [lat2, lon2])
        }, ignore_index=True)
        

        


# In[ ]:

distancesDF


# In[ ]:

distancesDF['start_station_id'] = distancesDF.start_station_id.astype(int)
distancesDF['end_station_id'] = distancesDF.end_station_id.astype(int)
distancesDF


# ### Training set

# In[ ]:

trainingSet.head()


# In[ ]:

trainingSet.dtypes


# ### convert to date to datetime

# In[ ]:

trainingSet['start_date'] = pd.to_datetime(trainingSet.start_date)
trainingSet['end_date'] = pd.to_datetime(trainingSet.end_date)


# In[ ]:

testingSet['start_date'] = pd.to_datetime(testingSet.start_date)
testingSet['end_date'] = pd.to_datetime(testingSet.end_date)


# In[ ]:

trainingSet.dtypes


# In[ ]:

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


# In[ ]:

testingSet['start_dayOfWeek'] = testingSet.start_date.dt.dayofweek
testingSet['start_week'] = testingSet.start_date.dt.week
testingSet['start_quarter'] = testingSet.start_date.dt.quarter
testingSet['start_time'] = testingSet.start_date.dt.time
testingSet['start_hour'] = testingSet.start_date.dt.hour
testingSet['start_minute'] = testingSet.start_date.dt.minute
testingSet['start_date'] = testingSet.start_date.dt.date

testingSet['end_dayOfWeek'] = testingSet.end_date.dt.dayofweek
testingSet['end_week'] = testingSet.end_date.dt.week
testingSet['end_quarter'] =testingSet.end_date.dt.quarter
testingSet['end_time'] = testingSet.end_date.dt.time
testingSet['end_hour'] = testingSet.end_date.dt.hour
testingSet['end_minute'] = testingSet.end_date.dt.minute
testingSet['end_date'] = testingSet.end_date.dt.date


# In[ ]:

trainingSet.head()


# In[ ]:

list(trainingSet.columns.values)


# In[ ]:

list(testingSet.columns.values)


# In[ ]:

testingSet = testingSet[['id', 
                           'start_date',
                           'start_dayOfWeek',
                           'start_week',
                           'start_quarter', 
                           'start_time',
                           'start_hour',
                           'start_minute',
                           'start_station_name', 
                           'start_station_id', 
                           'end_date', 
                           'end_dayOfWeek',
                           'end_week',
                           'end_quarter', 
                           'end_time',
                           'end_hour',
                           'end_minute',
                           'end_station_name', 
                           'end_station_id', 
                           'bike_id',
                           'subscription_type',
                           'zip_code']]
                  


# In[ ]:

trainingSet.dtypes


# In[ ]:

testingSet.dtypes


# In[ ]:

trainingSet


# In[ ]:

testingSet


# ## Save processed data

# In[ ]:

trainingSet.to_csv('../CSVs/improved_trip_train.csv', index=False)


# In[ ]:

testingSet.to_csv('../CSVs/improved_trip_test.csv', index=False)


# # ///////////////////////////////////////////////////////////////////////////////////////////////
# # ///////////////////////////////////////////////////////////////////////////////////////////////
# # ///////////////////////////////////////////////////////////////////////////////////////////////

# In[ ]:

stationDF = pd.read_csv('../CSVs/station.csv')
trainingSet = pd.read_csv('../CSVs/improved_trip_train.csv')
testingSet = pd.read_csv('../CSVs/improved_trip_test.csv')


# ## Feature Distancia

# In[ ]:

distancesDF = pd.DataFrame(columns=["start_station_id", "end_station_id", "distance"])
for station, lat, lon in zip(stationDF.id, stationDF.lat, stationDF.long):
    for station2, lat2, lon2 in zip(stationDF.id, stationDF.lat, stationDF.long):
        distancesDF = distancesDF.append({
            "start_station_id": station,
            "end_station_id": station2,
            "distance": scipy.spatial.distance.cityblock([lat, lon], [lat2, lon2])
        }, ignore_index=True)
        
distancesDF['start_station_id'] = distancesDF.start_station_id.astype(int)
distancesDF['end_station_id'] = distancesDF.end_station_id.astype(int)
distancesDF    


# In[ ]:

trainingSet = pd.merge(trainingSet,distancesDF,on =['start_station_id','end_station_id'],how = 'inner')


# In[ ]:

testingSet = pd.merge(testingSet,distancesDF,on =['start_station_id','end_station_id'],how = 'inner')


# ## Feature Historico

# In[ ]:

trainingShort = trainingSet.loc[:,['id','duration','start_station_name','start_station_id','end_station_name','end_station_id']]


# In[ ]:

import math
listaStart = []
listaEnd = []
for i in list(trainingShort.start_station_id.values):
    if i not in listaStart:
        listaStart.append(i)
for i in list(trainingShort.end_station_id.values):
    if i not in listaEnd:
        listaEnd.append(i)
listaHistorico = []
for i in listaStart:
    for j in listaEnd:
        df = trainingShort[(trainingShort['start_station_id'] == i) & (trainingShort['end_station_id'] == j)]
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

trainingShort = pd.merge(trainingShort,dfData,on =['start_station_id','end_station_id'],how = 'inner')
trainingShort = trainingShort [['id', 'historical']]
trainingSet = pd.merge(trainingSet,trainingShort,on =['id'],how = 'inner')


# In[ ]:

trainingSet['historical'] = trainingSet.historical.astype(int)
trainingSet


# ## Agrego datos de TP1

# In[ ]:

trainingSet.loc [:,['id','duration','start_station_id', 'end_station_id','start_time', 'end_time','historical','distance' ]]


# In[ ]:

dfTrip = pd.read_csv('../CSVs/trip.csv')
# dfTrip = dfTrip.loc[:,['id','duration']]
# dfTrip = dfTrip.rename(columns={'duration':'durationPosta'})
# dfScore = pd.merge(testingSet,dfTrip,on =['id'],how = 'inner')
# dfScore


# In[ ]:

dfTrip.corr()['duration']


# In[ ]:

dfTrip.loc[dfTrip.id == 192809,:]


# In[ ]:

trainingSet


# In[ ]:

# GLORIOSO DF DEL TP1
#(0 = Monday, 1 = Tuesday...)
dfSF_Bay = pd.read_csv('../CSVs/dfSF_Bay.csv')
dfSF_Bay


# In[ ]:

dfSF_Bay.corr()['viajes']


# In[ ]:

dfSF_Bay.loc[dfSF_Bay.month == 2,:].loc[dfSF_Bay.year == 2014,:].loc[dfSF_Bay.weekday == 5,:]


# In[ ]:

dfSF_Bay.dtypes


# In[ ]:

#Elijo las mejores variables en funcion del TP1
dfSF_Bay = dfSF_Bay.loc [:,['viajes','max_temperature_c','min_temperature_c','max_humidity','max_sea_level_pressure_cm','precipitation_cm','Fog','Normal','Rain','business_day','holiday','year','month','weekday','date']]


# In[ ]:

dfSF_Bay.dtypes


# In[ ]:

trainingSet['year'] = pd.to_datetime(trainingSet['start_date']).dt.year
trainingSet['month'] = pd.to_datetime(trainingSet['start_date']).dt.month
trainingSet['weekday'] = pd.to_datetime(trainingSet['start_date']).dt.weekday
trainingSet


# In[ ]:

trainingSet = pd.merge(trainingSet,dfSF_Bay,left_on ='start_date',right_on='date',how = 'inner')


# In[ ]:

trainingSet.dtypes


# In[ ]:

trainingSet.drop(['year_x','month_x','weekday_x'],1,inplace=True)


# In[ ]:

trainingSet.drop(['date'],1,inplace=True)


# In[ ]:

trainingSet = trainingSet.rename(columns={'year_y':'year','month_y':'month','weekday_y': 'weekday'})


# In[ ]:

trainingSet.dtypes


# ## Save processed data

# In[ ]:

trainingSet.to_csv('../CSVs/improved_trip_train.csv', index=False)


# In[ ]:

testingSet.to_csv('../CSVs/improved_trip_test.csv', index=False)


# # //////////////////////////////////////////////////////////////////////////////

# # //////////////////////////////////////////////////////////////////////////////

# # //////////////////////////////////////////////////////////////////////////////

# # Discretizacion y Normalizacion

# ## Discretizacion

# In[ ]:

trainingSet = pd.read_csv('../CSVs/improved_trip_train.csv')
testingSet = pd.read_csv('../CSVs/improved_trip_test.csv')


# In[ ]:

trainingSet.columns.values


# In[ ]:

def crearLista (listadoCompleto):
    listaReducida = []
    for i in listadoCompleto:
        if i not in listaReducida:
            listaReducida.append(i)
    listaReducida.sort()
    return listaReducida


# In[ ]:

def discretizar(columna,listaReducida,nombre):
    v = list(range(len(columna)))
    listaCompleta = list(columna)
    for i in listaReducida:
        for j in range(len(listaCompleta)):
            if(listaCompleta[j] == i):
                v[j] = 1
            else:
                v[j] = 0
        trainingSet[nombre+str(i)] = v


# In[ ]:

listaStartName = crearLista(trainingSet.start_station_name)
listaEndName = crearLista(trainingSet.end_station_name)


# In[ ]:

discretizar(trainingSet.start_station_name,listaStartName,'start ')


# In[ ]:

trainingSet


# In[ ]:

trainingSet.loc[:,['start_station_name','start Harry Bridges Plaza (Ferry Building)']]


# In[ ]:

discretizar(trainingSet.end_station_name,listaEndName,'end ')


# In[ ]:

listaStartWeekDay = crearLista(trainingSet.start_dayOfWeek)


# In[ ]:

discretizar(trainingSet.start_dayOfWeek,listaStartWeekDay,'start_dayOfWeek_id')


# In[ ]:

listaEndWeekDay = crearLista(trainingSet.end_dayOfWeek)


# In[ ]:

discretizar(trainingSet.end_dayOfWeek,listaEndWeekDay,'end_dayOfWeek_id')


# In[ ]:

trainingSet.loc[:,['end_dayOfWeek_id3','end_dayOfWeek']]


# In[ ]:

listaSubscriptionType = crearLista(trainingSet.subscription_type)


# In[ ]:

discretizar(trainingSet.subscription_type,listaSubscriptionType,'subscription_type_')


# In[ ]:

trainingSet


# In[ ]:

trainingSet[trainingSet.zip_code.str.isnumeric() == False].loc[:,['zip_code']]


# In[ ]:

listaStartYear = crearLista(trainingSet.start_year)


# In[ ]:

discretizar(trainingSet.start_year,listaStartYear,'start_year_')


# In[ ]:

listaEndYear = crearLista(trainingSet.end_year)


# In[ ]:

discretizar(trainingSet.end_year,listaEndYear,'end_year_')


# In[ ]:

listaStartMonth = crearLista(trainingSet.start_month)


# In[ ]:

discretizar(trainingSet.start_month,listaStartMonth,'start_month_')


# In[ ]:

listaEndMonth = crearLista(trainingSet.end_month)


# In[ ]:

discretizar(trainingSet.end_month,listaEndMonth,'end_month_')


# In[ ]:

listaStartDay = crearLista(trainingSet.start_day)


# In[ ]:

discretizar(trainingSet.start_day,listaStartDay,'start_day_')


# In[ ]:

listaEndDay = crearLista(trainingSet.end_day)


# In[ ]:

discretizar(trainingSet.end_day,listaEndDay,'end_day_')


# In[ ]:

listaStartQuarter = crearLista(trainingSet.start_quarter)


# In[ ]:

discretizar(trainingSet.start_quarter,listaStartQuarter,'start_quarter_')


# In[ ]:

listaEndQuarter = crearLista(trainingSet.end_quarter)


# In[ ]:

discretizar(trainingSet.end_quarter,listaEndQuarter,'end_quarter_')


# In[ ]:

listaStartHour = crearLista(trainingSet.start_hour)


# In[ ]:

discretizar(trainingSet.start_hour,listaStartHour,'start_hour_')


# In[ ]:

listaEndHour = crearLista(trainingSet.end_hour)


# In[ ]:

discretizar(trainingSet.end_hour,listaEndHour,'end_hour')


# In[ ]:

trainingSet


# In[ ]:

trainingSet.columns.values


# In[ ]:

#A veces tarda a veces no
trainingSet = trainingSet.drop(labels = ['start_date','start_dayOfWeek','start_quarter','start_time','start_station_name','end_date',
                        'end_dayOfWeek','end_quarter','end_time','end_station_name','subscription_type','year','month',
                         'weekday','start_hour','end_hour'],axis = 1)


# In[ ]:

trainingSet.columns.values


# In[ ]:

trainingSet


# ## Save processed data

# In[ ]:

trainingSet.to_csv('../CSVs/improved_trip_train.csv', index=False)


# ## Normalizacion

# In[ ]:

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

trainingSet.loc[:,['duration','max_temperature_c','min_temperature_c','max_humidity','max_sea_level_pressure_cm','precipitation_cm']]


# # ///////////////////////////////////////////////////////////////////////////////////////////////
# # ///////////////////////////////////////////////////////////////////////////////////////////////
# # ///////////////////////////////////////////////////////////////////////////////////////////////
# # CORRER A PARTIR DE ACA

# # Filtrado de OutLiers

# In[ ]:

trainingSet = pd.read_csv('../CSVs/improved_trip_train.csv')


# In[ ]:

trainingSet


# In[ ]:

trainingSet.columns.values


# In[ ]:

trainingSet.duration.describe()


# In[ ]:

trainingSet.loc[:,['duration','start_year','start_month','start_day','end_year','end_month','end_day']].sort_values(by='duration',ascending = False).head(200)


# In[ ]:

duration = list(trainingSet.duration.sort_values(ascending = False).head(100))


# In[ ]:

import plotly 
import numpy as np
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()
plotly.tools.set_credentials_file(username='AARTURI', api_key='qiQqOxKJXDlziMFzaB8j')
plotly.offline.init_notebook_mode(connected=True)


# In[ ]:

trace0 = go.Scatter(
    y = duration,
    mode = 'markers',
    name = 'h=0.1'
)


data = [trace0]
fig = Figure(data=data)
plotly.offline.iplot(fig, filename='styled-scatter')


# In[ ]:

trainingSet = trainingSet.loc[trainingSet.duration < 1000000,:]


# In[ ]:

trainingSet


# In[ ]:

trainingSet[trainingSet['duration'] > 1000000]


# ## Save processed data

# In[ ]:

trainingSet.to_csv('../CSVs/improved_trip_train.csv', index=False)

