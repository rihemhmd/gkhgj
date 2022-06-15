# evaluate an ARIMA model using a walk-forward validation
from math import sqrt
from random import randint

import warnings
from tkinter import *
import time, json
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)


# lire le dataset

series = pd.read_csv('TBdata.csv',header=0)
#-- imprimer les 5 premieres lignes                      
print(series.head(5))

#-- imprimer les noms de champs
print(series.info())

#-- convertir le mois en type date
series['Mois'] = pd.to_datetime(series['Mois'], format="%Y-%m")
#--print(series.info())
#-- imprimer les 5 premieres lignes                      
 
series.set_index("Mois", inplace = True)
plt.plot(series,label="Nbre de cas par mois")
plt.legend() 
plt.show()

# split into train and test sets
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:]
history = [x for x in train]
predictions = list()
testMonth = list()
predictionsBest=list()

BFC=100
param = (0,1,2)
cpt=1
while cpt<3: 
    # walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=param)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        testMonth.append(t)
        print('prévus=%f, observés=%f' % (yhat, obs))

     # evaluate forecasts
    rmse = sqrt(mean_squared_error(test , predictions))
    print('Test RMSE: %.3f' % rmse)
    if rmse < BFC :
        try:
            BFC = rmse
        except:
            continue
        paramBest = param
        predictionsBest = predictions.copy()
         

    predictions = list()
    testMonth  = list()
    
    cpt = cpt+1
    param = (randint(1,2), randint(0,1), randint(0,2))
    print('iteration:',cpt)
    print('param:',param)
   

# plot forecasts against actual outcomes

  
plt.plot(test,label="nbr de cas observés ", color="black")
plt.plot(predictionsBest,testMonth,label="nbr de cas prévus ", color='red')
plt.legend()
plt.show()

