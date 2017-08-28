#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 20:04:30 2017

@author: canf
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pylab as pl

from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from datetime import tzinfo, timedelta, datetime
from numpy import fft
from statsmodels.tsa.seasonal import seasonal_decompose

#%matplotlib inline

rcParams['figure.figsize'] = 12, 5
sns.set_style(style='white')

train = pd.read_csv('./input-data/sf_train_fixed.csv', parse_dates=['Dates'])
train['DayOfYear'] = train['Dates'].map(lambda x: x.strftime("%m-%d"))

df_global = train[['Category','DayOfYear']].groupby(['DayOfYear']).count()
df_global.plot(y='Category', label='N\u00famero de eventos', figsize=(6,4)) 
plt.title("Patrones criminales")
plt.ylabel('N\u00famero de cr\u00edmenes')
plt.xlabel('D\u00eda del a\u00f1o')
plt.grid(True)
plt.savefig('./output-data/Distribution_of_Crimes_by_Day_Year.png')

plt.show()
plt.close()

#train['Daily'] = train['Dates'].map(lambda x: x.strftime("%Y-%m-%d"))
#df_daily = train[['Category','Daily']].groupby(['Daily']).count()
#df_daily.plot(y='Category', label='N\u00famero de eventos', figsize=(6,4))
#plt.xticks(rotation=90)
#plt.grid(True)
#
#plt.show()
#plt.close()

################################################################################

#turn strings into dates
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
train = pd.read_csv('./input-data/sf_train_fixed.csv', parse_dates = ['Dates'], index_col = 'Dates', date_parser = dateparse)
cats = list(set(train.Category))

#print(train.head(5))
#print(train.index)

df_daily = train.groupby(train.index.date).count()
df_daily = df_daily.loc[:, 'Y']
df_daily.plot(y='Y', label='N\u00famero de eventos', figsize=(6,4))
plt.xticks(rotation=90)
plt.grid(True)

plt.show()
plt.close()

#df_day = pd.DataFrame({'Dates':df_daily.index, 'Y':df_daily.values})
#df_day.reset_index(inplace=True)
#df_day['Dates'] = pd.to_datetime(df_day['Dates'])
#df_day = df_day.set_index('Dates')
#
#print(df_day)
#print(df_day.Y)
#decomposition = seasonal_decompose(df_day.Y, freq='D')
#input("Presiona una tecla para continuar...(II)")


def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=30).mean()
    rolstd = pd.Series(timeseries).rolling(window=30).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    plt.show()
    plt.close()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(df_daily)

df_daily_log = np.log(df_daily)
moving_avg_log = pd.Series(df_daily_log).rolling(window=30).mean()
df_daily_log.plot(y='Y', label='N\u00famero de eventos', figsize=(6,4))
moving_avg_log.plot(color = 'red')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()
plt.close()

df_daily_log_mv_avg_diff = df_daily_log - moving_avg_log
df_daily_log_mv_avg_diff.dropna(inplace=True)

test_stationarity(df_daily_log_mv_avg_diff)

def fourierExtrapolation(x, n_predict):
    n = x.size
    n_harm = 1000                   # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n)              # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))
 
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t

x = np.array(df_daily, dtype=pd.Series)
n_predict = 100
extrapolation = fourierExtrapolation(x, n_predict)
pl.plot(np.arange(0, x.size), x, 'b', label = 'x', linewidth = 3)
pl.plot(np.arange(0, extrapolation.size), extrapolation, 'r', label = 'extrapolation')
pl.legend()
pl.show()
plt.close()

test_stationarity(extrapolation)

residual = x - extrapolation[0:(x.size)]
pl.plot(np.arange(0, residual.size), residual, 'g', label= 'Series residuals')
pl.legend()
pl.show()
plt.close()

test_stationarity(residual)

################################################################################
#
# Parte X: Descomposicion de la serie
#
################################################################################



print("")
print("************************")
print("*** Fin del programa ***")
print("************************")








