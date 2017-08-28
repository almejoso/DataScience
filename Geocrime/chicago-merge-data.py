#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 16:38:27 2017

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

#train = pd.read_csv('./input-data/chicago_train.csv')
#print(train.head(5))

#train['DayOfYear'] = train['Dates'].map(lambda x: x.strftime("%m-%d"))
#
#df_global = train[['Category','DayOfYear']].groupby(['DayOfYear']).count()
#df_global.plot(y='Category', label='N\u00famero de eventos', figsize=(6,4)) 
#plt.title("Patrones criminales")
#plt.ylabel('N\u00famero de cr\u00edmenes')
#plt.xlabel('D\u00eda del a\u00f1o')
#plt.grid(True)
#plt.savefig('./output-data/Distribution_of_Crimes_by_Day_Year.png')
#
#plt.show()
#plt.close()

#train['Daily'] = train['Dates'].map(lambda x: x.strftime("%Y-%m-%d"))
#df_daily = train[['Category','Daily']].groupby(['Daily']).count()
#df_daily.plot(y='Category', label='N\u00famero de eventos', figsize=(6,4))
#plt.xticks(rotation=90)
#plt.grid(True)
#
#plt.show()
#plt.close()

#input("Presiona una tecla para continuar...(I)")

################################################################################

#turn strings into dates
dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y %I:%M:%S %p')
train = pd.read_csv('./input-data/chicago_train.csv', parse_dates = ['DATE'], index_col = 'DATE', date_parser = dateparse)
test = pd.read_csv('./input-data/chicago_test.csv', parse_dates = ['DATE'], index_col = 'DATE', date_parser = dateparse)
cats = list(set(train.CATEGORY))

df_daily = train.groupby(train.index.date).count()
df_daily = df_daily.loc[:, 'BLOCK']

df_daily_test = test.groupby(test.index.date).count()
df_daily_test = df_daily_test.loc[:, 'BLOCK']

def fourierExtrapolation(x, n_predict):
    n = x.size
    n_harm = 185                    # number of harmonics in model
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
n_predict = 15
extrapolation = fourierExtrapolation(x, n_predict)

newindex = df_daily.index.union(df_daily_test.index)
df_daily = df_daily.reindex(newindex)
df_daily_test = df_daily_test.reindex(newindex)

x = np.array(df_daily, dtype=pd.Series)
y = np.array(df_daily_test, dtype=pd.Series)

pl.plot(np.arange(0, x.size), x, 'b', label = 'x', linewidth = 3)
pl.plot(np.arange(0, x.size), y, 'g', label = 'y', linewidth = 2)
pl.plot(np.arange(0, extrapolation.size), extrapolation, 'r', label = 'extrapolation')
pl.legend()
pl.show()
plt.close()