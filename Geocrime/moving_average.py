#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 18:58:53 2017

@author: canf
"""

import pandas as pd
import numpy as np
#import scipy
#import math
# Importar librerías gráficas
import seaborn as sns
import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
# Importar librerías de formato
from datetime import tzinfo, timedelta, datetime
# Importar modelos de aprendizaje
#from sklearn.model_selection import cross_val_score
#from sklearn.neighbors import RadiusNeighborsClassifier
#from sklearn.neighbors import KNeighborsClassifier

sns.set_style(style='white')

drive_in = './input-data/'
drive_out = './output-data/'
train = pd.read_csv(drive_in+'sf_train_fixed.csv')

#get a unique list of categories
cats = list(set(train.Category))

#turn strings into dates
dates = []
datesAll = np.array([datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
            for date in train.Dates])

#set up pandas
startDate = (np.min(datesAll)).date()
endDate = (np.max(datesAll)).date()
alldates_month = pd.bdate_range(startDate, endDate, freq="m")
alldates_daily = pd.bdate_range(startDate, endDate, freq="d")

print(alldates_month)
print(alldates_daily)