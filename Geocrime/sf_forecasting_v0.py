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
from datetime import tzinfo, timedelta, datetime
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
sns.set_style(style='white')

drive_in = './input-data/'
drive_out = './output-data/'
train = pd.read_csv(drive_in+'sf_train_fixed.csv')

#get a unique list of categories
cats = list(set(train.Category))

#map data
mapdata = np.loadtxt(drive_in+"sf_map_copyright_openstreetmap.txt")
lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]

Crime_Categories = list(train.loc[:,"Category"].unique())
number_of_crimes = train.Category.value_counts()
relative_crime = number_of_crimes / sum(number_of_crimes)
relative_crime = relative_crime.cumsum()
SubCrime_Categories = list(relative_crime[0:12].index)

#turn strings into dates
dates = []
datesAll = np.array([datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
            for date in train.Dates])

#set up pandas
startDate = (np.min(datesAll)).date()
endDate = (np.max(datesAll)).date()
alldates = pd.bdate_range(startDate, endDate, freq="m")
dayDF = pd.DataFrame(np.NAN, index=alldates, columns=['x'])

for cat in cats:
    saveFile = cat+'.png'
    if cat in SubCrime_Categories:
        fig = plt.figure(figsize = (11.69, 8.27))
        plt.title(cat)
        
################################################################################
        #ploteo
        ax = plt.subplot(2,2,1)
        ax.imshow(mapdata, cmap=plt.get_cmap('gray'), 
              extent=lon_lat_box)
    
        Xcoord = (train[train.Category==cat].X).values
        Ycoord = (train[train.Category==cat].Y).values
        dates = datesAll[np.where(train.Category==cat)]
        Z = np.ones([len(Xcoord),1])
            
        #dataframe
        df = pd.DataFrame([ [ Z[row][0],Xcoord[row],Ycoord[row]  ] for row in range(len(Z))],
               index=[dates[row] for row in range(len(dates))],
               columns=['z','xcoord','ycoord']) 
         
        df2 = df.resample('m').sum()
        
        #kernel density plot by year
        kdeMaxX = []
        kdeMaxY = []
        for yLoop in range(2003,2015):
            allData2 = df[(df.index.year == yLoop)]
                   
        sns.kdeplot(np.array(allData2['xcoord']), np.array(allData2['ycoord']),shade=True, cut=10, clip=clipsize,alpha=0.5)

################################################################################  
        #create uniform time series
        allTimes = dayDF \
        .join(df2) \
        .drop('x', axis=1) \
        .fillna(0)
    
        #movAv = pd.rolling_mean(allTimes['z'],window=12,min_periods=1)
        movAv = pd.Series(allTimes['z']).rolling(window=12,min_periods=1).mean()
    
        #time series plot with 12 month moving average
        ax = plt.subplot(2,1,2)
        plt.plot(allTimes.index,allTimes['z'])
        plt.plot(allTimes.index,movAv,'r')
        
        #heatmap to look how data varies by day of week
        ax = plt.subplot(2,2,2)
        heatData = []
        yLoopCount=0
        weekName = ['mon','tue','wed','thu','fri','sat','sun']
        yearName = ['2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014']
        for yLoop in range(2003,2015):
            heatData.append([])
            for dLoop in range(7):
                allData3 = df[(df.index.year == yLoop) & (df.index.weekday == dLoop)]
                heatData[yLoopCount].append(sum(allData3['z'].values))
            yLoopCount+=1
        
        #normlise
        heatData = np.array(heatData)/np.max(np.array(heatData))
        sns.heatmap(heatData, annot=True,xticklabels=weekName,yticklabels=yearName, vmin=0, vmax=1);

        plt.title(cat)
        plt.savefig(drive_out+saveFile)
        #print("La categoria {0} es relevante".format(cat))

    else:
        print("La categoria {0} no es principal".format(cat))
        #input("Presiona una tecla para continuar...(IV)")        

#pLoop+=1

print("Fin del programa")