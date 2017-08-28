#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 21:21:10 2017

@author: canf
"""
import geopandas as gpd
import matplotlib.pyplot as plt

df = gpd.read_file('./input-data/san-francisco_california_roads.geojson')
print( df.shape )

df.plot()
plt.show()