#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:18:38 2017

@author: canf
"""

import pandas as pd

(pd.read_csv('./input-data/sf_train.csv')
   .replace(r'LARCENY/THEFT', r'LARCENY-THEFT', regex=True)
   .replace(r'FORGERY/COUNTERFEITING', r'FORGERY-COUNTERFEITING', regex=True)
   .replace(r'PORNOGRAPHY/OBSCENE MAT', r'PORNOGRAPHY-OBSCENE MAT', regex=True)
   .replace(r'DRUG/NARCOTIC', r'DRUG-NARCOTIC', regex=True)
   .to_csv('./input-data/sf_train_fixed.csv', index=False))

print("Sustitucion de diagonal hecha")