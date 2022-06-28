# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:48:11 2022

@author: mi'to
"""

import pandas as pd
import numpy as np
import collections

df = pd.read_csv(r"C:\Users\mi'to\Desktop\result.csv")


res = []
smiles = np.array(list(df['SMILES']))

a = collections.Counter(smiles)
max = 0
sum = 0
for i in a:
    if a[i] > max:
        max = a[i]
    if a[i] >= 4:
        sum += 1
        res.append(str(i))
        

df = pd.read_csv(r"C:\Users\mi'to\Desktop\train_data.csv")
train_smiles = np.array(list(df['SMILES']))

b = train_smiles.tolist()
for i in res:
    if b.count(i):
        res.remove(i)

