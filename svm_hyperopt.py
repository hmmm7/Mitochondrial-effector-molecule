# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 12:51:50 2022

@author: mi'to
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import DataStructs
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc,roc_auc_score  ###计算roc和auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import warnings
from sklearn import svm
from hyperopt import hp,STATUS_OK,Trials,fmin,tpe
import random
import statistics

warnings.filterwarnings("ignore")

c = np.load(r"train_data_part_del0.95.npy")
a = c[:,0:1024] #x值
b = c[:,1024:] #y值



