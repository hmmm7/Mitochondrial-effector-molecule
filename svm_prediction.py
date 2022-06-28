# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:52:51 2022

@author: mi'to
"""


from rdkit import Chem, DataStructs
import math
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem.EState import Fingerprinter
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
from sklearn.metrics import roc_curve, auc,roc_auc_score  ###计算roc和auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc,roc_auc_score  ###计算roc和auc
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#交叉验证
from sklearn.model_selection import cross_val_score
#gridSearchCv调参
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.svm import SVC
import statistics


svm = joblib.load('svm.pkl')
c = np.load(r"train_data_part4_del0.95.npy")
x = c[:,0:1024]
y = c[:,1024:]

svm.fit(x,y)

df = pd.read_csv(r"fu.csv")

mols = [Chem.MolFromSmiles(i) for i in df['SMILES']]

des=[]
for mol in mols:
    fp1_morgan_hashed = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
    des.append(fp1_morgan_hashed)
x_predict = np.array(des)
y_predict = svm.predict_proba(x_predict)

'''
df = pd.read_csv(r"pre_train.csv")

a = df['SMILES']

a = np.array(a)

df = pd.read_csv(r"result.csv")

b = df['SMILES']

b = np.array(b)

c = []

for i in b:
    flag = 0
    for j in a:
        if i == j:
            flag = 1
    if flag == 0:
        c.append(i)




c = np.load(r"train_data_full_del0.95.npy")
svm = SVC(probability=True)
np.random.shuffle(c)

num = len(c) * 0.1
if len(c) * 0.1 + 0.5 > int(num + 1):
    num = int(num + 1)
else:
    num = int(num)
#划分test的长度，后续代码test的长度与valid长度相等，剩下的全为train，比例train：valid：test = 8 ： 1 ： 1
test_all = []
train_all = []
valid_all = []
i = 0 #代表索引
k = 0 #代表test从前往后已经到了第几块
while i < len(c) and k < 10:
    if k == 9:
        test = c[i:len(c)]
        valid = c[i - len(test) : i]
        train = c[0 : i - len(test)]
    else:
        test = c[i : i + num]
        valid = c[i + num : i + num + len(test)]
        tmp1 = c[0:i]
        tmp2 = c[i + num + len(test) : len(c)]
        train = np.concatenate((tmp1,tmp2))
    test_all.append(test)
    train_all.append(train)
    valid_all.append(valid)
    i += num
    k += 1
#划分完成，下面进行交叉验证    
acc_train = []
auc_train = []

auc_valid = []
acc_valid = []

auc_test = []
acc_test = []
for i in range(len(train_all)):
    print('第%d次：  ',i)
    x_train, x_valid, x_test = train_all[i][:,0:1024], valid_all[i][:,0:1024], test_all[i][:,0:1024]
    y_train, y_valid, y_test = train_all[i][:,1024:], valid_all[i][:,1024:], test_all[i][:,1024:]
    svm.fit(x_train, y_train)
    pred = svm.predict(x_train)
    pred_pro = svm.predict_proba(x_train)[:, 1]
    acc = metrics.accuracy_score(y_train, pred)
    auc = roc_auc_score(y_train, pred_pro)
    acc_train.append(acc)
    auc_train.append(auc)
    
    pred = svm.predict(x_valid)
    pred_pro = svm.predict_proba(x_valid)[:, 1]
    acc = metrics.accuracy_score(y_valid, pred)
    auc = roc_auc_score(y_valid, pred_pro)
    acc_valid.append(acc)
    auc_valid.append(auc)


    pred = svm.predict(x_test)
    pred_pro = svm.predict_proba(x_test)[:, 1]
    acc = metrics.accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, pred_pro)
    acc_test.append(acc)
    auc_test.append(auc)
    
print("acc_train: {:.3f} ± {:.3f}".format(statistics.mean(acc_train),statistics.stdev(acc_train)))
print("auc_train: {:.3f} ± {:.3f}".format(statistics.mean(auc_train),statistics.stdev(auc_train)))

print("acc_valid: {:.3f} ± {:.3f}".format(statistics.mean(acc_valid),statistics.stdev(acc_valid)))
print("auc_valid: {:.3f} ± {:.3f}".format(statistics.mean(auc_valid),statistics.stdev(auc_valid)))

print("acc_test: {:.3f} ± {:.3f}".format(statistics.mean(acc_test),statistics.stdev(acc_test)))
print("auc_test: {:.3f} ± {:.3f}".format(statistics.mean(auc_test),statistics.stdev(auc_test)))

'''