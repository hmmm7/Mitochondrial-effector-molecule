# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 20:34:03 2022

@author: mi'to
"""



import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import DataStructs
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import time
import warnings
import sys
from hyperopt import hp,STATUS_OK,Trials,fmin,tpe
from sklearn.feature_selection import SelectPercentile, f_classif, SelectFromModel
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc, mean_squared_error, \
    r2_score, mean_absolute_error
from xgboost import XGBRegressor, XGBClassifier
import multiprocessing
from sklearn.model_selection import cross_val_score
import statistics
warnings.filterwarnings("ignore")
import statistics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

space_nn = {
    'batch_size':hp.choice('batch_size',[10, 20, 40, 60, 80, 100]),
    "solver": hp.choice('solver', ['adam', 'sgd', 'lbfgs']),
    'alpha': hp.uniform('alpha', 0.1, 1.0),
    'activation' : hp.choice('activation',['identity', 'logistic', 'tanh', 'relu']),
    "hidden_layer_sizes":hp.choice('hidden_layer_sizes', [(100,), (100, 30)]) 
}
solver_ls = ['adam', 'sgd', 'lbfgs']
activation_ls = ['identity', 'logistic', 'tanh', 'relu']
hidden_layer_sizes_ls = [(100,), (100, 30)]
batch_size_ls = [10, 20, 40, 60, 80, 100]


c = np.load(r"train_data_part4_del0.95.npy")
a = c[:,0:1024]
b = c[:,1024:]




x_train_all,x_test,y_train_all,y_test=train_test_split(a, b, test_size=0.1,random_state=0)
x_train,x_valid,y_train,y_valid =train_test_split(x_train_all, y_train_all, test_size=0.1,random_state=0)

def hyper_opt(args):
    model = MLPClassifier(**args) 
    model.fit(x_train, y_train)
    val_preds = model.predict_proba(x_valid) 
    loss = 1 - roc_auc_score(y_valid, val_preds[:, 1]) 
    return {'loss': loss, 'status': STATUS_OK}

# start hyper-parameters optimization
trials = Trials()
best_results = fmin(hyper_opt, space_nn, algo=tpe.suggest, max_evals=50, trials=trials)
print('the best hyper-parameters : ' , best_results)
best_model = MLPClassifier(solver= solver_ls[best_results['solver']],
                                    activation = activation_ls[best_results['activation']],
                                    hidden_layer_sizes = hidden_layer_sizes_ls[best_results['hidden_layer_sizes']],
                                    batch_size = batch_size_ls[best_results['batch_size']],
                                    alpha = best_results['alpha']
                                    ) 
    

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
    best_model.fit(x_train, y_train)
    pred = best_model.predict(x_train)
    pred_pro = best_model.predict_proba(x_train)[:, 1]
    acc = metrics.accuracy_score(y_train, pred)
    auc = roc_auc_score(y_train, pred_pro)
    acc_train.append(acc)
    auc_train.append(auc)
    
    pred = best_model.predict(x_valid)
    pred_pro = best_model.predict_proba(x_valid)[:, 1]
    acc = metrics.accuracy_score(y_valid, pred)
    auc = roc_auc_score(y_valid, pred_pro)
    acc_valid.append(acc)
    auc_valid.append(auc)


    pred = best_model.predict(x_test)
    pred_pro = best_model.predict_proba(x_test)[:, 1]
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