# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 18:52:00 2022

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
from xgboost import XGBRegressor, XGBClassifier
import statistics
from sklearn.model_selection import cross_val_score
warnings.filterwarnings("ignore")

c = np.load(r"train_data_part4_del0.95.npy")
a = c[:,0:1024]
b = c[:,1024:]

x_train_all,x_test,y_train_all,y_test=train_test_split(a, b, test_size=0.1,random_state=0)
x_train,x_valid,y_train,y_valid =train_test_split(x_train_all, y_train_all, test_size=0.1,random_state=0)

space_ = {'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
          'gamma': hp.uniform('gamma', 0, 0.2),
          'min_child_weight': hp.choice('min_child_weight', range(1, 6)),
          'subsample': hp.uniform('subsample', 0.7, 1.0),
          'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
          'max_depth': hp.choice('max_depth', range(3, 10)),
          'n_estimators': hp.choice('n_estimators', [50, 100, 200, 300, 400, 500, 1000])}
min_child_weight_ls = range(1, 6)
max_depth_ls = range(3, 10)
n_estimators_ls = [100, 200, 300, 400, 500, 1000, 1500, 2000]

def hyper_opt(args):

    model = XGBClassifier(**args, n_jobs=6, random_state=1, seed=1,) 
    model.fit(x_train, y_train, eval_metric='auc',
              eval_set=[(x_valid, y_valid)],
              early_stopping_rounds=30, verbose=False)
    val_preds = model.predict_proba(x_valid, ntree_limit=model.best_ntree_limit) 
    loss = 1 - roc_auc_score(y_valid, val_preds[:, 1]) 
    return {'loss': loss, 'status': STATUS_OK}
trials = Trials()
best_results = fmin(hyper_opt, space_, algo=tpe.suggest, max_evals=50, trials=trials)
print('the best hyper-parameters : ' , best_results)
best_model = XGBClassifier(n_estimators=n_estimators_ls[best_results['n_estimators']],
                           max_depth=max_depth_ls[best_results['max_depth']],
                           min_child_weight=min_child_weight_ls[best_results['min_child_weight']],
                           learning_rate=best_results['learning_rate'],
                           gamma=best_results['gamma'],
                           subsample=best_results['subsample'],
                           colsample_bytree=best_results['colsample_bytree'],
                           n_jobs = -1, random_state=1, seed=1)
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

parameters = ['learning_rate', 'gamma', 'min_child_weight']  
cols = len(parameters)
f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(20, 5))
cmap = plt.cm.jet
for i, val in enumerate(parameters):
    print(i,val)
    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    ys = [1-t['result']['loss'] for t in trials.trials]
    ys = np.array(ys)
    axes[i].scatter(
        xs,
        ys,
        s=20,
        linewidth=0.01,
        alpha=1,
        c=cmap(float(i) / len(parameters)))
    axes[i].set_title(val)



