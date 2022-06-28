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


#读取数据
c = np.load(r"train_data_part4_del0.95.npy")
a = c[:,0:1024]
b = c[:,1024:]

#划分训练集，测试集，验证集
x_train_all,x_test,y_train_all,y_test=train_test_split(a, b, test_size=0.1,random_state=0)
x_train,x_valid,y_train,y_valid =train_test_split(x_train_all, y_train_all, test_size=0.1,random_state=0)

#仿照论文贝叶斯调参代码
#参数列表
space_ = {'n_estimators': hp.choice('n_estimators', [10, 50, 100, 200, 300, 400, 500]),
          'max_depth': hp.choice('max_depth', range(3, 12)),
          'min_samples_leaf': hp.choice('min_samples_leaf', [1, 3, 5, 10, 20, 50]),
          'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0, 0.01),
          'max_features': hp.choice('max_features', [0.5, 0.6, 0.7, 0.8, 0.9])
          }
n_estimators_ls = [10, 50, 100, 200, 300, 400, 500]
max_depth_ls = range(3, 12)
min_samples_leaf_ls = [1, 3, 5, 10, 20, 50]
max_features_ls = [0.5, 0.6, 0.7, 0.8, 0.9]
#

def hyper_opt(args):
    model = RandomForestClassifier(**args, n_jobs=6, random_state=1, verbose=0, class_weight='balanced')        
    model.fit(x_train, y_train)
    val_preds = model.predict_proba(x_valid)
    loss = 1 - roc_auc_score(y_valid, val_preds[:, 1])
    return {'loss': loss, 'status': STATUS_OK}

# start hyper-parameters optimization
trials = Trials()
best_results = fmin(hyper_opt, space_, algo=tpe.suggest, max_evals=50, trials=trials)
print('the best hyper-parameters : ',best_results)

best_model = RandomForestClassifier(n_estimators=n_estimators_ls[best_results['n_estimators']],
                                    max_depth=max_depth_ls[best_results['max_depth']],
                                    min_samples_leaf=min_samples_leaf_ls[best_results['min_samples_leaf']],
                                    max_features=max_features_ls[best_results['max_features']],
                                    min_impurity_decrease=best_results['min_impurity_decrease'],
        
                                n_jobs=6, random_state=1, verbose=0, class_weight='balanced') 

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

parameters = ['max_depth', 'max_features', 'min_samples_leaf']  
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

'''
from sklearn.model_selection import KFold 

kf = KFold(n_splits=10, random_state=None) # 10折
acc_train = []
auc_train = []

auc_valid = []
acc_valid = []

auc_test = []
acc_test = []
#显示具体划分情况
i = 1
for train_index, valid_index in kf.split(x_train_all):
      print('\n{} of kfold {}'.format(i,kf.n_splits))
      x_train, x_valid = x_train_all[train_index], x_train_all[valid_index] 
      y_train, y_valid = y_train_all[train_index], y_train_all[valid_index] 
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
     
      i += 1

print("acc_train: {:.3f} ± {:.3f}".format(statistics.mean(acc_train),statistics.stdev(acc_train)))
print("auc_train: {:.3f} ± {:.3f}".format(statistics.mean(auc_train),statistics.stdev(auc_train)))

print("acc_valid: {:.3f} ± {:.3f}".format(statistics.mean(acc_valid),statistics.stdev(acc_valid)))
print("auc_valid: {:.3f} ± {:.3f}".format(statistics.mean(auc_valid),statistics.stdev(auc_valid)))

print("acc_test: {:.3f} ± {:.3f}".format(statistics.mean(acc_test),statistics.stdev(acc_test)))
print("auc_test: {:.3f} ± {:.3f}".format(statistics.mean(auc_test),statistics.stdev(auc_test)))

'''
