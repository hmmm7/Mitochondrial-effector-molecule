# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 13:28:30 2022

@author: mi'to
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import DataStructs
import pandas as pd
import numpy as np


df = pd.read_csv(r"pre_train.csv")

mols = [Chem.MolFromSmiles(i) for i in df['SMILES']]

value = df['active'].values
tmp = []
for i in range(len(value)):
    tmp.append(value[i])
    
des=[]
for mol in mols:
    fp1_morgan_hashed = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
    des.append(fp1_morgan_hashed)
    
del_num = 4
similar = 0.8

    
'''
for i in range(len(des)):
    j = i + 1
    aaa = []
    while i < len(des) and j < len(des):
        if DataStructs.DiceSimilarity(des[i],des[j]) > 0.8:
            del(des[j])
            del(tmp[j])
            continue
        else:
            
            j += 1
  
a = np.array(des)
b = np.array(tmp)
b= b.reshape(-1, 1)
c = np.concatenate((a,b),axis=1)

np.save("train_data_full_del0.95.npy",c)
'''




aa = []
for i in range(len(des)):
    j = i + 1
    aaa = []
    while i < len(des) and j < len(des):       
        if DataStructs.DiceSimilarity(des[i],des[j]) > similar:
            aaa.append(j)
        j += 1
    aa.append(aaa)

bb = []
for i in range(len(aa)):
    if len(aa[i]) <= del_num:
        continue
    else:
        j = del_num
        while j < len(aa[i]):
            bb.append(aa[i][j])
            j += 1
    
lis1 = list(set(bb))
lis1.sort()
k = 0
for i in range(len(lis1)):
    del(des[lis1[i] - k])
    del(tmp[lis1[i] - k])
    k += 1
a = np.array(des)
b = np.array(tmp)
b= b.reshape(-1, 1)
c = np.concatenate((a,b),axis=1)
np.save("train_data_part{}_del{}.npy".format(del_num,similar),c)
'''
bb = []
for i in range(len(aa)):
    j = 0
    while  j < (len(aa[i]) // 2): 
        bb.append(aa[i][j])
        j += 1
        
lis1 = list(set(bb))
lis1.sort()
k = 0
for i in range(len(lis1)):
    del(des[lis1[i] - k])
    del(tmp[lis1[i] - k])
    k += 1
a = np.array(des)
b = np.array(tmp)
b= b.reshape(-1, 1)
c = np.concatenate((a,b),axis=1)

np.save("train_data_part5_del0.8.npy",c)


'''