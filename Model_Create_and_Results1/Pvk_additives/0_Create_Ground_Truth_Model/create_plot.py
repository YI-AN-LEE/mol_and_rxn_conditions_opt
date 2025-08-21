# another plot
from sklearn.ensemble import RandomForestRegressor
# from rdkit.Chem import PandasTools
import sklearn
import pickle
from sklearn.metrics import r2_score
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
import os 
import re
import ast
import glob
import pickle
import collections

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# another plot
from sklearn.ensemble import RandomForestRegressor
# from rdkit.Chem import PandasTools
import pickle
from sklearn.metrics import r2_score
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem

size_model_path = '/home/ianlee/opt_ian/Model_Create_and_Results1/Pvk_additives/0_Create_Ground_Truth_Model/pvkadditives/pvk_rfr_size.pkl'

with open(size_model_path, 'rb') as f:
    rf_regressor = pickle.load(f)

data = pd.read_csv('/home/ianlee/opt_ian/Model_Create_and_Results1/Pvk_additives/0_Create_Ground_Truth_Model/datasets/data_nocorr.csv')


df = pd.DataFrame(data).dropna()
y = df['crystal_size']
# modred feature calculating
calc = Calculator(descriptors, ignore_3D=True)
smiles_list = list(df['SMILES'])
print(smiles_list)
mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
mordred_df = calc.pandas(mols)

cols_to_convert = ['ATSC5v', 'AATSC5Z', 'MATS8se']

#df = pd.concat((df, mordred_df['ATSC5v'] ,mordred_df['AATSC5Z'], mordred_df['MATS8se']), axis=1)

pvk_size_feature_list = ['Reagent1 (ul)','Reagent2 (ul)','Reagent3 (ul)','Reagent4 (ul)','lab_code','ATSC5v', 'AATSC5Z', 'MATS8se']

df = df[pvk_size_feature_list]

# 縮減 X 到只包含前五十個特徵
# X_reduced = X[list(features)]

X_reduced = df[pvk_size_feature_list]

# 進行交叉驗證
scores = cross_val_score(rf_regressor, X_reduced, y, cv=10, scoring='r2')

# 計算 R^2 的平均值
average_r2 = np.mean(scores)

# 打印 R^2 的平均值
print('Average R^2:', average_r2)

# 計算平均交叉驗證分數和標準差
mean = np.mean(scores)
std = np.std(scores)

# 打印模型名稱和交叉驗證分數
print(f'Model: RandomForestRegressor\nCross-validation scores: {scores}')
y_pred = rf_regressor.predict(X_reduced)
r2 = r2_score(y, y_pred)
plt.figure(figsize=(10, 8), dpi=400)
plt.scatter(y, y_pred, alpha=0.8, color='#0F5257', s=20)  # 調整點的顏色為 'darkorange'，大小為 80
plt.xlabel('Experiment Value (Percentage)')
plt.ylabel('Prediction Value (Percentage)')

# 繪製圖表
plt.plot([0, 100], [0, 100], color='#0B3142')  # 調整對角線的顏色為 'skyblue'
plt.title(f'RandomForestRegressor for Crystal Size Prediction (8 features)')
plt.text(0.02, 0.91, f'Regression $R^2$: {r2:.4f}\nMean cross-validation score: {mean:.4f}\nStandard deviation: {std:.4f}', transform=plt.gca().transAxes)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.savefig(f'/home/ianlee/opt_ian/Model_Create_and_Results1/Pvk_addtives/0_Create_Ground_Truth_Model/Regression_Model_Analysis/Parity_plot_RandomForestRegressor_8_features.png')