# Import RDkit packages
from rdkit.Chem import PandasTools
from rdkit import Chem
import pandas as pd
from mordred import Calculator, descriptors

data_path = './datasets/combined_compiledData.csv'
data_Xy = pd.read_csv(data_path)
data_Xy = pd.DataFrame(data_Xy)

data_noAdditive = data_Xy[data_Xy['SMILES']!= 'no_additive'] # 條件篩選出沒有添加劑的數據，並將其載入 data_noAdditive 變數中。
data_noAdditive = data_noAdditive.dropna(subset=['crystal_score','crystal_size'])
X_withSMILES = data_noAdditive.drop(['crystal_score','crystal_size'],axis=1) 
esol_data = X_withSMILES['SMILES']
esol_data = pd.DataFrame(X_withSMILES['SMILES'])
PandasTools.AddMoleculeColumnToFrame(esol_data, smilesCol='SMILES')

print('clean data done')

calc = Calculator(descriptors)
mordred_data = calc.pandas(esol_data['ROMol'])
mordred_data = mordred_data.dropna(axis='columns')
mordred_data.to_csv('./datasets/mordred_data_2.csv', index = False)