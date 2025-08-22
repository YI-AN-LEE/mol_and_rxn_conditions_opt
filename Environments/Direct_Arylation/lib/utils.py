import re
import os
import random

import pandas as pd

from rdkit import Chem

BOUNBD_PATH = 'Environments/Direct_Arylation/lib/bounds.csv'
BASE_PATH = 'Model_Create_and_Results1/Direct_ary/0_Create_Ground_Truth_Model/data/base_mordred.csv'
SOLVENT_PATH = 'Model_Create_and_Results1/Direct_ary/0_Create_Ground_Truth_Model/data/solvent_mordred.csv'
LIGAND_PATH = 'Model_Create_and_Results1/Direct_ary/0_Create_Ground_Truth_Model/data/ligand_mordred.csv'
ARY_MORDRED_FEATURE_PATH = '/Model_Create_and_Results1/Direct_ary/3_Make_New_Data_Predictor/Predictor_feature.csv'

BASE_DICT = {'O=C([O-])C.[K+]': 0,
             'O=C([O-])C(C)(C)C.[K+]': 1,
             'O=C([O-])C.[Cs+]': 2,
             'O=C([O-])C(C)(C)C.[Cs+]': 3,
             }
SOLVENT_DICT = {'CCCCOC(C)=O': 0,
                'CC1=CC=C(C)C=C1': 1,
                'CCCC#N': 2,
                'CC(N(C)C)=O': 3,
                }

BASE_DICT_REVERSED = {v: k for k, v in BASE_DICT.items()}
SOLVENT_DICT_REVERSED = {v: k for k, v in SOLVENT_DICT.items()}

def load_ary_data(ary_ini_data_path, index = None):
    
    # X = Ligand Base Solvent Conc. Temp. Y = yield
    # ary_data = pd.read_csv(os.path.join(ary_ini_data_path, cycle_files[-1]), engine='python')
    ary_data = pd.read_csv(ary_ini_data_path)
    ligand_smiles = ary_data['Ligand_SMILES'].tolist()

    # Get the first center dataframe.
    if index == None:
        index = random.randint(0, len(ligand_smiles) - 1)
        print(f'Center index: {index}')
        ary_data = ary_data.iloc[[index]]
        ary_data['Base_SMILES'] = ary_data['Base_SMILES'].map(BASE_DICT)
        ary_data['Solvent_SMILES'] = ary_data['Solvent_SMILES'].map(SOLVENT_DICT)
        ary_data = ary_data.drop(columns=['Unnamed: 0'])
        ary_data = ary_data.rename(columns={
            'Ligand_SMILES': 'ligand_SMILES',
            'Base_SMILES': 'base_SMILES',
            'Solvent_SMILES': 'solvent_SMILES'
        })

    else:
        ary_data['Base_SMILES'] = ary_data['Base_SMILES'].map(BASE_DICT)
        ary_data['Solvent_SMILES'] = ary_data['Solvent_SMILES'].map(SOLVENT_DICT)
        ary_data = ary_data.drop(columns=['Unnamed: 0'])
        ary_data = ary_data.rename(columns={
            'Ligand_SMILES': 'ligand_SMILES',
            'Base_SMILES': 'base_SMILES',
            'Solvent_SMILES': 'solvent_SMILES'
        })
    
    return ary_data.reindex(columns=['ligand_SMILES', 'Temp_C', 'Concentration', 'base_SMILES', 'solvent_SMILES', 'yield'])