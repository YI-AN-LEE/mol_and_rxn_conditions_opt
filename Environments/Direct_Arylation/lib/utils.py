import re
import os
import random

import pandas as pd

from rdkit import Chem

BOUNBD_PATH = '/home/ianlee/opt_ian/Environments/Direct_Arylation/lib/bounds.csv'
BASE_PATH = '/home/ianlee/opt_ian/Model_Create_and_Results1/Direct_ary/0_Create_Ground_Truth_Model/data/base_mordred.csv'
SOLVENT_PATH = '/home/ianlee/opt_ian/Model_Create_and_Results1/Direct_ary/0_Create_Ground_Truth_Model/data/solvent_mordred.csv'
LIGAND_PATH = '/home/ianlee/opt_ian/Model_Create_and_Results1/Direct_ary/0_Create_Ground_Truth_Model/data/ligand_mordred.csv'
ARY_MORDRED_FEATURE_PATH = '/home/ianlee/opt_ian/Model_Create_and_Results1/Direct_ary/3_Make_New_Data_Predictor/Predictor_feature.csv'

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

def change_phos_to_nitrogen(smiles):
    '''
    Change the phosphorus atom to nitrogen atom in the smiles if it has three bonds.
    '''
    mol = Chem.MolFromSmiles(smiles)
    atoms = mol.GetAtoms()
    for atom in atoms:
        if atom.GetSymbol() == 'P' and atom.GetTotalDegree() == 3:
            atom.SetAtomicNum(7)  # Set the atomic number to 7 (Nitrogen)

    # Convert the modified molecule back to SMILES
    new_smiles = Chem.MolToSmiles(mol)
    return new_smiles

def change_nitro_to_phos(smiles):
    '''
    Change a nitrogen atom to a phosphorus atom in the smiles if it has three bonds.
    Randomly select one if there are multiple.
    '''
    # 如果輸入是一個列表，則對列表中的每個元素執行此函數
    if isinstance(smiles, list):
        return [change_nitro_to_phos(s) for s in smiles]
    
    mol = Chem.MolFromSmiles(smiles)
    # Chem.Kekulize(mol)
    # Chem.RWMol(mol)
    atoms = mol.GetAtoms()

    # Find all nitrogen atoms with three bonds
    nitro_atoms = [atom for atom in atoms if atom.GetSymbol() == 'N' and atom.GetTotalDegree() == 3 and atom.GetExplicitValence() <= 3 and not atom.IsInRing()]
    nitro_atoms_no_h = [atom for atom in atoms if atom.GetSymbol() == 'N' and atom.GetTotalDegree() == 3 and atom.GetExplicitValence() <= 3 and not atom.IsInRing() and atom.GetTotalNumHs() == 0]
    selected_atoms = nitro_atoms_no_h if nitro_atoms_no_h else nitro_atoms

    if selected_atoms:
    # Randomly select one
        selected_atom = random.choice(selected_atoms)
        selected_atom.SetAtomicNum(15)  # Set the atomic number to 15 (Phosphorus)

        # Convert the modified molecule back to SMILES
        new_smiles = Chem.MolToSmiles(mol)
        return new_smiles
    else:
        return smiles

# 20240529

def load_ary_data(ary_ini_data_path, index = None):
    
    """
    files = os.listdir(ary_ini_data_path)
    pattern = re.compile(r'cycle\d+\.csv')
    cycle_files = [file for file in files if pattern.match(file)]
    numbers = [int(re.search(r'\d+', file).group()) for file in cycle_files]
    cycle_files = [file for _, file in sorted(zip(numbers, cycle_files))]
    """
    
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

# if __name__ == "__main__":
#     smiles = "CN(C)c1ccc(nc1)[N+](=O)[O-]"
#     print(change_nitro_to_phos(smiles))