import sys
sys.path.append('/home/ianlee/opt_ian')

import torch
import rdkit
#import hgraph
import random
import numpy as np
import pyfiglet
import math
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "g1-eno1[0]"
import pickle


from Algorithms.Fittness import Fitness
#from Algorithms.PSO.Arg import arg_builder
from Algorithms.PSO.Create import create_tensor, create_first_particle
from Environments.PvkAdditives.lib.Pvk_Predictor import PvkTransform, Pvk_Ensemble_Predictor


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from mordred import Calculator, descriptors
import pandas as pd
from rdkit.Chem import PandasTools
#sys.path.append('/home/ianlee/JTVAE/JTVAE/GPU-P3')
from fast_jtnn import *

cutoff_dict = {"crystal_size": [1, None],
            # "crystal_score": [2, None],
        }
bounds = [(176.19463140292243, 0.5665854306219236), 
          (161.68509695291618, 0.4245779984228497), 
          (189.92178705226962, 1.0922707345129656), 
          (169.03341776133132, 0.7437770922756348), 
          (1, 0)]
   
# 讀取model
import pickle

# 指定模型存儲路徑
size_model_path = '/home/ianlee/opt_ian/Model_Create_and_Results1/Pvk_additives/0_Create_Ground_Truth_Model/pvkadditives/pvk_rfr_size.pkl'
score_model_path = '/home/ianlee/opt_ian/Model_Create_and_Results1/Pvk_additives/0_Create_Ground_Truth_Model/pvkadditives/pvk_rfc_score.pkl'
# 使用 pickle.load() 方法讀取模型
with open(size_model_path, 'rb') as f:
    rf_regressor = pickle.load(f)

with open(score_model_path, 'rb') as f:
    rf_classifier = pickle.load(f)

def adjust_position_and_get_smiles(latent, radius, transform:PvkTransform, latent_size, JTVAE):
    random_noise = torch.randn_like(latent) * radius #radius is the std of the noise
    # scaling_factors = torch.tensor([random.uniform(0, 1) for _ in range(len(total_paticle_position))]).to('cuda:0')
    total_smiles_position = latent + random_noise
    total_smiles = transform.get_smiles_from_position(total_smiles_position)  #get the SMILES list
    
    #firt time filter: put the SMILES that have error to none
    for i in range(len(total_smiles)):
        try:
            #see if any kekulize or encoding error exists
            JTVAE.encode_latent_mean([total_smiles[i]])
        except:
            total_smiles[i] = None
    
    counts = 0
    while isinstance(total_smiles, list) and None in total_smiles:
        print('rebuild again')
        none_indicy = [index for index, smiles in enumerate(total_smiles) if smiles is None]
        if counts < 20:
            none_position = total_smiles_position[none_indicy]
            random_noise = torch.randn_like(none_position) * radius
            scaling_factor = torch.tensor([random.uniform(0, 1) for _ in range(len(none_position))]).to('cuda')
            none_position = none_position + random_noise * scaling_factor.unsqueeze(1)
            none_smiles = transform.get_smiles_from_position(none_position)
        else:
            none_position = total_smiles_position[none_indicy]
            none_position = torch.rand_like(none_position) #*2-1
            none_smiles = transform.get_smiles_from_position(none_position)
        for idx, smiles_position, smiles in zip(none_indicy, none_position, none_smiles):
            if smiles is not None:
                total_smiles_position[idx] = smiles_position
                total_smiles[idx] = smiles
        for j in range(len(total_smiles)):
            try:
                #see if any kekulize or encoding error exists
                JTVAE.encode_latent_mean([total_smiles[j]])
            except:
                total_smiles[j] = None
        counts += 1
    return total_smiles

def pvk_crystal_predict(df: pd.DataFrame, 
                        pvk_rfr: RandomForestRegressor, 
                        pvk_rfc: RandomForestClassifier):
    calc = Calculator(descriptors)
    
    # mored feature creating
    esol_data = pd.DataFrame(df['SMILES'])
    PandasTools.AddMoleculeColumnToFrame(esol_data, smilesCol='SMILES')
    mordred_data = calc.pandas(esol_data['ROMol'])
    mordred_data = mordred_data.dropna(axis='columns')
    numeric_cols = mordred_data.select_dtypes(exclude='number')
    mordred_data.drop(numeric_cols, axis=1, inplace=True)
    print(mordred_data.shape)

    # processing feature creating
    df = df.drop("SMILES", axis=1)

    # combined two data
    try:
        df = df.join(mordred_data)
    except:
        pass

    # scroe predict
    pvk_size_feature_list = ['Reagent1 (ul)', 'Reagent2 (ul)', 'Reagent3 (ul)', 'Reagent4 (ul)', 'lab_code', 'ATSC5v', 'AATSC5Z', 'MATS8se']
    pvk_score_feature_list = ['Reagent1 (ul)', 'Reagent2 (ul)', 'Reagent3 (ul)', 'Reagent4 (ul)', 'AATS2p', 'ATSC5Z']
    df_size = df[pvk_size_feature_list]
    df_score = df[pvk_score_feature_list]

    return pvk_rfr.predict(df_size), pvk_rfc.predict(df_score), df[pvk_size_feature_list], df[pvk_score_feature_list]

def generate_random_list(length, max_min):
    """
    生成指定長度但數字隨機的list
    Args:
        length: 列表長度
        max_min: 最大最小值元組
    Returns:
        隨機數列表
    """
    max_value = max_min[0]
    min_value = max_min[1]

    # 生成隨機數列表
    random_list = [random.uniform(min_value, max_value) for _ in range(length)]

    return random_list

def natural_log(x):
  return math.log(x)

def load_pvk_data(data_path):
    pvk_feature_list = ['Reagent1 (ul)','Reagent2 (ul)','Reagent3 (ul)','Reagent4 (ul)','lab_code']
    source_data = pd.read_csv(data_path).sample(n=1)
    source_data[pvk_feature_list[:4]] = source_data[pvk_feature_list[:4]].apply(np.log)
    return source_data['SMILES'].iloc[0], source_data[pvk_feature_list].values

if __name__ == "__main__":

    new_bounds = []
    for i, bound in enumerate(bounds):
        if i == 4:
            new_bounds.append(bound)
        else:
            new_bounds.append((natural_log(bound[0]), natural_log(bound[1])))

    #args = arg_builder()

    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    vocab_path = '/home/ianlee/JTVAE/Ian_train/Vocabulary/smi_vocab-2.txt'
    model_path = '/home/ianlee/JTVAE/Ian_train/Train/MODEL-TRAIN-3/model.epoch-26'
    
    # Load vocabulary
    vocab = [x.strip("\r\n ") for x in open(vocab_path)]
    vocab = Vocab(vocab)

    radius = 1.0
    freeze = False
    #vae_model = HierVAE(args).cuda()
    #vae_model.load_state_dict(torch.load(args.model)[0])
    #vae_model.eval()
    
    # Initial Step for VAE
    vae_model = JTNNVAE(vocab, hidden_size=450, latent_size=32, depthT=3, depthG=20)
    vae_model.load_state_dict(torch.load(model_path, map_location='cuda'))
    vae_model.cuda()
    vae_model.eval()

    # target_smiles, initial_process_position = load_pvk_data(args.pvk_data)
    target_smiles, initial_process_position = load_pvk_data('/home/ianlee/opt_ian/Model_Create_and_Results1/Pvk_additives/0_Create_Surrogate_Model/combined_compiledData_wmol.csv')
    print("Original SMILES:", target_smiles)
    latent = vae_model.encode_latent_mean([target_smiles])
    
    #dataset = hgraph.MoleculeDataset([target_smiles], args.vocab, args.atom_vocab, args.batch_size)
    #loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=lambda x:x[0], shuffle=True, num_workers=0)
    #batch = next(iter(loader))
    
    torch.manual_seed(random.randint(0, 2**32 - 1))

    # root_vecs, z_log_var = vae_model.extract_latent_vectors(batch)
    # log_bounds, initial_log_process_position = load_leveler_data(args.leveler_data)
    # print(root_vecs.dtype)
    
    transform = PvkTransform(vae_model, latent_size=32)
    fitness = Fitness(cutoff_dict)
    
    # Create the right feature list
    calc = Calculator(descriptors)

    # Create the right feature list
    print(pyfiglet.figlet_format('Start Create Pool'))
    #生出第一個點當作中心點
    
    valid_smile_df = pd.DataFrame(columns=["SMILES", "ATSC5v", "AATSC5Z", "MATS8se"])
    #valid_tensor = []
    while len(valid_smile_df) < 10: #original: 16
        #add noise and decode, 可以全部重寫
        #total_paticle_position, total_smiles = create_tensor(center_particle, bounds, freeze, radius, 1, transform)
        latent = torch.stack([latent] * 1) # torch.Size([pop_size, 43])
        total_smiles = adjust_position_and_get_smiles(latent, radius, transform, latent_size=32, JTVAE=vae_model)
        try:
            vae_model.encode_latent_mean(total_smiles)
            if (total_smiles[0] not in valid_smile_df['SMILES']):
                if any(sub in str(total_smiles[0]) for sub in ['[n-]', 'H+', '[NH2+]', '[O-]', 'S-', '+']):
                    print('invalid smiles')
                else:
                    print(len(valid_smile_df), total_smiles[0])
                    esol_data = pd.DataFrame({'SMILES':total_smiles})
                    PandasTools.AddMoleculeColumnToFrame(esol_data, smilesCol='SMILES')
                    mordred_data = calc.pandas(esol_data['ROMol'])
                    mordred_data = mordred_data.dropna(axis='columns')
                    numeric_cols = mordred_data.select_dtypes(exclude='number')
                    mordred_data.drop(numeric_cols, axis=1, inplace=True)
                    print(mordred_data.shape)
                    # 檢查每個列是否存在於 DataFrame 中
                    columns_exist = all(item in mordred_data.columns for item in ["ATSC5v", "AATSC5Z", "MATS8se",'AATS2p', 'ATSC5Z'])

                    if columns_exist:
                        # 如果所有列都存在，創建一個新的 DataFrame，只包含 "smile" 列和這三個分子特徵列
                        new_data = pd.DataFrame({
                            "ATSC5v": mordred_data["ATSC5v"],
                            "AATSC5Z": mordred_data["AATSC5Z"],
                            "MATS8se": mordred_data["MATS8se"],
                            "AATS2p": mordred_data["AATS2p"],
                            "ATSC5Z": mordred_data["ATSC5Z"],
                            "SMILES": [total_smiles[0]]
                        })
                        valid_smile_df = pd.concat([valid_smile_df, new_data], ignore_index=True)
                        #valid_smile_df.append(new_data, ignore_index=True) #keep the unique SMILES
                        #valid_tensor.append(total_paticle_position[0][:32])
                #valid_tensor = torch.stack(valid_tensor).cuda()
        except:
            print('invalid smiles')
            continue
        
    print(valid_smile_df)
    #return the final value
    print('start predict crystal size value')

    max_min_dict = {'Reagent1 (ul)': (177, 0), 'Reagent2 (ul)': (162, 0), 'Reagent3 (ul)': (190, 0), 'Reagent4 (ul)': (170, 0), 'lab_code': (0, 1)}

   

    # Duplicate the DataFrame to increase its size by a factor of 10
    #valid_smile_df = pd.concat([valid_smile_df] * 10, ignore_index=True)
    valid_smile_df = valid_smile_df.reset_index(drop=True)
    
    valid_proc_dict = {}
    
    for key, value in max_min_dict.items():
        valid_proc_dict[key] = generate_random_list(len(valid_smile_df), value)
    valid_proc_dict = pd.DataFrame(valid_proc_dict)
    valid_proc_dict['lab_code'] = np.where(valid_proc_dict['lab_code'] > 0.5, 1, 0)
    
    valid_proc_dict = valid_proc_dict.reset_index(drop=True)
    combined_df = pd.concat([valid_smile_df, valid_proc_dict], axis=1)

    pvk_gt_pred = pvk_crystal_predict(combined_df, rf_regressor, rf_classifier)
    combined_df['crystal_size'] = list(pvk_gt_pred[0])
    combined_df['crystal_score'] = list(pvk_gt_pred[1])
    combined_df.to_csv('/home/ianlee/opt_ian/Model_Create_and_Results1/Pvk_additives/0_Create_Ground_Truth_Model/cycle0_new.csv')

    print('final df saved as cycle0.csv')