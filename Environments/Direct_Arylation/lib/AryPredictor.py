import os
import re
# import sys
import glob
import torch
import pickle

import pandas as pd
import numpy as np

from typing import List
#from hgraph import HierVAE
from xgboost import XGBRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors
from Algorithms.ABC.Bee import Bee
from Algorithms.SA.Solution import Solution
from Algorithms.GA.Individual import Individual
from Algorithms.PSO.Particle import Particle
from Environments.Direct_Arylation.lib.AryTypes import ExperimentProperty
from Environments.Direct_Arylation.lib.utils import change_nitro_to_phos, ARY_MORDRED_FEATURE_PATH, BASE_PATH, SOLVENT_PATH, BASE_DICT_REVERSED, SOLVENT_DICT_REVERSED
from fast_jtnn import JTNNVAE

class AryTransform():
    def __init__(self, JTVAE, proc_feature, latent_size, feature_data_path = ARY_MORDRED_FEATURE_PATH) -> None:
        self.JTVAE : JTNNVAE = JTVAE
        #self.predictor_feature = pd.read_csv(feature_data_path)['Feature'].tolist()
        
        #we do not need the proc feature anymore since there cannot be any filtered steps
        self.proc_feature = proc_feature
        self.latent_size = latent_size

    def get_smiles_from_position(self, positions):
        '''
        Decode the SMILES from the latent points in the HierVAE model. Sometimes the decoding may fail.
        :param positions: A list of torch tensors of latent points.
        :return: A list of decoded SMILES.
        '''
        smiles_list = []
        for i in range(len(positions)):
            tree_vec,mol_vec = torch.split(torch.reshape(positions[i], (1,self.latent_size)), self.latent_size//2, dim=1)
            smiles = self.JTVAE.decode(tree_vec, mol_vec, prob_decode=False)
            smiles_list.append(smiles) 
        return [self.cannonicalize_smiles(smiles) for smiles in smiles_list]
    
    
        """
        except:
            smiles_list = []
            for position in positions:
                try:
                    smiles = self.hier_vae.decode_from_latent_vectors(position.unsqueeze(0))[0]
                    #smiles = change_nitro_to_phos(smiles)
                    smiles_list.append(smiles)
                except:
                    smiles_list.append(None)
            return smiles_list
        """
        
        
    def cannonicalize_smiles(self, smiles):
        '''
        Ensure that different chemical structures have the same representation.
        '''
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    
    def to_dataframe(self, smiles_list, process_tensor:torch.Tensor, column_names):
        """
        Convert smiles list and process tensor to dataframe.
        Args:
            smiles_list: The list to be converted.
            process_tensor: The tensor to be converted.
            column_names: The column names of tensor_data.
        Returns:
            Dataframe.
        """
        # print('*************')
        # print('smiles_list', smiles_list)
        # print(process_tensor[0])
        # print('column_names', column_names)
        # print('========================')
        process_tensor_cpu = process_tensor.detach().cpu()
        feature_df = pd.DataFrame(data=smiles_list, columns=['ligand_SMILES'])
        feature_df = pd.concat([feature_df, pd.DataFrame(data=process_tensor_cpu.numpy(), columns=column_names)], axis=1)
        # print('*** feature_df.columns ###',feature_df.columns)
        return feature_df
    
    def transform_psopool_to_feature(self, pool:List[Particle]):
        #total_smile_name = [particle.smiles for particle in pool]
        total_position = [particle.position for particle in pool]
        total_position = torch.stack(total_position, dim=0)
        #total_process_position = total_position[:, self.latent_size:]
        return total_position
    
    def transform_sapool_to_feature(self, solution_pool:List[Solution]):
        total_smile_name = [solution.candidate_smiles for solution in solution_pool]
        total_position = [solution.candidate_position for solution in solution_pool]
        total_position = torch.stack(total_position, dim=0)
        total_process_position = total_position[:, self.latent_size:]
        return self.to_dataframe(total_smile_name, total_process_position, self.proc_feature)
    
    def transform_beeswarm_to_feature(self, bee_swarm:List[Bee]):
        #total_smile_name = [bee.candidate_smiles for bee in bee_swarm]
        total_position = [bee.candidate_position for bee in bee_swarm]
        total_position = torch.stack(total_position, dim=0)
        #total_process_position = total_position[:, self.latent_size:]
        return total_position
    
    def transform_gapop_to_feature(self, population:List[Individual]):
        total_smile_name = [individual.smiles for individual in population]
        total_position = [individual.position for individual in population]
        total_position = torch.stack(total_position, dim=0)
        total_process_position = total_position[:, self.latent_size:]
        return self.to_dataframe(total_smile_name, total_process_position, self.proc_feature)
    
class AryEnsmblePredictor():
    def __init__(self,
                 JTVAE,
                 model_path,
                 latent_size,
                 normalize_dict = None,
                 base_path = BASE_PATH,
                 solv_path = SOLVENT_PATH,
                 log = False
                 ) -> None:
        self.model_path = model_path
        self.feature_df: pd.DataFrame = None
        self.ary_yield_models: List[XGBRegressor] = []
        self.latent_size = latent_size
        #self.predictor_feature = pd.read_csv(feature_data_path)['Feature'].tolist()
        #if 'base_SMILES' not in self.predictor_feature:
            #self.predictor_feature.append('base_SMILES')
        #if 'solvent_SMILES' not in self.predictor_feature:
            #self.predictor_feature.append('solvent_SMILES')
            
        self.base_mordred = pd.read_csv(base_path)
        self.solv_mordred = pd.read_csv(solv_path)
        
        #self.base_mordred = self.base_mordred.filter(items=self.predictor_feature)
        #self.solv_mordred = self.solv_mordred.filter(items=self.predictor_feature)
        
        #change the ligand feature to VAE latent vector
        #self.ligand_feature_list = [feature.replace('ligand_', '') for feature in self.predictor_feature if feature.startswith('ligand_')]
        #self.calc = Calculator(descriptors)
        
        self.normalize_dict = normalize_dict
        self.log = log
        self.load_model()
        self.JTVAE : JTNNVAE = JTVAE
    
    def load_model(self):
        max_num = 0
        pattern = r'^(.*)_c(\d+)\.pkl$'
        ary_yield_model_paths = glob.glob(os.path.join(self.model_path, "*ary_yield_xgboost*.pkl"))
        for xgb in ary_yield_model_paths:
            match = re.match(pattern, xgb)
            _, group_num = match.groups()
            if int(group_num) > max_num:
                max_num = int(group_num) 
        ary_yield_model_paths = glob.glob(os.path.join(self.model_path, f"*c{max_num}*.pkl"))

        for path in ary_yield_model_paths:
            with open(path, "rb") as f:
                model = pickle.load(f)
            self.ary_yield_models.append(model)
        # print(self.ary_yield_models[0].get_booster().feature_names)
    
    def ensemble_predict(self, total_particle_position):
        # print(type(self.normalize_dict), self.normalize_dict)
        feature_df = pd.DataFrame()
        feature_df['Concentration'] = total_particle_position[:, self.latent_size+1].detach().cpu().numpy()
        feature_df['Temp_C'] = total_particle_position[:, self.latent_size].detach().cpu().numpy()

        if self.normalize_dict:
            for column, should_normalize in self.normalize_dict.items():
                if should_normalize:
                    feature_df[column] = np.expm1(feature_df[column])
        
        # print(feature_df[['ligand_SMILES','Temp_C','Concentration']])
        # print('------------------------------')

        latent_points = total_particle_position[:, :self.latent_size].detach().cpu().numpy()

        # map the number of base and solvent to their SMILES 
        feature_df['base_SMILES'] = total_particle_position[:, self.latent_size+2].detach().cpu().numpy().astype(int)
        feature_df['base_SMILES'] = feature_df['base_SMILES'].map(BASE_DICT_REVERSED)
        base_df = feature_df['base_SMILES']
        feature_df['solvent_SMILES'] = total_particle_position[:, self.latent_size+3].detach().cpu().numpy().astype(int)
        feature_df['solvent_SMILES'] = feature_df['solvent_SMILES'].map(SOLVENT_DICT_REVERSED)
        solvent_df = feature_df['solvent_SMILES']
        
        # Map the solv, base,  SMILES to their Mordred features
        feature_df = pd.merge(feature_df, self.base_mordred, on='base_SMILES')
        feature_df = pd.merge(feature_df, self.solv_mordred, on='solvent_SMILES')
        feature_df = feature_df.drop(columns=['base_SMILES', 'solvent_SMILES'])
        
        
        #print(feature_df.columns)
        #print(latent_points.shape)
        #feature_df = pd.merge(feature_df, ligand_features, on='ligand_SMILES')
        #feature_df.to_csv('/home/ianlee/optimizer/OW_DOE/test3.csv')
        #feature_df.sort_values('reaction_index', inplace=True)
        #feature_df.reset_index(drop=True, inplace=True)  # 重設索引

        #null 先不處理
        """
        feature_df_nonull = feature_df.dropna()
        feature_df_nonull = feature_df_nonull.drop(columns=['base_SMILES', 'solvent_SMILES','ligand_SMILES'])
        feature_df_nonull = feature_df_nonull.apply(pd.to_numeric, errors='coerce')
        feature_df_nonull = feature_df_nonull.reindex(columns=self.predictor_feature[:-2])
        # print(feature_df_nonull[['Temp_C','Concentration']].head())
        deleted_rows = feature_df.index[~feature_df.index.isin(feature_df_nonull.index)].tolist()
        """
        
        feature_df_nonull = np.concatenate((latent_points, feature_df), axis=1)
        
        yield_results = []
        yield_mean_and_std = []
        for yield_prediction_model in self.ary_yield_models:
            yield_results.append(yield_prediction_model.predict(feature_df_nonull))
                
        for i in range(len(yield_results[0])):
            values = [prediction[i] for prediction in yield_results]
            yield_mean_and_std.append((np.mean(values), np.std(values)))
    
        #for position in deleted_rows:
            #yield_mean_and_std = yield_mean_and_std[:position] + [(1e-8, 0)] + yield_mean_and_std[position:]
            
        yield_mean_and_std = [tuple(row) for row in yield_mean_and_std]
        #feature_df = feature_df.assign(reation_yield = yield_mean_and_std)
        total_fitness = yield_mean_and_std[:]
        total_fitness = [row[0] for row in total_fitness]

        """
        sulfoxide_smarts = 'S(=O)'
        sulfoxide_query = Chem.MolFromSmarts(sulfoxide_smarts)

        for i in range(len(feature_df)):
            # print(feature_df.loc[i, 'ligand_SMILES'])
            mol = Chem.MolFromSmiles(feature_df.loc[i, 'ligand_SMILES'])
            mol_weight = Descriptors.MolWt(mol)
            phos_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 15]

            # Check if any phosphorus atom has 0 hydrogen atoms
            if len(phos_atoms) == 1 and any(atom.GetTotalNumHs() == 0 for atom in phos_atoms) and not mol.HasSubstructMatch(sulfoxide_query):
                mean = total_fitness[i]
                total_fitness[i] = float(mean) * 1.5 # float(mean) * 1.5

            if mol_weight > 600:
                mean = total_fitness[i]
                total_fitness[i] = float(mean) * 0.1

            if not phos_atoms:
                mean = total_fitness[i]
                total_fitness[i] = float(mean) * 0.5
        """
        
        # print('ensemble_predict')
        # for smiles, fitness, yield_ in zip(feature_df['ligand_SMILES'], total_fitness, feature_df['reation_yield']):
        #     print(f'{smiles}\t{fitness}\t{yield_[0]}')
        # print('ensemble_predict')
        total_propertys = []
        # sys.exit()
        for i in range(len(total_particle_position)):
            properties = ExperimentProperty(
                Base = base_df[i],
                Solvent = solvent_df[i],
                Temperature = total_particle_position[i, self.latent_size].detach().cpu().numpy(),
                Concentration = total_particle_position[i, self.latent_size+1].detach().cpu().numpy(),
                reation_yield = yield_mean_and_std[i],
        )
            total_propertys.append(properties)
        return total_fitness, total_propertys
    
    def ensemble_predict_BO(self, total_particle_position):
        # print(type(self.normalize_dict), self.normalize_dict)
        feature_df = pd.DataFrame()
        feature_df['Concentration'] = total_particle_position[:, self.latent_size+1].detach().cpu().numpy()
        feature_df['Temp_C'] = total_particle_position[:, self.latent_size].detach().cpu().numpy()

        if self.normalize_dict:
            for column, should_normalize in self.normalize_dict.items():
                if should_normalize:
                    feature_df[column] = np.expm1(feature_df[column])
        
        # print(feature_df[['ligand_SMILES','Temp_C','Concentration']])
        # print('------------------------------')

        latent_points = total_particle_position[:, :self.latent_size].detach().cpu().numpy()

        # map the number of base and solvent to their SMILES 
        feature_df['base_SMILES'] = total_particle_position[:, self.latent_size+2].detach().cpu().numpy().astype(int)
        feature_df['base_SMILES'] = feature_df['base_SMILES'].map(BASE_DICT_REVERSED)
        base_df = feature_df['base_SMILES']
        feature_df['solvent_SMILES'] = total_particle_position[:, self.latent_size+3].detach().cpu().numpy().astype(int)
        feature_df['solvent_SMILES'] = feature_df['solvent_SMILES'].map(SOLVENT_DICT_REVERSED)
        solvent_df = feature_df['solvent_SMILES']
        
        # Map the solv, base,  SMILES to their Mordred features
        feature_df = pd.merge(feature_df, self.base_mordred, on='base_SMILES')
        feature_df = pd.merge(feature_df, self.solv_mordred, on='solvent_SMILES')
        feature_df = feature_df.drop(columns=['base_SMILES', 'solvent_SMILES'])
        
        
        #print(feature_df.columns)
        #print(latent_points.shape)
        #feature_df = pd.merge(feature_df, ligand_features, on='ligand_SMILES')
        #feature_df.to_csv('/home/ianlee/optimizer/OW_DOE/test3.csv')
        #feature_df.sort_values('reaction_index', inplace=True)
        #feature_df.reset_index(drop=True, inplace=True)  # 重設索引

        #null 先不處理
        """
        feature_df_nonull = feature_df.dropna()
        feature_df_nonull = feature_df_nonull.drop(columns=['base_SMILES', 'solvent_SMILES','ligand_SMILES'])
        feature_df_nonull = feature_df_nonull.apply(pd.to_numeric, errors='coerce')
        feature_df_nonull = feature_df_nonull.reindex(columns=self.predictor_feature[:-2])
        # print(feature_df_nonull[['Temp_C','Concentration']].head())
        deleted_rows = feature_df.index[~feature_df.index.isin(feature_df_nonull.index)].tolist()
        """
        
        feature_df_nonull = np.concatenate((latent_points, feature_df), axis=1)
        
        yield_results = []
        yield_mean_and_std = []
        for yield_prediction_model in self.ary_yield_models:
            yield_results.append(yield_prediction_model.predict(feature_df_nonull))
                
        for i in range(len(yield_results[0])):
            values = [prediction[i] for prediction in yield_results]
            yield_mean_and_std.append((np.mean(values), np.std(values)))
    
        #for position in deleted_rows:
            #yield_mean_and_std = yield_mean_and_std[:position] + [(1e-8, 0)] + yield_mean_and_std[position:]
            
        yield_mean_and_std = [tuple(row) for row in yield_mean_and_std]
        #feature_df = feature_df.assign(reation_yield = yield_mean_and_std)
        total_fitness = yield_mean_and_std[:]
        total_fitness = [row[0] for row in total_fitness]
        
        # print('ensemble_predict')
        # for smiles, fitness, yield_ in zip(feature_df['ligand_SMILES'], total_fitness, feature_df['reation_yield']):
        #     print(f'{smiles}\t{fitness}\t{yield_[0]}')
        # print('ensemble_predict')
        total_propertys = []
        # sys.exit()
        for i in range(len(total_particle_position)):
            properties = ExperimentProperty(
                Base = base_df[i],
                Solvent = solvent_df[i],
                Temperature = total_particle_position[i, self.latent_size].detach().cpu().numpy(),
                Concentration = total_particle_position[i, self.latent_size+1].detach().cpu().numpy(),
                reation_yield = yield_mean_and_std[i],
        )
            total_propertys.append(properties)
        return total_fitness, total_propertys, yield_mean_and_std