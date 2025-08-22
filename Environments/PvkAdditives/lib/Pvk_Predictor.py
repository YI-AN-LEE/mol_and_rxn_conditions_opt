import os
import glob
import pickle
import torch
import re
from scipy.stats import norm
from xgboost import XGBRegressor
from typing import List
#from hgraph import HierVAE
from rdkit import Chem
from rdkit.Chem import PandasTools
import numpy as np
import pandas as pd

from Environments.PvkAdditives.lib.Pvk_Types import ExperimentProperty
from Algorithms.PSO.Particle import Particle
from Algorithms.SA.Solution import Solution
from Algorithms.ABC.Bee import Bee
from Algorithms.GA.Individual import Individual

from fast_jtnn import JTNNVAE

class Pvk_Ensemble_Predictor():
    
    #change the predictor to input of latent points and the corresponding reaction conditions
    def __init__(self, JTVAE, model_path, latent_size, maximize=True):
        self.model_path = model_path
        self.feature_df: pd.DataFrame = None
        self.pvk_size_models: List[XGBRegressor] = []
        self.pvk_size_feature_list = ['Reagent1 (ul)','Reagent2 (ul)','Reagent3 (ul)','Reagent4 (ul)','lab_code']
        self.load_model()
        self.latent_size = latent_size
        self.JTVAE : JTNNVAE = JTVAE
        self.maximize = maximize
        
    def load_model(self):
        #load the xgb models
        max_num = 0
        pattern = r'^(.*)_c(\d+)\.pkl$'

        pvk_size_model_paths = glob.glob(os.path.join(self.model_path, "*pvk_size_xgboost*.pkl"))
        #print(pvk_size_model_paths)
        
        """
        for xgb in pvk_size_model_paths:
            match = re.match(pattern, xgb)
            _, group_num = match.groups()
            if int(group_num) > max_num:
                max_num = int(group_num) 
        pvk_size_model_paths = glob.glob(os.path.join(self.model_path, f"*c{max_num}*.pkl"))
        """
        
        for path in pvk_size_model_paths:
            with open(path, "rb") as f:
                model = pickle.load(f)
            self.pvk_size_models.append(model)

    def ensemble_predict(self, total_particle_position):

        
        #result prediction
        feature = total_particle_position.detach().cpu().numpy()
        size_results = []
        for pvk_size_model in self.pvk_size_models:
            #predict the values here
            size_results.append(pvk_size_model.predict(feature))
            
        size_mean_and_std = []
        fitness = []
        for i in range(len(size_results[0])):
            values = [prediction[i] for prediction in size_results]
            size_mean_and_std.append((np.mean(values), np.std(values)))
            fitness.append(np.mean(values))
        # handle the nan values later
        
        #feature_df = feature_df.assign(crystal_size=size_mean_and_std)
        total_propertys = []
        for i in range(len(feature)):
            properties = ExperimentProperty(
                Reagent1=feature[i, 32],
                Reagent2=feature[i, 33],
                Reagent3=feature[i, 34],
                Reagent4=feature[i, 35],
                lab_code=feature[i, 36],
                crystal_size=size_mean_and_std[i],
        )
            total_propertys.append(properties)
        #return {'crystal_size':size_mean_and_std}, total_propertys
        return fitness, total_propertys
    
    def ensemble_predict_BO(self, total_particle_position):
        # 前面會送一個numpy array進來
        
        #result prediction
        feature = total_particle_position.detach().cpu().numpy()
        size_results = []
        for pvk_size_model in self.pvk_size_models:
            #predict the values here
            size_results.append(pvk_size_model.predict(feature))
            
        size_mean_and_std = []
        fitness = []
        for i in range(len(size_results[0])):
            values = [prediction[i] for prediction in size_results]
            size_mean_and_std.append((np.mean(values), np.std(values)))
            fitness.append(np.mean(values))
        # handle the nan values later
        
        #feature_df = feature_df.assign(crystal_size=size_mean_and_std)
        total_propertys = []
        for i in range(len(feature)):
            properties = ExperimentProperty(
                Reagent1=feature[i, 32],
                Reagent2=feature[i, 33],
                Reagent3=feature[i, 34],
                Reagent4=feature[i, 35],
                lab_code=feature[i, 36],
                crystal_size=size_mean_and_std[i],
        )
            total_propertys.append(properties)
        #return {'crystal_size':size_mean_and_std}, total_propertys
        return fitness, total_propertys, size_mean_and_std

    def expected_improvement(self, mu, sigma, f_best, xi=0.01):
        improvement = mu - f_best - xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        scale = 10e08
        return ei*scale

    def ensemble_predict_EI(self, total_particle_position, f_best):
        
        #result prediction
        feature = total_particle_position.detach().cpu().numpy()
        size_results = []
        for pvk_size_model in self.pvk_size_models:
            #predict the values here
            size_results.append(pvk_size_model.predict(feature))
            
        size_mean_and_std = []
        fitness = []
        for i in range(len(size_results[0])):
            values = [prediction[i] for prediction in size_results]
            size_mean_and_std.append((np.mean(values), np.std(values)))
            fitness.append(self.expected_improvement(np.mean(values),
                            np.std(values),f_best))

        # handle the nan values later
        
        #feature_df = feature_df.assign(crystal_size=size_mean_and_std)
        total_propertys = []
        for i in range(len(feature)):
            properties = ExperimentProperty(
                Reagent1=feature[i, 32],
                Reagent2=feature[i, 33],
                Reagent3=feature[i, 34],
                Reagent4=feature[i, 35],
                lab_code=feature[i, 36],
                crystal_size=size_mean_and_std[i],
        )
            total_propertys.append(properties)
        #return {'crystal_size':size_mean_and_std}, total_propertys
        return fitness, total_propertys

class PvkTransform():
    def __init__(self, JTVAE, latent_size) -> None:
        #self.hier_vae : HierVAE = hier_vae
        self.JTVAE : JTNNVAE = JTVAE
        self.pvk_feature_list = ['Reagent1 (ul)','Reagent2 (ul)','Reagent3 (ul)','Reagent4 (ul)','lab_code']
        self.latent_size = latent_size

    def get_smiles_from_position(self, positions):
        '''
        Decode the SMILES from the latent points in the HierVAE model. Sometimes the decoding may fail.
        :param positions: A list of torch tensors of latent points.
        :return: A list of decoded SMILES.
        '''
        
        #JTVAE always decode something, no need to try and except here
        
        smiles_list = []
        for i in range(len(positions)):
            tree_vec,mol_vec = torch.split(torch.reshape(positions[i], (1,32)), 16, dim=1)
            smiles = self.JTVAE.decode(tree_vec, mol_vec, prob_decode=False)
            smiles_list.append(smiles) 
        return [self.cannonicalize_smiles(smiles) for smiles in smiles_list]
        
        
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
        process_tensor_cpu = process_tensor.detach().cpu()
        feature_df = pd.DataFrame(data=smiles_list, columns=['SMILES'])
        feature_df = pd.concat([feature_df, pd.DataFrame(data=process_tensor_cpu.numpy(), columns=column_names)], axis=1)
        return feature_df
    
    def exp_tensor(self, tensor, dim=None):
        if dim is None:
            return torch.exp(tensor)
        else:
            return torch.exp(tensor[:, dim[0]:dim[1]])
    
    def transform_psopool_to_feature(self, pool:List[Particle]):
        #total_smile_name = [particle.smiles for particle in pool]
        total_position = [particle.position for particle in pool]
        total_position = torch.stack(total_position, dim=0)
        #tensor_exp = self.exp_tensor(total_position, dim=(self.latent_size, self.latent_size+4))
        #total_position[:, self.latent_size:self.latent_size+4] = tensor_exp
        #total_process_position = total_position[:, self.latent_size:]
        return total_position
    
    def transform_sapool_to_feature(self, solution_pool:List[Solution]):
        total_smile_name = [solution.candidate_smiles for solution in solution_pool]
        total_position = [solution.candidate_position for solution in solution_pool]
        total_position = torch.stack(total_position, dim=0)
        #total_process_position = total_position[:, self.latent_size:]
        #return self.to_dataframe(total_smile_name, total_process_position, self.pvk_feature_list)
        return total_position
    
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
        return self.to_dataframe(total_smile_name, total_process_position, self.pvk_feature_list)
    
