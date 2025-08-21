import random
import torch

import pandas as pd

# from Environments.PvkAdditives.lib.Pvk_Types import ExperimentProperty
from Environments.Direct_Arylation.lib.AryTypes import ExperimentProperty

class Particle:
    def __init__(
            self,
            position: torch.Tensor,
            #expt_df: pd.DataFrame = None,
            bounds: torch.Tensor = None,
            latent_size = 32,
            smiles: str = None,
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bounds = bounds
        self.bounds = torch.cat((torch.stack([torch.full((latent_size,), 5, dtype=torch.float32), torch.full((latent_size,), -5, dtype=torch.float32)], dim=1), bounds), dim=0)
        self.bounds = self.bounds.to(torch.float32)
        self.bounds = self.bounds.to(self.device)
        self.velocity = torch.tensor([random.uniform(-1, 1) for _ in range(len(position))]).to('cuda')
        
        """
        if expt_df is not None:
            self.best_expt_property = ExperimentProperty(
                ABC  = expt_df['ABC'],
                Base = expt_df['Base'],
                Solvent = expt_df['Solvent'],
                SpAbs_A = expt_df['SpAbs_A'],
                AATS0dv = expt_df['AATS0dv'],
                Temperature = expt_df['Temperature'],
                Concentration = expt_df['Concentration'],
                reation_yield = (expt_df['reation_yield'][0], expt_df['reation_yield'][1]) if isinstance(expt_df['reation_yield'], tuple) else (expt_df['reation_yield'], 0),
            )
            self.best_smiles = expt_df['SMILES']
        """
        
        self.best_expt_property = None
        self.best_smiles = None
        
        self.scaling_factors = (self.bounds[:, 1] - self.bounds[:, 0]) / 2.0  # shape: (dim,)
            
        self.best_fitness = 1e-8
        self.position = position
        
        if self.best_smiles is not None:
            self.smiles = self.best_smiles
        else:
            self.smiles = smiles
        self.best_position = self.position.clone().detach()
    
    def update_velocity(self, global_best_position, inertia_weight, cognitive_weight, social_weight):
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        cognitive_component = cognitive_weight * r1 * (self.best_position - self.position) # personal best
        social_component = social_weight * r2 * (global_best_position - self.position) # global best
        self.velocity = inertia_weight * self.velocity + cognitive_component + social_component
        self.velocity = self.velocity
        # self.velocity[:32] = 0

    def update_position_sto(self, mult):
        # 取得 scaling_factors：每一維度的範圍除以 2
        # 擴展成跟 position 同樣大小
        #scaling_factors = scaling_factors.unsqueeze(0).expand_as(self.position)  # shape: (num_particles, dim)

        epsilon_vector = torch.ones_like(self.position)

        # 對 VAE 維度設小一點的 epsilon
        vae_dim = 37  # 假設前128維是分子 latent space
        epsilon_vector[:vae_dim] = 0.01
        epsilon_vector[vae_dim:] = 0.25

        # 每個維度都有自己的 scaling factor 與 epsilon
        stochastic_noise = epsilon_vector * self.scaling_factors * torch.randn_like(self.position)

        # 更新位置
        self.position = self.position + self.velocity * mult + stochastic_noise


        # Check that the positions are within bounds using while loop
        for i in range(-1 * len(self.bounds), 0):
            while True:
                if self.position[i] > self.bounds[i][0]:  # upper bound
                    excess = self.position[i] - self.bounds[i][0]
                    self.position[i] -= 2 * excess
                    # 防止浮點誤差，若仍超出，再修正一次
                    if self.position[i] > self.bounds[i][0]:
                        self.position[i] = self.bounds[i][0]
                elif self.position[i] < self.bounds[i][1]:  # lower bound
                    deficit = self.bounds[i][1] - self.position[i]
                    self.position[i] += 2 * deficit
                    if self.position[i] < self.bounds[i][1]:
                        self.position[i] = self.bounds[i][1]
                else:
                    # 在範圍內，跳出 while
                    break
                # print(f'{self.position[i]} should > {self.bounds[i][1]}')
        # if self.position[-1] > 0.5: # here sould be change in the future for the binary feature involving
        #     self.position[-1] = 1
        # elif self.position[-1] < 0.5:
        #     self.position[-1] = 0
            

    def update_position(self, mult):
        self.position = self.position + self.velocity * mult 

        # Check that the positions are within bounds using while loop
        for i in range(-1 * len(self.bounds), 0):
            while True:
                if self.position[i] > self.bounds[i][0]:  # upper bound
                    excess = self.position[i] - self.bounds[i][0]
                    self.position[i] -= 2 * excess
                    # 防止浮點誤差，若仍超出，再修正一次
                    if self.position[i] > self.bounds[i][0]:
                        self.position[i] = self.bounds[i][0]
                elif self.position[i] < self.bounds[i][1]:  # lower bound
                    deficit = self.bounds[i][1] - self.position[i]
                    self.position[i] += 2 * deficit
                    if self.position[i] < self.bounds[i][1]:
                        self.position[i] = self.bounds[i][1]
                else:
                    # 在範圍內，跳出 while
                    break
                # print(f'{self.position[i]} should > {self.bounds[i][1]}')
        # if self.position[-1] > 0.5: # here sould be change in the future for the binary feature involving
        #     self.position[-1] = 1
        # elif self.position[-1] < 0.5:
        #     self.position[-1] = 0
            