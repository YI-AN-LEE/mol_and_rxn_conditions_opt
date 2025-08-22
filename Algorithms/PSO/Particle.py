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

    def update_position(self, mult):
        self.position = self.position + self.velocity * mult 

        # Check that the positions are within bounds using while loop
        for i in range(-1 * len(self.bounds), 0):
            while True:
                if self.position[i] > self.bounds[i][0]:  # upper bound
                    excess = self.position[i] - self.bounds[i][0]
                    self.position[i] -= 2 * excess
                    if self.position[i] > self.bounds[i][0]:
                        self.position[i] = self.bounds[i][0]
                elif self.position[i] < self.bounds[i][1]:  # lower bound
                    deficit = self.bounds[i][1] - self.position[i]
                    self.position[i] += 2 * deficit
                    if self.position[i] < self.bounds[i][1]:
                        self.position[i] = self.bounds[i][1]
                else:
                    break

            