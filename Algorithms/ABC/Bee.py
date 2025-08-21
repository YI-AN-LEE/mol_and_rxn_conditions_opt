import torch
import pandas as pd

# self-made package
#
#from Environments.PvkAdditives.lib.Pvk_Types import ExperimentProperty
from Environments.Direct_Arylation.lib.AryTypes import ExperimentProperty

class Bee:
    def __init__(self,
                 position: torch.Tensor,
                 bounds: torch.Tensor = None,
                 expt_df: pd.DataFrame = None,
                #  smiles: str = None,q
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trail = 0
        self.radius = 0.25  #origin = 1.0, pvk: 0.5, aryal: 0.25
        self.bounds = bounds
        # self.binary = [-1]

        self.position = position
        if expt_df is not None:
            self.expt_property = ExperimentProperty(
                ABC  = expt_df['ABC'],
                Base = expt_df['Base'],
                Solvent = expt_df['Solvent'],
                SpAbs_A = expt_df['SpAbs_A'],
                AATS0dv = expt_df['AATS0dv'],
                Temperature = expt_df['Temperature'],
                Concentration = expt_df['Concentration'],
                reation_yield = (expt_df['reation_yield'][0], expt_df['reation_yield'][1]) if isinstance(expt_df['reation_yield'], tuple) else (expt_df['reation_yield'], 0),
            )
            self.smiles = expt_df['SMILES']
        else:
            self.expt_property :ExperimentProperty = None
            self.smiles = None
        
        self.fitness = 1e-8
        self.candidate_position = None
        self.candidate_smiles = None

    def increase_trail(self):
        self.trail += 1

    def reset_trail(self):
        self.trail = 0

    def temporarily_reduce_radius(self):
        self.radius = 0.1

    def get_candidate_position(self, others_position: torch.Tensor):
        self.candidate_position = self.position + torch.empty_like(self.position).uniform_(-1 * self.radius, self.radius) * (self.position - others_position) 
        #elf.candidate_position = self.position + torch.empty_like(self.position).uniform_(0, self.radius) * (self.position - others_position)       
        for i in range(-1* len(self.bounds), 0):
            if self.candidate_position[i] > self.bounds[i][0]: # upper bound
                excess = self.candidate_position[i] - self.bounds[i][0] # original method
                self.candidate_position[i] -= 2 * excess # original method
                if self.candidate_position[i] > self.bounds[i][0]:
                    self.candidate_position[i] = self.bounds[i][0]

            elif self.candidate_position[i] < self.bounds[i][1]: # lower bound
                deficit = self.bounds[i][1] - self.candidate_position[i] # original method
                self.candidate_position[i] += 2 * deficit # original method
                if self.candidate_position[i] < self.bounds[i][1]:
                    self.candidate_position[i] = self.bounds[i][1]

        # # Use regression with a cutoff to get the binary feature
        # for index in self.binary:
        #     if self.candidate_position[index] > 0.5: # here sould be change in the future for the binary feature involving
        #         self.candidate_position[index] = 1
        #     elif self.candidate_position[index] < 0.5:
        #         self.candidate_position[index] = 0