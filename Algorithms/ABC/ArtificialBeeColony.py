from Algorithms.ABC.Bee import Bee
from Environments.PvkAdditives.lib.Pvk_Predictor import PvkTransform, Pvk_Ensemble_Predictor
from Environments.Direct_Arylation.lib.AryPredictor import AryTransform, AryEnsmblePredictor
import re
import torch
import random

from typing import List
import numpy as np
from fast_jtnn import *

class ArtificialBeeColony:
    def __init__(self, dataset, transform, predictor, employed_bees, onlooker_bees, max_trails, max_iterations, bounds, latent_size, JTVAE, EI = False, f_best = None, radius = 0.25): #radius, try 1, 0.8, 0.5(pvk), if aryl, try 0.25
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_trails = max_trails
        self.max_iterations = max_iterations
        self.radius = radius
        self.EI = EI
        self.f_best = f_best
        if dataset =='pvk_additives':
            self.transform : PvkTransform = transform
            self.predictor : Pvk_Ensemble_Predictor = predictor
        if dataset =='direct_arylation':
            self.transform : AryTransform = transform
            self.predictor : AryEnsmblePredictor = predictor
        self.employed_bees: List[Bee] = employed_bees  #the initial bee object
        self.onlooker_bees: List[Bee] = onlooker_bees  
        self.employees_phase = False
        self.onlookers_phase = False
        self.latent_size = latent_size
        #look for more far away bounds (molecule) 4 --> 10
        #final bounds include molecules and process conditions !!
        self.bounds = torch.cat((torch.stack([torch.full((latent_size,), 5, dtype=torch.float32), torch.full((latent_size,), -5, dtype=torch.float32)], dim=1), bounds), dim=0)
        self.bounds = self.bounds.to(torch.float32)
        self.bounds = self.bounds.to(self.device)
        self.JTVAE: JTNNVAE = JTVAE

    def employed_phase(self):
        for bee in self.employed_bees:
            if bee.trail < self.max_trails:
                # choose a random bee that is not the current bee, j is the index of the random bee
                j = np.random.randint(0, len(self.employed_bees))
                fail = 0
                while j == self.employed_bees.index(bee):
                    j = np.random.randint(0, len(self.employed_bees))
                    fail+=1
                    if fail > 100:
                        print('find the same bee 100 times')
                        break
                    #may stuck here, which is ,always find the same bee ._.
                bee.get_candidate_position(self.employed_bees[j].position)
                #sometime the loop cannot get out
        with TemporarilySetAttribute(self, 'employees_phase', True):
            self.ensure_bee_with_smiles()
            self.evaluate_fitness_and_trail()

    def onlooker_phase(self):
        probabilities = np.array([bee.fitness for bee in self.employed_bees])
        probabilities = probabilities / np.sum(probabilities)
        for bee in self.onlooker_bees:
            chosen_bee = np.random.choice(self.employed_bees, p=probabilities)
            if bee.trail < self.max_trails:
                # set a function to control the radius with time (when iteration 50 use 1/2 radius)
                # if iteration = 50:
                #     bee.radius = bee.radius/2
                with TemporarilySetAttribute(bee, 'radius', bee.radius * random.uniform(0, 1)):
                    bee.get_candidate_position(chosen_bee.position)
        with TemporarilySetAttribute(self, 'onlookers_phase', True):
            self.ensure_bee_with_smiles()
            self.evaluate_fitness_and_trail()

    def scout_phase(self):
        for bee in self.employed_bees:
            if bee.trail >= self.max_trails:
                bee.position = torch.rand(len(self.bounds), device=self.device) * (self.bounds[:, 0] - self.bounds[:, 1]) + self.bounds[:, 1]
                # print(bee.position)
                bee.reset_trail()
        #with TemporarilySetAttribute(self, 'scout_phase', True):
             #self.ensure_bee_with_smiles()
             #self.evaluate_fitness_and_trail()

    def exchange_employee_onlooker(self):
        bee_swarm = self.employed_bees + self.onlooker_bees
        bee_swarm = sorted(bee_swarm, key=lambda bee: bee.fitness, reverse=True)
        half_length = len(bee_swarm) // 2
        self.employed_bees = bee_swarm[:half_length]
        self.onlooker_bees = bee_swarm[half_length:]
        
 

    def ensure_bee_with_smiles(self):
        '''
        Let employed bee that cannot obtain smiles find nearby locations where they can be decoded back to smlies
        '''

        '''
        now some of the molecules cannot be encoded, we need to filter those molecules
        we decode them and encode again until we get a valid molecule
        '''
        
        if self.employees_phase:
            total_smiles_position = torch.stack([bee.candidate_position[:self.latent_size] for bee in self.employed_bees])
        if self.onlookers_phase:
            total_smiles_position = torch.stack([bee.candidate_position[:self.latent_size] for bee in self.onlooker_bees])
        total_smiles = self.transform.get_smiles_from_position(total_smiles_position)
        counts = 0
        
        #firt time filter: put the SMILES that have error to none
        """
        for i in range(len(total_smiles)):
            try:
                #see if any kekulize or encoding error exists
                self.JTVAE.encode_latent_mean([total_smiles[i]])
            except:
                total_smiles[i] = None
        """
        
        while (isinstance(total_smiles, list) and None in total_smiles) : #or any(re.search(r'\[.*-\]', s) or '+' in s for s in total_smiles):
            print('rebuild again')
            
            none_indicy = [index for index, smiles in enumerate(total_smiles) if smiles is None]
            if counts < 20:
                none_position = total_smiles_position[none_indicy]
                random_noise = torch.randn_like(none_position) * self.radius
                scaling_factor = torch.tensor([random.uniform(0, 1) for _ in range(len(none_position))]).to(self.device)
                none_position = none_position + random_noise * scaling_factor.unsqueeze(1)
                none_smiles = self.transform.get_smiles_from_position(none_position)
            else:
                none_position = total_smiles_position[none_indicy]
                none_position = torch.rand_like(none_position) #*2-1
                none_smiles = self.transform.get_smiles_from_position(none_position)
            for idx, smiles_position, smiles in zip(none_indicy, none_position, none_smiles):
                if smiles is not None:
                    total_smiles_position[idx] = smiles_position
                    total_smiles[idx] = smiles
            
            """
            for j in range(len(total_smiles)):
                try:
                    #see if any kekulize or encoding error exists
                    self.JTVAE.encode_latent_mean([total_smiles[j]])
                except:
                    total_smiles[j] = None
            """
            counts += 1
            
        if self.employees_phase:
            # print('employees_phase# employees_phase')
            for bee, smile_position, smiles in zip(self.employed_bees, total_smiles_position, total_smiles):
                bee.candidate_position[:self.latent_size] = smile_position
                # print(bee.candidate_smiles, smiles)
                bee.candidate_smiles = smiles
            # print('employees_phase& employees_phase')
        if self.onlookers_phase:
            # print('onlookers_phase# onlookers_phase')
            for bee, smile_position, smiles in zip(self.onlooker_bees, total_smiles_position, total_smiles):
                bee.candidate_position[:self.latent_size] = smile_position
                # print(bee.candidate_smiles, smiles)
                bee.candidate_smiles = smiles
            # print('onlookers_phase& onlookers_phase')

    def evaluate_fitness_and_trail(self):
        if self.employees_phase:
            total_particle_position = self.transform.transform_beeswarm_to_feature(self.employed_bees)
            if EI == True:
                total_fitness , total_expt_propertys = self.predictor.ensemble_predict_EI(total_particle_position, self.f_best)
            else:
                total_fitness , total_expt_propertys = self.predictor.ensemble_predict(total_particle_position)
            total_fitness = total_fitness
            for bee, fitness, expt_porperty in zip(self.employed_bees, total_fitness, total_expt_propertys):
                if fitness > bee.fitness:
                    bee.position = bee.candidate_position
                    bee.fitness = fitness
                    bee.expt_property = expt_porperty
                    bee.smiles = bee.candidate_smiles
                    # print('$$$$$$$$$$$$$$$')
                    # print('fitness is about',fitness)
                    # print(bee.smiles)
                    # print(expt_porperty)
                    # print('@@@@@@@@@@@@@@@')
                    bee.reset_trail()
                else:
                    bee.increase_trail()
            # print('Employed')
            # for i, bee in enumerate(self.employed_bees):
                # print(f'{bee.smiles}\t{bee.fitness}\t{bee.expt_property.reation_yield[0]}')
            self.employed_bees = sorted(self.employed_bees, key=lambda bee: bee.fitness, reverse=True)
            


        if self.onlookers_phase:
            total_particle_position = self.transform.transform_beeswarm_to_feature(self.onlooker_bees)
            if EI == True:
                total_fitness , total_expt_propertys = self.predictor.ensemble_predict_EI(total_particle_position, self.f_best)
            else:
                total_fitness , total_expt_propertys = self.predictor.ensemble_predict(total_particle_position)
            total_fitness = total_fitness
            for bee, fitness, expt_porperty in zip(self.onlooker_bees, total_fitness, total_expt_propertys):
                if fitness > bee.fitness:
                    bee.position = bee.candidate_position
                    bee.fitness = fitness
                    bee.expt_property = expt_porperty
                    bee.smiles = bee.candidate_smiles
                    # print('***************')
                    # print(fitness)
                    # print(bee.smiles)
                    # print(expt_porperty)
                    # print('###############')
                    bee.reset_trail()
                else:
                    bee.increase_trail()
            # print('Onlooker')
            # for i, bee in enumerate(self.onlooker_bees):
                # print(f'{bee.smiles}\t{bee.fitness}\t{bee.expt_property.reation_yield[0]}')
            self.onlooker_bees = sorted(self.onlooker_bees, key=lambda bee: bee.fitness, reverse=True)        

    def run(self):
        for iteration in range(self.max_iterations):
            self.employed_phase()
            print('Employed Phase Done')
            self.onlooker_phase()
            print('Onlooker Phase Done')
            self.scout_phase()
            print('Scout Phase Done')
            self.exchange_employee_onlooker()
            print('Exchange Done')
            
            print(f'Iteration {iteration + 1}')
            print(f'Best Smiles: {self.employed_bees[0].smiles}')
            print(f'Fitness: {self.employed_bees[0].fitness:.4e}')
            print(f'Experiment Property:\n{self.employed_bees[0].expt_property}')
            
            print('new employed vs onlooker distribution')

        
                

    def show_results(self):
        bee_swarm = self.employed_bees + self.onlooker_bees
        bee_swarm = sorted(bee_swarm, key=lambda bee: bee.fitness, reverse=True)
        for index, bee in enumerate(bee_swarm):
            print(f'\tRank {index + 1}')
            print(f'Smiles: {bee.smiles}')
            print(f'Fitness: {bee.fitness:.4e}')
            print(f'Experiment Property:')
            print(bee.expt_property)

class TemporarilySetAttribute:
    def __init__(self, obj, attr, value):
        self.obj = obj
        self.attr = attr
        self.value = value
        self.original_value = None

    def __enter__(self):
        self.original_value = getattr(self.obj, self.attr)
        setattr(self.obj, self.attr, self.value)

    def __exit__(self, type, value, traceback):
        setattr(self.obj, self.attr, self.original_value)