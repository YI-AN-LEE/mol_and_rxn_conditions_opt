from Algorithms.PSO.Particle import Particle
from Environments.PvkAdditives.lib.Pvk_Predictor import PvkTransform, Pvk_Ensemble_Predictor
from Environments.Direct_Arylation.lib.AryPredictor import AryTransform, AryEnsmblePredictor
from typing import List
from tqdm import tqdm

import torch
import random

class ParticleSwarmOptimization:
    def __init__(
            self,
            pool: List[Particle], 
            max_iterations: int,
            transform: PvkTransform,
            predictor: Pvk_Ensemble_Predictor,
            generation,
            radius = 0.25,
            inertia_weight = 1,  # we can try another 0.45 1 0.5 0.25
            cognitive_weight = 2, #0.6 2.0 1.0 0.5 
            social_weight = 2,  #0.6 2.0 1.0 0.5, 
            mult = 0.125, #control 縮放倍率,
            EI = False,
            sto = False,
            f_best = None,
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pool = pool
        self.num_particle = len(pool)
        self.max_iterations = max_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.transform = transform
        self.predictor = predictor
        self.global_best_fitness = 1e-7
        self.global_best_position = None
        self.global_best_smiles = None
        self.global_best_expt_property = None
        self.EI = EI
        self.sto = sto
        self.f_best = f_best
        self.radius = radius
        self.generation = generation
        self.mult = mult

    def optimize(self):
        # Run the Optmization Iteration
        for iteration in range(self.max_iterations):
            self.evaluate_fitness()
            self.update_global_best()
            for particle in tqdm(self.pool, desc='Pool Progress'):
                particle.update_velocity(self.global_best_position, self.inertia_weight,self.cognitive_weight, self.social_weight)
                if self.sto is True:
                    particle.update_position_sto(self.mult)
                else:
                    particle.update_position(self.mult)
            self.ensure_particles_with_smiles()
            print(f'\tIteration {self.generation * self.max_iterations + iteration + 1}')
            print(f'Best  Smiles: {self.global_best_smiles}')
            print(f'Fitness: {self.global_best_fitness:.4e}')
            print(f'Experiment Property:')
            print(self.global_best_expt_property)

    def evaluate_fitness(self):
        '''
        Calculate the all particles' fitness and molecule (physcial) properties.
        Check that recent one is better or not.
        '''
        total_particle_position = self.transform.transform_psopool_to_feature(self.pool)
        if self.EI == True:
            total_fitness, total_expt_propertys = self.predictor.ensemble_predict_EI(total_particle_position, self.f_best)
        else:
            total_fitness, total_expt_propertys = self.predictor.ensemble_predict(total_particle_position)
        for particle, fitness, expt_porperty in zip(self.pool, total_fitness, total_expt_propertys):
            if fitness > particle.best_fitness:
                particle.best_smiles = particle.smiles
                particle.best_position = particle.position.clone()
                particle.best_fitness = fitness
                particle.best_expt_property = expt_porperty

    # def ensure_particles_with_smiles(self):
    #     '''
    #     Let particles that cannot obtain smiles find nearby locations where they can be decoded back to smlies
    #     '''
    #     total_smiles_position = torch.stack([particle.position[:32] for particle in self.pool])
    #     total_smiles = self.transform.get_smiles_from_position(total_smiles_position)
        
    #     while isinstance(total_smiles, list) and None in total_smiles:
    #         none_indicy = [index for index, smiles in enumerate(total_smiles) if smiles is None]
    #         none_position = total_smiles_position[none_indicy]
    #         random_noise = torch.randn_like(none_position) * self.radius
    #         scaling_factor = torch.tensor([random.uniform(0, 1) for _ in range(none_position)]).to('cuda:0')
    #         none_position = none_position + random_noise * scaling_factor.unsqueeze(1)
    #         none_smiles = self.transform.get_smiles_from_position(none_position)
    #         for idx, smiles_position, smiles in zip(none_indicy, none_position, none_smiles):
    #             if smiles is not None:
    #                 total_smiles_position[idx] = smiles_position
    #                 total_smiles[idx] = smiles
    #     for particle, smile_position, smiles in zip(self.pool, total_smiles_position, total_smiles):
    #         particle.position[:32] = smile_position
    #         particle.smiles = smiles

    def ensure_particles_with_smiles(self):
        total_smiles_position = torch.stack([particle.position[:32] for particle in self.pool])
        total_smiles = self.transform.get_smiles_from_position(total_smiles_position)
        while isinstance(total_smiles, list) and None in total_smiles:
            none_indicy = [index for index, smiles in enumerate(total_smiles) if smiles is None]
            none_position = total_smiles_position[none_indicy]
            random_noise = torch.randn_like(none_position) * self.radius
            scaling_factor = torch.tensor([random.uniform(0, 1) for _ in range(len(none_position))]).to(self.device)
            none_position = none_position + random_noise * scaling_factor.unsqueeze(1)
            none_smiles = self.transform.get_smiles_from_position(none_position)
            
            for idx, smiles_position, smiles in zip(none_indicy, none_position, none_smiles):
                if smiles is not None:
                    total_smiles_position[idx] = smiles_position
                    total_smiles[idx] = smiles
        for particle, smile_position, smiles in zip(self.pool, total_smiles_position, total_smiles):
            particle.position[:32] = smile_position
            particle.smiles = smiles

    def update_global_best(self):
        global_best_particle = max(self.pool, key=lambda particle: particle.best_fitness)
        self.global_best_fitness = global_best_particle.best_fitness
        self.global_best_position = global_best_particle.best_position
        self.global_best_smiles = global_best_particle.best_smiles
        self.global_best_expt_property = global_best_particle.best_expt_property
