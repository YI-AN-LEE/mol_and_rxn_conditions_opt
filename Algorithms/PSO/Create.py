import torch
import random
import pickle
# import json
import datetime

from Algorithms.PSO.Particle import Particle
from typing import List
from Environments.PvkAdditives.lib.Pvk_Predictor import PvkTransform, Pvk_Ensemble_Predictor
from Environments.Direct_Arylation.lib.AryPredictor import AryTransform, AryEnsmblePredictor


def create_first_particle(vae_vector, process_cond, bounds, freeze, smile):
    vae_vector = vae_vector
    process_cond = torch.tensor(process_cond).to('cuda')
    initial_particle_position = torch.cat((vae_vector.float(), process_cond.float()), dim=1)
    return Particle(initial_particle_position[0], bounds, freeze, smile)

def create_pool(center_particle: Particle, bounds, freeze, radius, pop_size, transform:PvkTransform) -> List[Particle]:
    # Generate all initial particles' position.
    total_paticle_position = center_particle.position.clone() # torch.Size([43])
    total_paticle_position = torch.stack([total_paticle_position] * pop_size) # torch.Size([pop_size, 43])
    total_paticle_position, total_smiles = adjust_position_and_get_smiles(total_paticle_position, bounds, radius, transform)
    return [Particle(paticle_position, bounds, freeze, smiles) for paticle_position, smiles in zip(total_paticle_position, total_smiles)]

def create_tensor(center_particle: Particle, bounds, freeze, radius, pop_size, transform:PvkTransform) -> List[Particle]:
    # Generate all initial particles' position.
    today = datetime.date.today()
    today = today.strftime("%Y%m%d")
    total_paticle_position = center_particle.position.clone() # torch.Size([43])
    total_paticle_position = torch.stack([total_paticle_position] * pop_size) # torch.Size([pop_size, 43])
    total_paticle_position, total_smiles = adjust_position_and_get_smiles(total_paticle_position, bounds, radius, transform)
    return total_paticle_position, total_smiles

def sort_filter_and_reappend(pool:List[Particle], bounds, radius, pop_size, transform:PvkTransform) -> List[Particle]:
    # Step 1: Sort by 'smiles'
    pool = sorted(pool, key=lambda particle: particle.smiles)
    # Step 2: Count occurrences of 'smiles' and track corresponding 'particle' instances
    smiles_count = {}
    for particle in pool:
        smiles = particle.smiles
        if smiles in smiles_count:
            smiles_count[smiles].append(particle)
        else:
            smiles_count[smiles] = [particle]

    # Step 3: Filter and keep only top 1 'particle' instances for each repeated 'smiles'
    for smiles, small_pool in smiles_count.items():
        if len(small_pool) > 1:
            small_pool.sort(key=lambda particle: particle.best_fitness, reverse=True)
            smiles_count[smiles] = small_pool[:1]

    # Step 4: Reconstruct the sorted and filtered 'pool' list
    pool = []
    for small_pool in smiles_count.values():
        pool.extend(small_pool)
    pool = sorted(pool, key=lambda particle: particle.best_fitness, reverse=True)

    if pop_size == len(pool):
        return pool
    else:
        # Step 5: Append new particles by the global best particle position until the pool is full(equal the # we set)
        total_new_position = pool[0].best_position.clone() # torch.Size([43])
        total_new_position = torch.stack([total_new_position] * (pop_size - len(pool))) # torch.Size([pop_size, 43])
        total_new_position, total_new_smiles = adjust_position_and_get_smiles(total_new_position, bounds, radius, transform)
        return pool + [Particle(position = paticle_position, bounds = bounds, smiles = smiles) for paticle_position, smiles in zip(total_new_position, total_new_smiles)]

def adjust_position_and_get_smiles(total_paticle_position, bounds, radius, transform:PvkTransform):
    random_noise = torch.randn_like(total_paticle_position) * radius

    scaling_factors = torch.full((total_paticle_position.shape[1],), 2.0, device='cuda:0') # 2 is the std of the vae ls vector
    for i in range(-bounds.shape[0], 0):
        scaling_factors[i] = (bounds[i][0] - bounds[i][1]) / 2
    scaling_factors = torch.stack([scaling_factors] * total_paticle_position.shape[0])
    # print('scaling_factors', scaling_factors.shape,scaling_factors[0])
    # print('random_noise', random_noise.shape, random_noise)

    total_paticle_position = total_paticle_position + random_noise * scaling_factors
    
    for i in range(len(total_paticle_position)): # i for the index of particle
        for j in range(-1 * len(bounds), 0): # j for the index of the position
            if total_paticle_position[i][j] > bounds[j][0]:
                total_paticle_position[i][j] = bounds[j][0]
            elif total_paticle_position[i][j] < bounds[j][1]:
                total_paticle_position[i][j] = bounds[j][1]
        # if total_paticle_position[i][-1] > 0.5:
        #     total_paticle_position[i][-1] = 1
        # else:
        #     total_paticle_position[i][-1] = 0
    # Check for correct generation of initial smiles.
    total_smiles_position = total_paticle_position[:, :32]
    total_smiles = transform.get_smiles_from_position(total_smiles_position)
    while isinstance(total_smiles, list) and None in total_smiles:
        none_indicy = [index for index, smiles in enumerate(total_smiles) if smiles is None]
        none_position = total_smiles_position[none_indicy]
        random_noise = torch.randn_like(none_position) * radius
        scaling_factor = torch.tensor([random.uniform(0, 1) for _ in range(len(none_position))]).to('cuda')
        none_position = none_position + random_noise * scaling_factor.unsqueeze(1)
        none_smiles = transform.get_smiles_from_position(none_position)
        for idx, smiles_position, smiles in zip(none_indicy, none_position, none_smiles):
            if smiles is not None:
                total_smiles_position[idx] = smiles_position
                total_smiles[idx] = smiles
    total_paticle_position[:, :32] = total_smiles_position
    return total_paticle_position, total_smiles


def generate_initial_particle_pool(total_position_tensor, total_expt_df, bounds, pop_size):
    '''
    Generate initial solution pool by using the experiment tensor and dataframe.
    '''
    particle_pool = []
    for i in range(total_position_tensor.size(0)):
        particle_pool.append(Particle(total_position_tensor[i], total_expt_df.iloc[i],bounds))
    # particle_pool = sorted(particle_pool, key=lambda solution: solution.best_fitness, reverse=True)[:pop_size]
    particle_pool = random.sample(particle_pool, min(pop_size, len(particle_pool)))
    return particle_pool


def create_pool_v2(center_position, pop_size, bounds, radius, transform:PvkTransform, predictor:Pvk_Ensemble_Predictor) -> List[Particle]:
    #with torch.no_grad():
    #center_position[32:36] = torch.log(center_position[32:36])
    total_paticle_position = torch.stack([center_position] * pop_size)
    total_paticle_position, total_smiles = adjust_position_and_get_smiles(total_paticle_position, bounds, radius, transform)
    #feature_df = transform.to_dataframe(total_smiles, total_process_position, transform.pvk_feature_list)
    _ , total_propertys = predictor.ensemble_predict(total_paticle_position)
    
    pool = [Particle(position = position, bounds = bounds, smiles = smiles) for position, smiles in zip(total_paticle_position, total_smiles)]
    for particle, expt_property in zip(pool, total_propertys):
        particle.best_expt_property = expt_property
    return pool

# 20240615 ary datasets
def create_pool_v3(center_position, pop_size, bounds, radius, transform:AryTransform, predictor:AryEnsmblePredictor, proc_feature_list) -> List[Particle]:
    # Generate the initial pool.
    total_particle_position = torch.stack([center_position] * pop_size)
    total_particle_position, total_smiles = adjust_position_and_get_smiles(total_particle_position, bounds, radius, transform)
    total_process_position = total_particle_position[:, 32:]
    #feature_df = transform.to_dataframe(total_smiles, total_process_position, proc_feature_list)
    total_fitness, total_propertys = predictor.ensemble_predict(total_particle_position)
    pool = [Particle(position = position, bounds = bounds, smiles = smiles) for position, smiles in zip(total_particle_position, total_smiles)]

    for particle, expt_property, fitness in zip(pool, total_propertys, total_fitness):
        particle.best_expt_property = expt_property
        particle.best_fitness = fitness
        particle.best_smiles = particle.smiles
    return pool