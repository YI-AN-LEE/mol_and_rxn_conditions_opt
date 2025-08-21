# import os
import re
import torch
import random
import pandas as pd

from Algorithms.ABC.Bee import Bee
from Environments.PvkAdditives.lib.Pvk_Predictor import PvkTransform, Pvk_Ensemble_Predictor
from fast_jtnn import *
from Environments.Direct_Arylation.lib.AryPredictor import AryTransform, AryEnsmblePredictor


def generate_initial_bee_swarm(total_position_tensor: torch.Tensor, total_expt_df: pd.DataFrame, pop_size):
    '''
    Generate initial bee swarm by using the experiment tensor and dataframe.
    '''
    bee_swarm = []
    for i in range(total_position_tensor.size(0)):
        bee_swarm.append(Bee(total_position_tensor[i], total_expt_df.iloc[i]))
    bee_swarm = sorted(bee_swarm, key=lambda bee: bee.fitness, reverse=True)[:pop_size]
    
    half_size = pop_size // 2
    return bee_swarm[:half_size], bee_swarm[half_size:]

# 20240322

def adjust_position_and_get_smiles(total_paticle_position, bounds, radius, transform:PvkTransform, latent_size, JTVAE):
    random_noise = torch.randn_like(total_paticle_position) * radius #radius is the std of the noise
    # scaling_factors = torch.tensor([random.uniform(0, 1) for _ in range(len(total_paticle_position))]).to('cuda:0')
    scaling_factors = torch.full((total_paticle_position.shape[1],), 2.0, device='cuda:0') # 2 is the std of the vae ls vector
    for i in range(-bounds.shape[0], 0):
        scaling_factors[i] = (bounds[i][0] - bounds[i][1]) / 2
    scaling_factors = torch.stack([scaling_factors] * total_paticle_position.shape[0])
    # print('scaling_factors', scaling_factors.shape,scaling_factors[0])
    # print('random_noise', random_noise.shape, random_noise)

    total_paticle_position = total_paticle_position + random_noise * scaling_factors
    # print('shape of total_paticle_position', total_paticle_position.shape)
    # Check the range of all initial process condition.
    # print('bounds bounds bounds', bounds)
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
    
    total_smiles_position = total_paticle_position[:, :latent_size]
    total_smiles = transform.get_smiles_from_position(total_smiles_position)  #get the SMILES list
    
    """
    #firt time filter: put the SMILES that have error to none
    for i in range(len(total_smiles)):
        try:
            #see if any kekulize or encoding error exists
            JTVAE.encode_latent_mean([total_smiles[i]])
        except:
            total_smiles[i] = None
    """
    
    counts = 0
    while (isinstance(total_smiles, list) and None in total_smiles) :#or any(re.search(r'\[.*-\]', s) or '+' in s for s in total_smiles):
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
        """
        for j in range(len(total_smiles)):
            try:
                #see if any kekulize or encoding error exists
                JTVAE.encode_latent_mean([total_smiles[j]])
            except:
                total_smiles[j] = None
        """
        counts += 1
    total_paticle_position[:, :latent_size] = total_smiles_position
    return total_paticle_position, total_smiles

def adjust_position_and_get_smiles_aryl(total_paticle_position, bounds, radius, transform:AryTransform, latent_size, JTVAE):
    random_noise = torch.randn_like(total_paticle_position) * radius #radius is the std of the noise
    # scaling_factors = torch.tensor([random.uniform(0, 1) for _ in range(len(total_paticle_position))]).to('cuda:0')
    scaling_factors = torch.full((total_paticle_position.shape[1],), 2.0, device='cuda:0') # 2 is the std of the vae ls vector
    for i in range(-bounds.shape[0], 0):
        scaling_factors[i] = (bounds[i][0] - bounds[i][1]) / 2
    scaling_factors = torch.stack([scaling_factors] * total_paticle_position.shape[0])
    # print('scaling_factors', scaling_factors.shape,scaling_factors[0])
    # print('random_noise', random_noise.shape, random_noise)

    total_paticle_position = total_paticle_position + random_noise * scaling_factors
    # print('shape of total_paticle_position', total_paticle_position.shape)
    # Check the range of all initial process condition.
    # print('bounds bounds bounds', bounds)
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
    
    total_smiles_position = total_paticle_position[:, :latent_size]
    total_smiles = transform.get_smiles_from_position(total_smiles_position)  #get the SMILES list
    
    """
    #firt time filter: put the SMILES that have error to none
    for i in range(len(total_smiles)):
        try:
            #see if any kekulize or encoding error exists
            JTVAE.encode_latent_mean([total_smiles[i]])
        except:
            total_smiles[i] = None
    """
    
    counts = 0
    while (isinstance(total_smiles, list) and None in total_smiles) :#or any(re.search(r'\[.*-\]', s) or '+' in s for s in total_smiles):
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
        """
        for j in range(len(total_smiles)):
            try:
                #see if any kekulize or encoding error exists
                JTVAE.encode_latent_mean([total_smiles[j]])
            except:
                total_smiles[j] = None
        """
        counts += 1
    total_paticle_position[:, :latent_size] = total_smiles_position
    return total_paticle_position, total_smiles


def create_swarm_abc(center_position, pop_size, bounds, radius, transform:PvkTransform, predictor:Pvk_Ensemble_Predictor, proc_feature_list, latent_size, JTVAE):
    # Generate the initial swarm.
    total_particle_position = torch.stack([center_position] * pop_size) #大大向量
    print('Finished position generating')
    print('Start generating inital SMILES')
    total_particle_position, total_smiles = adjust_position_and_get_smiles(total_particle_position, bounds, radius, transform, latent_size, JTVAE) #add noise and decode
    #print(total_smiles)
    #total_process_position = total_particle_position[:, latent_size:]
    #total_latent_position = total_particle_position[:, :latent_size]
    #feature_df = transform.to_dataframe(total_smiles, total_latent_position, total_process_position, proc_feature_list)
    #maybe another SMILES filter should be added here
    total_fitness, total_propertys = predictor.ensemble_predict(total_particle_position) #get the vector of all fitness and propertys
    swarm = [Bee(position, bounds) for position in total_particle_position]

    # for property, smiles, fitness in zip(total_propertys, total_smiles, total_fitness):
    #     print(smiles)
    #     print(property.reation_yield)
    #     print(fitness)
    #     print('\n')

    for bee, expt_property, smiles, fitness in zip(swarm, total_propertys, total_smiles, total_fitness):
        bee.expt_property = expt_property
        bee.smiles = smiles
        bee.fitness = float(fitness)
        
    # for bee_ in swarm:
    #         print(f'{bee_.smiles}\n{bee_.fitness}\n{bee_.expt_property}')
    half_size = pop_size // 2
    
    # decide the size for onlooker and employed bees
    # explore: 1/2, exploit: 1/2
    return swarm[:half_size], swarm[half_size:]

def create_swarm_abc_aryl(center_position, pop_size, bounds, radius, transform:AryTransform, predictor:AryEnsmblePredictor, proc_feature_list, latent_size, JTVAE):
    # Generate the initial swarm.
    total_particle_position = torch.stack([center_position] * pop_size) #大大向量
    print('Finished position generating')
    print('Start generating inital SMILES')
    total_particle_position, total_smiles = adjust_position_and_get_smiles_aryl(total_particle_position, bounds, radius, transform, latent_size, JTVAE) #add noise and decode
    #print(total_smiles)
    #total_process_position = total_particle_position[:, latent_size:]
    #total_latent_position = total_particle_position[:, :latent_size]
    #feature_df = transform.to_dataframe(total_smiles, total_latent_position, total_process_position, proc_feature_list)
    #maybe another SMILES filter should be added here
    total_fitness, total_propertys = predictor.ensemble_predict(total_particle_position) #get the vector of all fitness and propertys
    swarm = [Bee(position, bounds) for position in total_particle_position]

    # for property, smiles, fitness in zip(total_propertys, total_smiles, total_fitness):
    #     print(smiles)
    #     print(property.reation_yield)
    #     print(fitness)
    #     print('\n')

    for bee, expt_property, smiles, fitness in zip(swarm, total_propertys, total_smiles, total_fitness):
        bee.expt_property = expt_property
        bee.smiles = smiles
        bee.fitness = float(fitness)
        
    # for bee_ in swarm:
    #         print(f'{bee_.smiles}\n{bee_.fitness}\n{bee_.expt_property}')
    
    half_size = pop_size // 2 #for start, try employed: onlooker = 9:1
    #print('employed_size', employed_size)
    return swarm[:half_size], swarm[half_size:]