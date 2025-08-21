import torch
import rdkit
import random
import pyfiglet
import pandas as pd
import numpy as np
#from hgraph import HierVAE, PairVocab
from Algorithms.Fittness import Fitness
from Algorithms.PSO.Arg import PSOArgs
from Algorithms.PSO.Create import create_pool_v2, sort_filter_and_reappend
from Algorithms.PSO.ParticleSwarmOptimization import ParticleSwarmOptimization
from Environments.PvkAdditives.lib.Pvk_Predictor import Pvk_Ensemble_Predictor, PvkTransform
from Environments.PvkAdditives.lib.Bounds import pvk_bounds_v2
# this file is recomnad to run with python3.8.6(dupont envrioment)

from fast_jtnn import *

if __name__ == '__main__':
    args = PSOArgs()

    bounds = pvk_bounds_v2(log = False)[args.latent_size:] #bound need to be modified

    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
    #print(vocab)
    vocab = Vocab(vocab)

    # Initial Step for VAE
    vae_model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG).cuda()
    vae_model.load_state_dict(torch.load(args.model))
    vae_model.eval()

    transform = PvkTransform(vae_model, args.latent_size)
    predictor = Pvk_Ensemble_Predictor(vae_model, args.xgb_model_path, args.latent_size)

    seed = args.seed
    torch.manual_seed(seed)
    
    # Input data
    init_expt_data = pd.read_csv(args.ini_csv_path) #original data file
    proc_list = ['Reagent1 (ul)','Reagent2 (ul)','Reagent3 (ul)','Reagent4 (ul)', 'lab_code']
    #name of the columns   這兩行解決 + bound 就可以跑看看

    # Randomly select a row based on the seed
    num_rows = init_expt_data.shape[0]
    random_index = torch.randint(0, num_rows, (1,)).item()
    # Extract the data from the selected row
    selected_row = init_expt_data.iloc[random_index]

    # Print the selected row for verification
    print("Selected row based on seed:", random_index)
    
    #assure the starting point is not crystal size = 0
    while selected_row['crystal_size'] == 0:
        print('crytal size is 0, reselecting...')
        new_seed = torch.randint(0, 10000, (1,)).item()
        torch.manual_seed(new_seed)
        random_index = torch.randint(0, num_rows, (1,)).item()
        selected_row = init_expt_data.iloc[random_index]
        print("Selected row based on seed:", random_index)
    
    smi_data = selected_row['SMILES']
    latent_vecs = vae_model.encode_latent_mean([smi_data]) #returns a tensor storing all the latent vectors from a SMILES list
    # print('proc_feature', type(proc_feature), proc_feature)
    # print(latent_vecs[0])
    proc = selected_row[proc_list].values
    proc = np.float32(proc)
    center_position = torch.cat((latent_vecs[0], torch.tensor((proc), dtype=torch.float32).to('cuda')), dim = 0)

    print(pyfiglet.figlet_format('Start Create Pool'))
    pool = create_pool_v2(center_position, args.pop_size, bounds, args.radius, transform, predictor)
    # initial_position, index, pop_size, bounds, radius, transform:PvkTransform, predictor:Pvk_Ensemble_Predictor
    # pool = create_pool(center_particle, bounds, args.freeze, args.radius, args.pop_size, transform)

    # pool = generate_initial_particle_pool(init_x_tensor, init_expt_data, bounds, args.pop_size)
    
    for idx, particle in enumerate(pool):
        print(f'Particle {idx + 1}: {particle.smiles}')
    
    print(pyfiglet.figlet_format('Optimization'))
    for generation in range(args.generation):
        # Here we can just delete the duplicate particles especially for pop_size > 3.
        # Create new pool and add it into the old one.

        # open this to restart population (random sampling involved)
        # pool = sort_filter_and_reappend(pool, bounds, args.radius, args.pop_size, transform)
        
        pso = ParticleSwarmOptimization(pool, args.pso_epoch, transform, predictor, generation, radius=args.radius, sto = True) 
        #,inertia_weight = args.inertia_weight, cognitive_weight = args.cognitive_weight, social_weight = args.social_weight) 

        pso.optimize()
    pool = sorted(pool, key=lambda particle: particle.best_fitness, reverse=True)

    print(pyfiglet.figlet_format('Particles Rank'))
    for index, particle in enumerate(pool):
        print(f'\tRank {index + 1}')
        print(f'Smiles: {particle.best_smiles}')
        print(f'Fitness: {particle.best_fitness:.4e}')
        print(f'Molecule Property:')
        print(particle.best_expt_property)