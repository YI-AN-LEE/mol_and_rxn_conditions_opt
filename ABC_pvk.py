import rdkit
import torch
import random
import pyfiglet
import pandas as pd
import numpy as np

#from hgraph import HierVAE, PairVocab
from Algorithms.ABC.ArtificialBeeColony import ArtificialBeeColony
from Algorithms.ABC.BeeArgs import BeeArgs
from Algorithms.ABC.utils import create_swarm_abc
from Environments.PvkAdditives.lib.Pvk_Predictor import PvkTransform, Pvk_Ensemble_Predictor
from Environments.PvkAdditives.lib.Bounds import pvk_bounds_v2
import sys
#sys.path.append('/home/ianlee/JTVAE/JTVAE/GPU-P3')
from fast_jtnn import *

if __name__ == '__main__':
    args = BeeArgs()

    bounds = pvk_bounds_v2(log = False)[args.latent_size:] #bound is not for latent, fine 

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

    #init_x_tensor = torch.load(args.ini_tensor_path)[random.randint(0, init_expt_data.shape[0])] #process cond tensor, 不知道在尬嘛

    # Randomly select a row based on the seed
    num_rows = init_expt_data.shape[0]
    random_index = torch.randint(0, num_rows, (1,)).item()
    # Extract the data from the selected row
    selected_row = init_expt_data.iloc[random_index]

    # Print the selected row for verification
    print("Selected row based on seed:", random_index)
    #print(selected_row)
    #print(proc_list)
    #print(selected_row[proc_list].values)
    
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
    
    #center_position = torch.load(args.ini_tensor_path)[random.randint(0, init_expt_data.shape[0])]
    print(pyfiglet.figlet_format('Start Create Bee'))
    employed_bees, onlooker_bees = create_swarm_abc(center_position, args.pop_size, bounds, args.radius, transform, predictor, proc_list, args.latent_size, vae_model) #freeze_position = 0)
    
    # We print the initial Bee here
    
    print('initial employed bees')
    for idx, bee in enumerate(employed_bees):
        print(f'Bee {idx + 1}: {bee.smiles}')
    print('initial onlooker bees')
    for idx, bee in enumerate(onlooker_bees):
        print(f'Bee {idx + len(employed_bees) + 1}: {bee.smiles}')

    dataset = 'pvk_additives'    
    print(pyfiglet.figlet_format('Optimization'))
    abc = ArtificialBeeColony(dataset, transform, predictor, employed_bees, onlooker_bees, args.max_trials, args.max_iterations, bounds, args.latent_size, vae_model) #transform, predictor, employed_bees, onlooker_bees, max_trials) 
    abc.run()

    print(pyfiglet.figlet_format('Bee Rank'))
    abc.show_results()