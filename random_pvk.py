import rdkit
import torch
import random
import pyfiglet
import pandas as pd
import numpy as np

#from hgraph import HierVAE, PairVocab
from Algorithms.random.RandomOptimizer import RandomOptimizer
from Algorithms.random.RanArgs import RanArgs
from Environments.PvkAdditives.lib.Pvk_Predictor import PvkTransform, Pvk_Ensemble_Predictor
from Environments.PvkAdditives.lib.Bounds import pvk_bounds_v2
import sys
#sys.path.append('/home/ianlee/JTVAE/JTVAE/GPU-P3')
from fast_jtnn import *

if __name__ == '__main__':
    args = RanArgs()

    bounds = pvk_bounds_v2(log = False) #needs bound for latent for sampling molecules
    print('length of bound=',len(bounds))
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
    init_X = init_expt_data[proc_list].values
    init_y = init_expt_data['crystal_size'].values

    dataset = 'pvk_additives'    
    print(pyfiglet.figlet_format('Optimization'))
    print(pyfiglet.figlet_format('Random Rank'))
    ran = RandomOptimizer(37, predictor, transform, bounds, args.num_samples, vae_model, init_y, dataset) #transform, predictor, employed_bees, onlooker_bees, max_trials) 
    ran.run(n_iter=args.max_iterations, verbose=True)

    
    #bo.get_all_data()