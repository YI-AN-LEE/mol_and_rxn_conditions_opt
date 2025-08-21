import torch
import rdkit
import random
import pyfiglet

#from hgraph import HierVAE, PairVocab, MoleculeDataset
from Algorithms.random.RandomOptimizer import RandomOptimizer
from Algorithms.random.RanArgs import RanArgs
from Environments.Direct_Arylation.lib.Bounds import ary_bounds_v4
from Environments.Direct_Arylation.lib.AryPredictor import AryTransform, AryEnsmblePredictor
from Environments.Direct_Arylation.lib.utils import change_phos_to_nitrogen, load_ary_data
from fast_jtnn import *

if __name__ == '__main__':
    args = RanArgs()
    torch.manual_seed(args.seed)

    bounds, proc_feature = ary_bounds_v4(log = False)
    #print(proc_feature)
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
    vocab = Vocab(vocab)

    # Initial Step for VAE
    vae_model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG).cuda()
    vae_model.load_state_dict(torch.load(args.model))
    vae_model.eval()

    transform = AryTransform(vae_model, proc_feature, args.latent_size)
    predictor = AryEnsmblePredictor(vae_model, args.xgb_model_path, args.latent_size)
    # randomly choose one row as initial
    init_expt_data = load_ary_data(args.ini_csv_path, index = 1)
    #print('Origiinal Data', init_expt_data)
    
    #init_X = init_expt_data[proc_feature].values
    init_y = init_expt_data['yield'].values

    dataset = 'direct_arylation'    
    print(pyfiglet.figlet_format('Optimization'))
    print(pyfiglet.figlet_format('BO Rank'))
    bo = RandomOptimizer(37, predictor, transform, bounds, 
                            args.num_samples, vae_model, init_y, dataset, args.seed) #transform, predictor, employed_bees, onlooker_bees, max_trials) 
    bo.run(n_iter=args.max_iterations, verbose=True)