import rdkit
import torch
#import hgraph
import pyfiglet

#from hgraph import HierVAE, PairVocab
from Algorithms.ABC.ArtificialBeeColony import ArtificialBeeColony
from Algorithms.ABC.BeeArgs import BeeArgs
from Algorithms.ABC.utils import create_swarm_abc, create_swarm_abc_aryl
from Environments.Direct_Arylation.lib.Bounds import ary_bounds_v3
from Environments.Direct_Arylation.lib.AryPredictor import AryTransform, AryEnsmblePredictor
from Environments.Direct_Arylation.lib.utils import change_phos_to_nitrogen, load_ary_data
from fast_jtnn import *

if __name__ == '__main__':
    args = BeeArgs()
    torch.manual_seed(args.seed)

    # bounds maps T, P to numbers, ligand to VAE latent vector, base and solvent to number 1-4
    bounds, proc_feature = ary_bounds_v3(log = False)
    print(proc_feature)
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
    vocab = Vocab(vocab)

    # Initial Step for VAE
    vae_model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG).cuda()
    vae_model.load_state_dict(torch.load(args.model))
    vae_model.eval()

    # randomly choose one row as initial
    ary_dataframe = load_ary_data(args.ini_csv_path, index = None)
    # print('Origiinal Data', ary_dataframe)

    # map all things to proc_feautre
    proc_feature = ary_dataframe.columns.intersection(proc_feature.keys())
    print('the mapped process feautures')
    print(proc_feature)
    
    transform = AryTransform(vae_model, proc_feature, args.latent_size)
    predictor = AryEnsmblePredictor(vae_model, args.xgb_model_path, args.latent_size)

    smi_data =  ary_dataframe['ligand_SMILES']
    latent_vecs = vae_model.encode_latent_mean(smi_data) #returns a tensor storing all the latent vectors from a SMILES list
    # print('proc_feature', type(proc_feature), proc_feature)
    # print(ary_dataframe[proc_feature].values)
    
    center_position = torch.cat((latent_vecs[0], torch.tensor(ary_dataframe[proc_feature].values, dtype=torch.float32)[0].to('cuda')), dim = 0)

    dataset = 'direct_arylation'

    print(pyfiglet.figlet_format('Start Create Bee'))
    employed_bees, onlooker_bees = create_swarm_abc_aryl(center_position, args.pop_size, bounds, args.radius, transform, predictor, proc_feature, args.latent_size, vae_model)
    for idx, bee in enumerate(employed_bees):
        print(f'Bee {idx + 1}: {bee.smiles}')
    for idx, bee in enumerate(onlooker_bees):
        print(f'Bee {idx + len(employed_bees) + 1}: {bee.smiles}')

    print(pyfiglet.figlet_format('Optimization'))
    abc = ArtificialBeeColony(dataset, transform, predictor, employed_bees, onlooker_bees, args.max_trials, args.max_iterations, bounds, args.latent_size, vae_model) #transform, predictor, employed_bees, onlooker_bees, max_trials) 
    abc.run()

    print(pyfiglet.figlet_format('Bee Rank'))
    abc.show_results()