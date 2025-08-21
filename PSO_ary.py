import torch
import rdkit
import random
import pyfiglet

#from hgraph import HierVAE, PairVocab, MoleculeDataset
from Algorithms.PSO.Arg import PSOArgs
from Algorithms.PSO.Create import create_pool_v3, sort_filter_and_reappend
from Algorithms.PSO.ParticleSwarmOptimization import ParticleSwarmOptimization
from Environments.Direct_Arylation.lib.Bounds import ary_bounds_v3
from Environments.Direct_Arylation.lib.AryPredictor import AryTransform, AryEnsmblePredictor
from Environments.Direct_Arylation.lib.utils import change_phos_to_nitrogen, load_ary_data
from fast_jtnn import *

if __name__ == '__main__':
    args = PSOArgs()
    torch.manual_seed(args.seed)

    bounds, proc_feature = ary_bounds_v3(log = False)
    # print(proc_feature)
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
    print('Origiinal Data', ary_dataframe)
    
    # map all things to proc_feautre
    proc_feature = ary_dataframe.columns.intersection(proc_feature.keys())
    print('the mapped process feautures')
    print(proc_feature)

    transform = AryTransform(vae_model, proc_feature, args.latent_size)
    predictor = AryEnsmblePredictor(vae_model, args.xgb_model_path, args.latent_size)
    
    smi_data =  ary_dataframe['ligand_SMILES']
    print(smi_data)
    latent_vecs = vae_model.encode_latent_mean(smi_data) #returns a tensor storing all the latent vectors from a SMILES list
    # print('proc_feature', type(proc_feature), proc_feature)
    center_position = torch.cat((latent_vecs[0], torch.tensor(ary_dataframe[proc_feature].values, dtype=torch.float32)[0].to('cuda')), dim = 0)


    print(pyfiglet.figlet_format('Start Create Pool'))

    # initial pool sampled from the center position with radius = 2.5
    pool = create_pool_v3(center_position, args.pop_size, bounds, args.radius, transform, predictor, proc_feature)

    for idx, particle in enumerate(pool):
        print(f'Particle {idx + 1}: {particle.smiles}')
    
    print(pyfiglet.figlet_format('Optimization'))
    for generation in range(args.generation):
        pso = ParticleSwarmOptimization(pool, args.pso_epoch, transform, predictor, generation, radius=args.radius, sto = True)
        pso.optimize()
        print(pool)
    pool = sorted(pool, key=lambda particle: particle.best_fitness, reverse=True)

    print(pyfiglet.figlet_format('Particles Rank'))
    for index, particle in enumerate(pool):
        print(f'\tRank {index + 1}')
        print(f'Smiles: {particle.best_smiles}')
        print(f'Fitness: {particle.best_fitness:.4e}')
        print(f'Experiment Property:')
        print(particle.best_expt_property)