import rdkit
import torch
#import hgraph
import pyfiglet
#from hgraph import HierVAE, PairVocab
from Algorithms.ABC.ArtificialBeeColony import ArtificialBeeColony
from Algorithms.ABC.BeeArgs import BeeArgs
from Algorithms.ABC.utils import create_swarm_abc, create_swarm_abc_aryl, create_swarm_abc_aryl_freeze
from Environments.Direct_Arylation.lib.Bounds import ary_bounds_v3
from Environments.Direct_Arylation.lib.AryPredictor import AryTransform, AryEnsmblePredictor
from Environments.Direct_Arylation.lib.utils import change_phos_to_nitrogen, load_ary_data
from fast_jtnn import *
from rdkit import Chem

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
    latent_vecs = vae_model.encode_latent_mean(smi_data) 

    #print('chosen smiles',smi_data.iloc[0])

    #print(latent_vecs[0])

    """
    def canonicalize_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    """

    """
    while True:
        tree_vec,mol_vec = torch.split(latent_vecs[0], 16, dim=0)
        tree_vec = tree_vec.unsqueeze(0)
        mol_vec = mol_vec.unsqueeze(0)
        smiles = vae_model.decode(tree_vec, mol_vec, prob_decode=False)
        print('decoded smiles',smiles)
        if canonicalize_smiles(smi_data.iloc[0]) == canonicalize_smiles(smiles):
            break
        else:
            print('seed molecules can not be decoded correctly, choosing another seed...')
            torch.manual_seed(args.seed + 1)
            # randomly choose another row as initial
            ary_dataframe = load_ary_data(args.ini_csv_path, index = None)
            smi_data =  ary_dataframe['ligand_SMILES']
            latent_vecs = vae_model.encode_latent_mean(smi_data) 
    """
    center_position = torch.cat((latent_vecs[0], torch.tensor(ary_dataframe[proc_feature].values, dtype=torch.float32)[0].to('cuda')), dim = 0)
  
    dataset = 'direct_arylation'

    print('choose to decode:', args.decode)
    print(pyfiglet.figlet_format('Start Create Bee'))
    if args.freeze is None:
        employed_bees, onlooker_bees = create_swarm_abc_aryl(center_position, args.pop_size, bounds, args.radius, transform, predictor, proc_feature, args.latent_size, vae_model)
    else:
        employed_bees, onlooker_bees = create_swarm_abc_aryl_freeze(center_position, args.pop_size, bounds, args.radius, transform, predictor, proc_feature, args.latent_size, vae_model, args.freeze)
    for idx, bee in enumerate(employed_bees):
        print(f'Bee {idx + 1}: {bee.smiles}')
    for idx, bee in enumerate(onlooker_bees):
        print(f'Bee {idx + len(employed_bees) + 1}: {bee.smiles}')

    print(pyfiglet.figlet_format('Optimization'))
    abc = ArtificialBeeColony(dataset, transform, predictor, employed_bees, onlooker_bees, args.max_trials, args.max_iterations, bounds, args.latent_size, vae_model) #transform, predictor, employed_bees, onlooker_bees, max_trials) 
    abc.run(args.freeze, args.decode)

    print(pyfiglet.figlet_format('Bee Rank'))
    abc.show_results(freeze = args.freeze, freeze_smiles=smi_data.iloc[0])