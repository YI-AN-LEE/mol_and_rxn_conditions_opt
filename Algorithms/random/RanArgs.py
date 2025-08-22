import argparse
#from hgraph import common_atom_vocab

def str2bool(v):
    return v.lower() in ('true', '1', 'yes')

def RanArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', default='/home/ianlee/optimizer/OW_DOE/vae_ckpt/wen_chembl/vocab.txt',
                        help='vocab path where the motif is saved for graph generation')
    #parser.add_argument('--atom_vocab', default=common_atom_vocab)
    parser.add_argument('--model', default='/home/ianlee/optimizer/OW_DOE/vae_data/wen_chembl/model.ckpt', 
                        help='model save path')
    parser.add_argument('--radius', default=0.25, type=float,
                        help='the search space for initializing the PSO particles.')
    parser.add_argument('--similarity_threshold', default=0.15, type=float,
                        help='the minimum similarity requirement for molecule generation.')
    parser.add_argument('--stable_epoch', default=0, type=int,
                        help='In this epoch, the cache will be emptied and record the optimized molecules again.')
    parser.add_argument('--seed', type=int, default=7,
                       help='random seed')
    parser.add_argument('--nsample', type=int, default=10000)

    parser.add_argument('--rnn_type', type=str, default='LSTM')
    parser.add_argument('--hidden_size', type=int, default=450)
    parser.add_argument('--embed_size', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--latent_size', type=int, default=128)
    parser.add_argument('--process_cond_size', type=int, default=5)
    parser.add_argument('--depth', type=int, default=3)
    

    parser.add_argument('--depthT', type=int, default=20)
    parser.add_argument('--depthG', type=int, default=3)
    
    """
    #use for HierVAE
    parser.add_argument('--diterT', type=int, default=1)
    parser.add_argument('--diterG', type=int, default=3)
    
    """
    parser.add_argument('--dropout', type=float, default=0.0)


    parser.add_argument('--freeze', type=str2bool, default = False,
                        help = 'Freeze the position of the process condition, default condition is freeze')
    parser.add_argument('--xgb_model_path', default='/home/ianlee/optimizer/OW_DOE/Model_Create_and_Results1/Direct_ary/3_Make_New_Data_Predictor/ABC', type=str,
                        help='Path to the xgb model')
    parser.add_argument('--ini_tensor_path', default = '/home/ianlee/optimizer/OW_DOE/Environments/PvkAdditives/Predictor_and_data/SA/cycle0_tensor.pt', type=str,
                        help=r"Initial X tensor for BO's (X,y) pairs preparation")
    parser.add_argument('--ini_csv_path', default = '/home/ianlee/optimizer/OW_DOE/Model_Create_and_Results1/Direct_ary/3_Make_New_Data_Predictor/ABC', type=str,
                        help="Initial data points containing molecules, operating conditions, and corresponding scores")
    parser.add_argument('--max_iterations', default = 200, type=int,
                        help='Maximum iterations for Artificial Bee Colony Optimization')
    parser.add_argument('--pop_size', default = 100, type=int,
                        help='Initail Population size for Artificial Bee Colony Optimization')
    parser.add_argument('--xi', default = 0.01, type=float,
                        help='xi for Bayesian Optimization, default is 0.01') 
    parser.add_argument('--num_samples', type=int, default = 1000,
                        help = 'samples for one BO run')
    parser.add_argument('--center_index', default = None, type=int,
                        help='Corresponding row in the initial CSV data and index in the initial tensor')
    return parser.parse_args()