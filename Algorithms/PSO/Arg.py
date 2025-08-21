import argparse

def str2bool(v):
    # 将字符串转换为布尔值
    return v.lower() in ('true', '1', 'yes')

def PSOArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', default='/home/chenyo/OW_DOE/vae_ckpt/wen_chembl/vocab.txt',
                        help='vocab path where the motif is saved for graph generation')
    parser.add_argument('--freeze', type=str2bool, default = False,
                        help = 'Freeze the position of the process condition, default condition is freeze')
    parser.add_argument('--model', default='/home/chenyo/OW_DOE/vae_data/wen_chembl/model.ckpt', 
                        help='model save path')
    # parser.add_argument('--pvk_data', default = '/home/chenyo/OW_DOE/Environments/PvkAdditives/cycle0/cycle0.csv', type=str,
    #                     help='load the .csv file with used pvk task file.')
    # parser.add_argument('--pvk_gt_model', default = '/home/chenyo/OW_DOE/Model_Create_and_Results/pvk_train/pvk_rfr_size.pkl', type=str,
    #                     help='pvk ground truth model dir')
    # parser.add_argument('--pvk_xgb_model_path', default='/home/chenyo/OW_DOE/Environments/PvkAdditives/cycle_2', type=str,
    #                     help='the directory where the pvk ensemble xgboost models are placed.')
    parser.add_argument('--pop_size', default=100, type=int,
                        help='the size of particle pool in PSO.')
    parser.add_argument('--generation', default=4, type=int,
                        help='number of times the ball pool is cleaned.')
    parser.add_argument('--radius', default=0.25, type=float,
                        help='the search space for initializing the PSO particles.')
    parser.add_argument('--pso_epoch', default=50, type=int,
                        help='how many iterations for running PSO.')
    parser.add_argument('--similarity_threshold', default=0.15, type=float,
                        help='the minimum similarity requirement for molecule generation.')
    parser.add_argument('--topn', default=100, type=int,
                        help='finally outputs how many molecules.')
    parser.add_argument('--stable_epoch', default=0, type=int,
                        help='In this epoch, the cache will be emptied and record the optimized molecules again.')
    parser.add_argument('--xgb_model_path', default='/home/chenyo/OW_DOE/Model_Create_and_Results1/Direct_ary/3_Make_New_Data_Predictor/PSO', type=str,
                        help='Path to the xgb model')
    parser.add_argument('--ini_tensor_path', default = '/home/chenyo/OW_DOE/Environments/PvkAdditives/Predictor_and_data/SA/cycle0_tensor.pt', type=str,
                        help=r"Initial X tensor for BO's (X,y) pairs preparation")
    parser.add_argument('--ini_csv_path', default = '/home/chenyo/OW_DOE/Model_Create_and_Results1/Direct_ary/3_Make_New_Data_Predictor/PSO', type=str,
                        help="Initial data points containing molecules, operating conditions, and corresponding scores")
    parser.add_argument('--center_index', default = None, type=int,
                        help='Corresponding row in the initial CSV data and index in the initial tensor')

    parser.add_argument('--seed', type=int, default=42,
                       help='')
    parser.add_argument('--nsample', type=int, default=10000)

    parser.add_argument('--rnn_type', type=str, default='LSTM')
    parser.add_argument('--hidden_size', type=int, default=450)
    parser.add_argument('--embed_size', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--latent_size', type=int, default=32)
    parser.add_argument('--process_cond_size', type=int, default=5)
    parser.add_argument('--depth', type=int, default=3)

    parser.add_argument('--depthT', type=int, default=20)
    parser.add_argument('--depthG', type=int, default=3)

    return parser.parse_args()