import numpy as np
import random
from scipy.stats import norm
import torch
from Environments.PvkAdditives.lib.Pvk_Predictor import PvkTransform, Pvk_Ensemble_Predictor
from Environments.Direct_Arylation.lib.AryPredictor import AryTransform, AryEnsmblePredictor
from Environments.PvkAdditives.lib.Pvk_Types import ExperimentProperty
from Environments.Direct_Arylation.lib.AryTypes import ExperimentProperty

class BayesianOptimizer:
    def __init__(self, 
                 dim,
                 predict_fn,
                 transform,
                 bounds,
                 num_samples,
                 vae_model, 
                 init_y,
                 dataset,
                 fscale,
                 seed = None,
                 maximize=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = dim
        self.dataset = dataset
        #self.evaluate_fn = evaluate_fn
        self.vae_model = vae_model
        self.transform = transform
        self.maximize = maximize
        self.bounds = bounds    
        self.seed = seed
        self.num_samples = num_samples
        # 儲存所有歷史資料
        self.fscale = fscale
        self.y = init_y
        self.predict_fn = predict_fn

    def expected_improvement(self, mu, sigma, f_best, xi=0.01):
        sigma = np.maximum(sigma, 1e-8)
        improvement = (mu - f_best - xi) if self.maximize else (f_best - mu - xi)
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        return ei

    def run(self, n_iter=100, xi=0.01, verbose=True):
        for i in range(n_iter):
            if self.dataset == 'direct_arylation':
                smiles = "OOO"
                count = 0
                while "P" not in smiles:
                    count +=1
                    if count>1:
                        print("The molecule is", smiles, "which does not contain P, resampling...")
                    #sample candidates randomly
                    X_candidates = self.sampler_function(self.bounds, self.num_samples) 
                    #from ensemble predictor, size mean and standard
                    mu, sigma = zip(*self.predict_fn.ensemble_predict_BO(X_candidates)[2])
                    mu = np.array(mu)
                    sigma = np.array(sigma)
                    #print(mu[:100],sigma[:100])
                    #print(mu[:10], sigma.head[:10])
                    f_best = np.max(self.y)*self.fscale if self.maximize else np.min(self.y)
                    #print(f_best)
                    ei = self.expected_improvement(mu, sigma, f_best, xi=xi)

                    """
                    x_next = X_candidates[np.argmax(ei)]
                    # reshape 為 (1, 37)
                    x_next = x_next.reshape(1, -1)
                    #print(x_next)
                    y_pred_next, y_std_next, x_prop= self.predict_fn.ensemble_predict_BO(x_next)[2][0][0], self.predict_fn.ensemble_predict_BO(x_next)[2][0][1], self.predict_fn.ensemble_predict_BO(x_next)[1] # Predict the next point
                    #print(y_pred_next)
                    xlatent = torch.tensor(x_next[0][:32], device=self.device)  # Get the latent vector
                    xlatent = xlatent.float()
                    #print(xlatent)
                    smiles = self.transform.get_smiles_from_position([xlatent])[0] #device = self.device)])
                    self.seed += 1
                    self.reset_seed(self.seed) # reset seed to ensure the same result
                    """

                    # 1. 選出 EI 前 10 名的索引
                    top_k = 10
                    top_indices = np.argsort(ei)[-top_k:][::-1]  # 降冪排序，取前10名

                    # 2. 對每一個 top candidate 做 decode
                    decoded_smiles = []
                    for idx in top_indices:
                        x_candidate = X_candidates[idx].reshape(1, -1)
                        xlatent = torch.tensor(x_candidate[0][:32], device=self.device).float()
                        smiles = self.transform.get_smiles_from_position([xlatent])[0]
                        decoded_smiles.append((idx, smiles))

                    # 3. 過濾出含磷分子
                    phosphorus_molecules = [(idx, smi) for idx, smi in decoded_smiles if 'P' in smi]

                    x_next = X_candidates[top_indices[0]].reshape(1, -1)  # 取 EI 最高的分子
                    # 4. 印出結果
                    print("含磷分子 (前 10 EI):")
                    for idx, smi in phosphorus_molecules:
                        print(f"Index {idx}: {smi}")


                    self.seed += 1
                    self.reset_seed(self.seed) # reset seed to ensure the same result
                    
                    
                print(" Rank 1")
                print('Smiles:', smiles)
                print(x_prop[0])
                print("EI = ", np.max(ei))

            
            
            if self.dataset == 'pvk_additives':
                #sample candidates randomly
                X_candidates = self.sampler_function(self.bounds, self.num_samples) 
                #from ensemble predictor, size mean and standard
                mu, sigma = zip(*self.predict_fn.ensemble_predict_BO(X_candidates)[2])
                mu = np.array(mu)
                sigma = np.array(sigma)
                f_best = np.max(self.y) if self.maximize else np.min(self.y)

                print('mu:', mu[:10])          # 查看前 10 個預測值
                print('f_best:', f_best)       # 當前最佳真實值
                print('improvement:', mu[:10] - f_best)
                print('sigma:', sigma[:10])
                #print(mu[:100],sigma[:100])
                #print(mu[:10], sigma.head[:10])
                
                #print(f_best)
                ei = self.expected_improvement(mu, sigma, f_best, xi=xi)

                x_next = X_candidates[np.argmax(ei)]
                # reshape 為 (1, 37)
                x_next = x_next.reshape(1, -1)
                #print(x_next)
                y_pred_next, y_std_next, x_prop= self.predict_fn.ensemble_predict_BO(x_next)[2][0][0], self.predict_fn.ensemble_predict_BO(x_next)[2][0][1], self.predict_fn.ensemble_predict_BO(x_next)[1] # Predict the next point
                #print(y_pred_next)
                xlatent = torch.tensor(x_next[0][:32], device=self.device)  # Get the latent vector
                xlatent = xlatent.float()
                #print(xlatent)
                smiles = self.transform.get_smiles_from_position([xlatent]) #device = self.device)])
                print("	Rank 1")
                print('Smiles:', smiles[0])
                print('Fitness:', y_pred_next)
                print('Molecular Property:')
                print('	Process Condition') 
                print('Reagent1 (ul): %.4f \nReagent2 (ul): %.4f \nReagent3 (ul): %.4f \nReagent4 (ul): %.4f'%(x_next[0][32], 
                x_next[0][33], x_next[0][34], x_next[0][35]))
                print('lab_code:', float(x_next[0][36]))
                print(' Prediction Result')
                print('crystal_size: %.4f,%.4f'%(y_pred_next, y_std_next))
                print("EI = ", np.max(ei))
                print("-------------------------")
            
            

            #y_next = self.evaluate_fn(x_next)[0] # gives the mean value

            #self.X = np.vstack([self.X, x_next.reshape(1, -1)])
            #self.y = np.append(self.y, y_next)

            """
            if verbose:
                best_y = np.max(self.y) if self.maximize else np.min(self.y)
                idx = np.argmax(self.y) if self.maximize else np.argmin(self.y)
                x_best = self.X[idx]
                y_best = self.y[idx]

                # decode the molecular part back
                smiles = self.transform.get_smiles_from_position([x_best[:32]])
                print(f'Iteration {i + 1}')
                print(f'Best Smiles: {smiles}')
                print(f'Fitness: {y_best}')
                print(f'Experiment Property:',x_best[32:])

                print(f"[{i+1:03d}] New y: {y_next:.4f} | Best y: {best_y:.4f}")
            """

    def sampler_function(self, bounds, num_samples=1000):
        """
        bounds: shape = (dim, 2), bounds[:,0] = upper, bounds[:,1] = lower
        num_samples: number of samples to generate
        """
        dim = bounds.shape[0]
        upper_bounds = bounds[:, 0]
        lower_bounds = bounds[:, 1]

        # shape = (num_samples, dim)
        random_unit = np.random.rand(num_samples, dim)  # Uniform [0, 1] random
        samples = lower_bounds + (upper_bounds - lower_bounds) * random_unit

        return samples
        
    def reset_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    """    
    def top_k(self, k=10):
        idx = np.argsort(-self.y if self.maximize else self.y)[:k] #only take top 10 values
        return self.X[idx], self.y[idx]

    def get_all_data(self):
        smiles = self.transform.get_smiles_from_position(self.X[:, :32])
        expt_properties = self.X[:, 32:]
        y = self.y
        for i in range(len(smiles)):
            print('SMILES: ', smiles[i])
            print('Properites:', expt_properties[i])
            print('Fitness:', y[i])
    

    """