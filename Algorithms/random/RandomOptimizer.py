import numpy as np
import random
from scipy.stats import norm
import torch
from Environments.PvkAdditives.lib.Pvk_Predictor import PvkTransform, Pvk_Ensemble_Predictor
from Environments.Direct_Arylation.lib.AryPredictor import AryTransform, AryEnsmblePredictor
from Environments.Direct_Arylation.lib.AryPredictor_DMPNN import AryTransform, AryEnsmblePredictor
from Environments.PvkAdditives.lib.Pvk_Types import ExperimentProperty
from Environments.Direct_Arylation.lib.AryTypes import ExperimentProperty

class RandomOptimizer:
    def __init__(self, 
                 dim,
                 predict_fn,
                 transform,
                 bounds,
                 num_samples,
                 vae_model, 
                 init_y,
                 dataset,
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
        self.y = init_y
        self.predict_fn = predict_fn

    def run(self, n_iter=100, verbose=True):
        for i in range(n_iter):
            if self.dataset == 'direct_arylation':
                smiles = "OOO"
                count = 0
                while "P" not in smiles:
                    count +=1
                    if count>1:
                        print("The molecule is", smiles, "which does not contain P, resampling...")
                    X_candidates = self.sampler_function(self.bounds, self.num_samples) 

                    print('Finished sampling candidates, start predict')

                    #from ensemble predictor, size mean and standard
                    mu = self.predict_fn.ensemble_predict(X_candidates)[0]
                    mu = np.array(mu)

                    print('Finished predict, mu is', mu)
                    
                    print('end predict')

                    x_next = X_candidates[np.argmax(mu)]
    
                    x_next = x_next.reshape(1, -1)

                    print('x_next is',x_next)
                    
                    #print(x_next)
                    y_pred_next, x_prop= self.predict_fn.ensemble_predict(x_next)
                    #print(y_pred_next)
                    xlatent = torch.tensor(x_next[0][:32], device=self.device)  # Get the latent vector
                    xlatent = xlatent.float()

                    smiles = self.transform.get_smiles_from_position([xlatent])[0] #device = self.device)])
                    self.seed += 1
                    self.reset_seed(self.seed) # reset seed to ensure the same result
                print(" Rank 1")
                print('Smiles:', smiles)
                print(x_prop[0])

                
            if self.dataset == 'pvk_additives':
                #sample candidates randomly
                X_candidates = self.sampler_function(self.bounds, self.num_samples) 
                #from ensemble predictor, size mean and standard
                mu = self.predict_fn.ensemble_predict(X_candidates)[0]
                mu = np.array(mu)

                x_next = X_candidates[np.argmax(mu)]
                x_next = x_next.reshape(1, -1)
                #print(x_next)
                y_pred_next, x_prop= self.predict_fn.ensemble_predict(x_next)
                #print(y_pred_next)
                xlatent = torch.tensor(x_next[0][:32], device=self.device)  # Get the latent vector
                xlatent = xlatent.float()

                #print(xlatent)
                smiles = self.transform.get_smiles_from_position([xlatent]) #device = self.device)])
                print("	Rank 1")
                print('Smiles:', smiles[0])
                print(x_prop[0])
            


    def sampler_function(self, bounds, num_samples=100):
        """
        bounds: shape = (dim, 2), bounds[:,0] = upper, bounds[:,1] = lower
        num_samples: number of samples to generate
        """
        print('Sampling...')
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