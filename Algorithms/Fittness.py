from scipy.stats import norm
from functools import reduce
from operator import mul
from typing import Dict

cutoff_dict = {"crystal_size": [1, None],
            # "crystal_score": [2, None],
        }

class Fitness:
    def __init__(self, cutoff_dict):
        self.cutoff_dict = cutoff_dict
        self.prob_dict: Dict = {}
        
    def check_length(self, pred_dict:Dict):
        """
        Check if all the keys in the dict have the same length.
        Args:
            dict: The dict to be checked.
        Returns:
            If the lengths are the same, return the length. Otherwise, return None.
        """
        lengths = [len(key) for key in pred_dict.values()]
        if len(set(lengths)) == 1:
            return lengths[0]
        else:
            raise ValueError(f"Some targets have different list lengths: {lengths}")
    
    def calculate_greater_cdf(self, cutoff, mean, std):
        '''
        Calculate the cumulative distribution greater than a certain cutoff,
        that is, the probability greater than a certain cutoff
        '''
        return 1 - norm.cdf(cutoff, loc=mean, scale=std)
    
    def calculate_less_cdf(self, cutoff, mean, std): 
        '''
        Calculate the cumulative distribution smaller than a certain cutoff, that is, 
        the probability of being smaller than a certain cutoff
        '''
        return norm.cdf(cutoff, loc=mean, scale=std)
    
    def calculate_cdfs(self):
        self.prob_dict = {}
        for target, cutoff in self.cutoff_dict.items():
            cdfs = []
            for m_and_s in self.pred_dict[target]:
                if cutoff[0] is not None:
                    cdf = self.calculate_greater_cdf(cutoff[0], m_and_s[0], m_and_s[1]) # cutoff要大於的要放第0個
                elif cutoff[1] is not None:
                    cdf = self.calculate_less_cdf(cutoff[1], m_and_s[0], m_and_s[1]) # cutoff要大於的要放第1個
                cdfs.append(cdf)
            self.prob_dict[target] = cdfs
    
    def calculate_cdf_products(self):
        self.prob_list = []
        max_index = self.check_length(self.prob_dict)
        for i in range(max_index):
            single_set_probs = []
            for cdf_list in self.prob_dict.values():
                single_set_probs.append(cdf_list[i])
            self.prob_list.append(reduce(mul, single_set_probs))
    
    def cal_fitness_by_cdf_method(self, pred_dict: Dict):
        self.pred_dict = pred_dict
        self.calculate_cdfs()
        self.calculate_cdf_products()
        return self.prob_list
