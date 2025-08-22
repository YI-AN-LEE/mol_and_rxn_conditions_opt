import torch

import pandas as pd
import numpy as np

from Environments.Direct_Arylation.lib.utils import BOUNBD_PATH
# 20240418

def ary_bounds_v3(bound_file_path = BOUNBD_PATH, log = True, transpose = True):
    bounds = pd.read_csv(bound_file_path)

    # Get the values of the 'upper_bound', 'lower_bound' and 'dimension' columns
    upper_bounds = bounds['upper_bound'].values
    lower_bounds = bounds['lower_bound'].values
    dimensions = bounds['dimension'].values

    # Initialize an empty list for the bounds
    lower_bounds_list = []
    upper_bounds_list = []

    # Iterate over the bounds and dimensions
    for ub, lb, dim in zip(upper_bounds, lower_bounds, dimensions):
        # Repeat the upper and lower bounds according to the dimension and append to the list
        lower_bounds_list.extend([lb]*dim)
        upper_bounds_list.extend([ub]*dim)

    # Convert the list to a tensor
    lower_bounds_tensor = torch.tensor(lower_bounds_list)
    upper_bounds_tensor = torch.tensor(upper_bounds_list)

    # Stack the tensors to have 2 rows
    bounds_tensor = torch.stack((upper_bounds_tensor, lower_bounds_tensor))
    
    # Transpose
    if transpose:
        bounds_tensor = bounds_tensor.transpose(0, 1)
        
    
    return bounds_tensor[32:], {key: value for key, value in zip(bounds['column_name'].tolist(), bounds['normalize'].tolist())}

def ary_bounds_v4(bound_file_path = BOUNBD_PATH, log = True, transpose = True):
    bounds = pd.read_csv(bound_file_path)

    # Get the values of the 'upper_bound', 'lower_bound' and 'dimension' columns
    upper_bounds = bounds['upper_bound'].values
    lower_bounds = bounds['lower_bound'].values
    dimensions = bounds['dimension'].values

    # Get the rows where 'dimension' is not 'ls_vector'
    mask = bounds['normalize'] == 1

    # Get the values of the 'upper_bound', 'lower_bound' and 'dimension' columns
    if log:
        upper_bounds = np.where(mask, np.log1p(upper_bounds), upper_bounds)
        lower_bounds = np.where(mask, np.log1p(lower_bounds), lower_bounds)

    # Initialize an empty list for the bounds
    lower_bounds_list = []
    upper_bounds_list = []

    # Iterate over the bounds and dimensions
    for ub, lb, dim in zip(upper_bounds, lower_bounds, dimensions):
        # Repeat the upper and lower bounds according to the dimension and append to the list
        lower_bounds_list.extend([lb]*dim)
        upper_bounds_list.extend([ub]*dim)

    # Convert the list to a tensor
    lower_bounds_tensor = torch.tensor(lower_bounds_list)
    upper_bounds_tensor = torch.tensor(upper_bounds_list)

    # Stack the tensors to have 2 rows
    bounds_tensor = torch.stack((upper_bounds_tensor, lower_bounds_tensor))
    
    # Transpose
    if transpose:
        bounds_tensor = bounds_tensor.transpose(0, 1)
        
    
    return bounds_tensor, {key: value for key, value in zip(bounds['column_name'].tolist(), bounds['normalize'].tolist())}