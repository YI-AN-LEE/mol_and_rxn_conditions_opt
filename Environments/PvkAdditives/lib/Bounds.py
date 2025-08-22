import torch

import pandas as pd
import numpy as np

bound_path = 'Environments/PvkAdditives/lib/bounds_new.csv'

def pvk_bounds_v2(bound_file_path = bound_path, log = True, transpose = True):
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
    return bounds_tensor