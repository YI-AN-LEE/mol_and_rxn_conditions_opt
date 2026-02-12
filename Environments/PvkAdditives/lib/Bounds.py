import torch

import pandas as pd
import numpy as np

# def pvk_bounds(dtype, log = True, transpose = False):
#     # Set molecule dimensionality
#     d = 32
    
#     # Define bounds for molecule
#     bounds = torch.tensor([[5] * d, [-5] * d])
    
#     # Define bounds for process condition parameters
#     pro_bounds = torch.tensor([[176.1946, 0.5665], # R1
#                                [161.6851, 0.4246], # R2
#                                [189.9218, 1.0923], # R3
#                                [169.0334, 0.7438], # R4
#                                [1.0, 0.0]], dtype=dtype) # lab_code
    
#     # Compute logarithmic bounds for lab_code parameters
#     if log:
#         log_bounds = []
#         for i, bound in enumerate(pro_bounds):
#             if i == 4:
#                 log_bounds.append(bound)
#             else:
#                 log_bounds.append((torch.log(bound[0]), torch.log(bound[1])))
        
#         # Transform the list of bounds to a torch tensor
#         log_bounds = torch.tensor(log_bounds)
        
#         # Swap columns for the logarithmic bounds and transpose
#         log_bounds = log_bounds[:, [1, 0]].transpose(0, 1)
        
#         # Concatenate bounds and logarithmic bounds along dimension 1
#         bounds = torch.cat((bounds, log_bounds), dim=1)
#         if transpose:
#             bounds = bounds.transpose(0, 1) 
#     else:
#         # Concatenate bounds along dimension 1
#         bounds = torch.cat((bounds, pro_bounds.transpose(0, 1)), dim=1)
#         if transpose:
#             bounds = bounds.transpose(0, 1)
#     return bounds

# 20240418
bound_path = '/home/ianlee/opt_ian/Environments/PvkAdditives/lib/bounds_new.csv'

def pvk_bounds_v2(bound_file_path = bound_path, log = True, transpose = True):
    bounds = pd.read_csv(bound_file_path)

    # Get the values of the 'upper_bound', 'lower_bound' and 'dimension' columns
    upper_bounds = bounds['upper_bound'].values
    lower_bounds = bounds['lower_bound'].values
    dimensions = bounds['dimension'].values

    # Get the rows where 'dimension' is not 'ls_vector'
    mask = (bounds['column_name'] != 'ls_vector') & (bounds['continuous'] == 1)

    # Get the values of the 'upper_bound', 'lower_bound' and 'dimension' columns
    if log:
        upper_bounds = np.where(mask, np.log(upper_bounds), upper_bounds)
        lower_bounds = np.where(mask, np.log(lower_bounds), lower_bounds)

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