# import torch 

# process_condition_bounds = torch.tensor([[176.1946, 0.5665],
#                         [161.6851, 0.4246],
#                         [189.9218, 1.0923],
#                         [169.0334, 0.7438],
#                         [1.0, 0.0]])

# def normalize_tensor(input_tensor):
#     min_values = process_condition_bounds[:4][:, 1]
#     max_values = process_condition_bounds[:4][:, 0]
#     normalized_tensor = (input_tensor - min_values) / (max_values - min_values)

#     return normalized_tensor

# def unnormalize_tensor(input_tensor):
#     min_values = process_condition_bounds[:4][:, 1]
#     max_values = process_condition_bounds[:4][:, 0]
#     unnormalized_tensor = input_tensor * (max_values - min_values) + min_values

#     return unnormalized_tensor

