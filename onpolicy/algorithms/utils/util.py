import copy
import numpy as np

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
    # Flatten the list of lists
    flattened_sparse_tensors = input

    # Initialize lists to store batch indices and tensor indices
    batch_indices_list = []
    tensor_indices_list = []

    # Iterate through the flattened sparse tensors to create batch indices
    for batch_idx, tensor in enumerate(flattened_sparse_tensors):
        tensor_indices = tensor.indices()
        batch_indices = torch.full((1, tensor_indices.size(1)), batch_idx, dtype=torch.long)
        batch_indices_list.append(batch_indices)
        tensor_indices_list.append(tensor_indices)

    # Concatenate batch indices and tensor indices
    batch_indices = torch.cat(batch_indices_list, dim=1)
    tensor_indices = torch.cat(tensor_indices_list, dim=1)

    # Combine batch indices with tensor indices
    combined_indices = torch.cat([batch_indices, tensor_indices], dim=0)

    # Concatenate values from each sparse tensor
    batch_values = torch.cat([tensor.values() for tensor in flattened_sparse_tensors])

    # Determine the batch size
    batch_size = (len(flattened_sparse_tensors), *flattened_sparse_tensors[0].size())

    # Create the batch sparse tensor
    batch_sparse_tensor = torch.sparse_coo_tensor(combined_indices, batch_values, batch_size)

    return batch_sparse_tensor
