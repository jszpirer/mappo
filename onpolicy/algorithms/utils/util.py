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
    if type(input) == torch.Tensor:
        return input
    if type(input) == np.ndarray:
        if input.dtype == object:
            return torch.from_numpy(np.array(input, dtype=np.float32))
        return torch.from_numpy(input)
    if len(input[0].shape) == 1:
        return torch.tensor(np.array(input))
    # Flatten the list of lists
    flattened_sparse_tensors = input

    # Initialize lists to store batch indices and tensor indices
    batch_indices = []
    values = []
    x_indices = []
    y_indices = []

    # Iterate through the flattened sparse tensors to create batch indices
    for batch_idx, indices in enumerate(flattened_sparse_tensors):
        for i in range(len(indices[0])):
            batch_indices.append(batch_idx)
            x_indices.append(indices[0][i])
            y_indices.append(indices[1][i])
            values.append(1)

    # Convertir les listes en tensors
    batch_indices = torch.tensor(batch_indices, dtype=torch.float32)
    x_indices = torch.tensor(x_indices, dtype=torch.float32)
    y_indices = torch.tensor(y_indices, dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32)

    batch_indices = torch.stack([batch_indices, x_indices, y_indices])

    # Determine the batch size
    batch_size = (len(input), 77, 77)

    # Create the batch sparse tensor
    batch_sparse_tensor = torch.sparse_coo_tensor(batch_indices, values, batch_size)

    return batch_sparse_tensor
