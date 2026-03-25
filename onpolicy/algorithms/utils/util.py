import copy
import numpy as np
from numba import njit
import torch
import torch.nn as nn

@njit
def fill_indices(input_x, input_y, batch_indices, x_indices, y_indices):
    offset = 0
    for batch_idx in range(len(input_x)):
        x = input_x[batch_idx]
        y = input_y[batch_idx]
        n = len(x)
        for i in range(n):
            batch_indices[offset+i] = batch_idx
            x_indices[offset+i] = x[i]
            y_indices[offset+i] = y[i]
        offset+=n

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input, grid_size, device, list_values=None, padding=False, nonomniscient=False):
    if isinstance(input, torch.Tensor):
        return input.to(device), None
    if isinstance(input, np.ndarray):
        if input.dtype == object:
            return torch.from_numpy(np.array(input, dtype=np.float32)).to(device), None
        return torch.from_numpy(input).to(device), None
    if len(input[0].shape) == 1:
        return torch.tensor(np.array(input, dtype=np.float32)).to(device), None
    if nonomniscient:
        batch_size = len(input)
        batch = torch.full((batch_size, input[0].shape[0], input[0].shape[1]), fill_value=0, dtype=torch.float32, device=device)
        
        for i, arr in enumerate(input):
            batch[i, :, :] = torch.as_tensor(arr, device=device)
        return batch, None
    if padding:
        # In this case, attention mechanism and padding needed to do batch operations
        batch_size = len(input)
        n_max = max(a.shape[0] for a in input)

        # Pre allocation for the final batch tensors
        batch_np = np.zeros((batch_size, n_max, 2), dtype=np.float32)
        mask_padding = torch.ones((batch_size, n_max), dtype=torch.bool, device=device)

        # Filling in the tensors
        for i, arr in enumerate(input):
            n_i = arr.shape[0]
            batch_np[i, :n_i, :] = arr
            mask_padding[i, :n_i] = False
        batch = torch.from_numpy(batch_np).to(device)
        return batch, mask_padding

    #Étape 1 : calcul du nombre total d'éléments
    lengths = np.array([len(x[0]) for x in input], dtype=np.int32)
    total = lengths.sum()

    #Étape 3 : remplissage en un seul passage
    batch_indices_np = np.repeat(np.arange(len(input), dtype=np.int32), lengths)
    x_indices_np = np.concatenate([x[0] for x in input])
    y_indices_np = np.concatenate([x[1] for x in input])
    
    #Création des indices et valeurs
    batch_indices = torch.from_numpy(batch_indices_np).to(device)
    x_indices = torch.from_numpy(x_indices_np).to(device)
    y_indices = torch.from_numpy(y_indices_np).to(device)
    if list_values is None:
        values = torch.ones(total, dtype=torch.float32, device=device)
    else:
        values = torch.from_numpy(np.concatenate(list_values)).to(device)
    
    indices = torch.stack([batch_indices, x_indices, y_indices], dim=0)
    shape = (len(input), grid_size, grid_size)
    return torch.sparse_coo_tensor(indices, values, shape, device=device), None
