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

def check(input, grid_size, device):
    if isinstance(input, torch.Tensor):
        return input.to(device)
    if isinstance(input, np.ndarray):
        if input.dtype == object:
            return torch.from_numpy(np.array(input, dtype=np.float32)).to(device)
        return torch.from_numpy(input).to(device)
    if len(input[0].shape) == 1:
        return torch.tensor(np.array(input, dtype=np.float32)).to(device)
    # Flatten the list of lists
    flattened_sparse_tensors = input

    #Étape 1 : calcul du nombre total d'éléments
    #counts = [len(x[0]) for x in input]
    #total = sum(counts)
    #input_x = [x[0] for x in input]
    #input_y = [x[1] for x in input]
    lengths = np.array([len(x[0]) for x in input], dtype=np.int32)
    total = lengths.sum()
    #Étape 2 : préallocation sur le bon device
    #batch_indices_np = np.empty(total, dtype=np.int32)
    #x_indices_np = np.empty(total, dtype=np.int32)
    #y_indices_np = np.empty(total, dtype=np.int32)

    #fill_indices(input_x, input_y, batch_indices_np, x_indices_np, y_indices_np)
    #Étape 3 : remplissage en un seul passage
    #offset = 0
    #for batch_idx, (x, y) in enumerate(input):
	#n = len(x)
	#end = offset + n
    batch_indices_np = np.repeat(np.arange(len(input), dtype=np.int32), lengths)
    x_indices_np = np.concatenate([x[0] for x in input])
    y_indices_np = np.concatenate([x[1] for x in input])
	#offset = end
    #Création des indices et valeurs
    batch_indices = torch.from_numpy(batch_indices_np).to(device)
    x_indices = torch.from_numpy(x_indices_np).to(device)
    y_indices = torch.from_numpy(y_indices_np).to(device)
    values = torch.ones(total, dtype=torch.float32, device=device)
    
    indices = torch.stack([batch_indices, x_indices, y_indices], dim=0)
    shape = (len(input), grid_size, grid_size)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)
