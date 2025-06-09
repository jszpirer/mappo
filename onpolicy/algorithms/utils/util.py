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
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)

    # input est une liste de sparse tensors de même forme
    n = len(input)
    sparse = input[0]
    k = sparse.values().size(0)  # nombre de non-zéros par tensor (supposé constant)

    # Stack des indices et valeurs
    tensor_indices = torch.stack([t.indices() for t in input])  # (n, ndim, k)
    tensor_values = torch.stack([t.values() for t in input])    # (n, k)

    # Création des indices de batch
    batch_indices = torch.arange(n).view(n, 1).expand(n, k)     # (n, k)
    batch_indices = batch_indices.unsqueeze(1)                  # (n, 1, k)

    # Concaténation des indices batch + indices tensoriels
    full_indices = torch.cat([batch_indices, tensor_indices], dim=1)  # (n, ndim+1, k)
    full_indices = full_indices.reshape(full_indices.size(1), -1)     # (ndim+1, n*k)

    # Aplatissement des valeurs
    full_values = tensor_values.reshape(-1)

    # Taille finale
    shape = (n,) + sparse.size()

    return torch.sparse_coo_tensor(full_indices, full_values, shape)


