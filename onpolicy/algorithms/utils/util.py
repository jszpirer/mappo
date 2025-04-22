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
    print(input)
    # Assuming the input is a list of sparse Tensors, we can create a big batched sparse Tensor
    # Concatenate the sparse tensors along a new dimension (batch dimension)
    batch_indices = torch.cat([tensor.indices() for tensor in input], dim=1)
    print([tensor.indices() for tensor in input])
    batch_values = torch.cat([tensor.values() for tensor in input])
    size = (*input[0].size(),)
    print(size)

    # Create the batch sparse tensor
    batch_sparse_tensor = torch.sparse_coo_tensor(batch_indices, batch_values, (len(input), *input[0].size(),))

    print(batch_sparse_tensor)

    return batch_sparse_tensor
