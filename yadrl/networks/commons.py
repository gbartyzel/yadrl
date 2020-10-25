import numpy as np
import torch.nn as nn

from yadrl.networks.noisy_linear import (FactorizedNoisyLinear,
                                         IndependentNoisyLinear)


def fan_init(x: nn.Parameter):
    size = x.data.size()[1]
    val = 1 / np.sqrt(size)
    return -val, val


def orthogonal_init(x: nn.Module):
    classname = x.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.orthogonal_(x.weight.data, gain=np.sqrt(2))
        nn.init.constant_(x.bias.data, 0.0)


def get_layer(layer_type, input_dim, output_dim, sigma_init):
    if layer_type == 'none':
        return nn.Linear(input_dim, output_dim)
    elif layer_type == 'factorized':
        return FactorizedNoisyLinear(input_dim, output_dim, sigma_init)
    elif layer_type == 'independent':
        return IndependentNoisyLinear(input_dim, output_dim, sigma_init)
    raise ValueError(
        'Wrong layer type, choose between: none, factorized, independent')
