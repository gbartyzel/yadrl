from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn


def hidden_init(x):
    size = x.data.size()[0]
    val = 1 / np.sqrt(size)
    return -val, val


class BaseMLPNetwork(nn.Module):
    def __init__(self,
                 input_dim: Union[int, Sequence[int]],
                 output_dim: int,
                 activation_fn: nn.Module = nn.ReLU):
        super(BaseMLPNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._activation_fn = activation_fn()

    def forward(self, *input):
        return NotImplementedError


class MLPNetwork(BaseMLPNetwork):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: Sequence[int],
                 activation_fn: nn.Module = nn.ReLU):
        super(MLPNetwork, self).__init__(input_dim, hidden_dim[-1], activation_fn)

        layers = (input_dim,) + tuple(hidden_dim)
        self._dense_layers = [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]

    def forward(self, x: torch.Tensor):
        for layer in self._dense_layers:
            x = self._activation_fn(layer(x))
        return x


class DDPGMLPNetwork(BaseMLPNetwork):
    def __init__(self,
                 input_dim: Sequence[int],
                 hidden_dim: Sequence[int],
                 activation_fn: nn.Module = nn.ReLU):
        super(DDPGMLPNetwork, self).__init__(input_dim, hidden_dim[-1], activation_fn)

        layers = (input_dim[0],) + tuple(hidden_dim)
        self._dense_layers = [
            nn.Linear(layers[i] + input_dim[1] if i == 1 else layers[i], layers[i + 1])
            for i in range(len(layers) - 1)
        ]

        self._initialize_variables()

    def _initialize_variables(self):
        for layer in self._dense_layers:
            layer.weight.data.uniform_(*hidden_init(layer.weight))
            layer.bias.data.uniform_(*hidden_init(layer.bias))

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        for i, layer in enumerate(self._dense_layers):
            if i == 1:
                x = torch.cat((x, u), dim=1)
            x = self._activation_fn(layer(x))
        return x
