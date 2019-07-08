from typing import Sequence, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(x: nn.Parameter):
    size = x.data.size()[0]
    val = 1 / np.sqrt(size)
    return -val, val


def init(x: nn.Module):
    classname = x.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.orthogonal_(x.weight.data, np.sqrt(2))
        nn.init.constant_(x.bias.data, 0.0)


class _BaseMLPNetwork(nn.Module):
    def __init__(self,
                 input_dim: Union[int, Tuple[int, ...]],
                 output_dim: int,
                 activation_fn: Callable = F.relu):
        super(_BaseMLPNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._activation_fn = activation_fn

    def forward(self, *input):
        return NotImplementedError


class MLPNetwork(_BaseMLPNetwork):
    def __init__(self,
                 input_dim: Union[int, Tuple[int, ...]],
                 hidden_dim: Tuple[int, ...],
                 activation_fn: nn.Module = nn.ReLU()):
        super(MLPNetwork, self).__init__(input_dim, hidden_dim[-1],
                                         activation_fn)

        self._size = len(hidden_dim)
        self._body = nn.Sequential()
        layers = (input_dim,) + hidden_dim
        for i in range(self._size):
            self._body.add_module('Linear_{}'.format(i),
                                  nn.Linear(layers[i], layers[i + 1]))
            self._body.add_module('Activation_{}'.format(i), activation_fn)
        self._body.apply(init)

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        if isinstance(x, tuple):
            x = torch.cat(x, dim=1)
        return self._body(x)


class DDPGMLPNetwork(_BaseMLPNetwork):
    def __init__(self,
                 input_dim: Tuple[int, int],
                 hidden_dim: Tuple[int, int],
                 activation_fn: Callable = F.relu):
        super(DDPGMLPNetwork, self).__init__(input_dim, hidden_dim[-1],
                                             activation_fn)

        self._size = len(hidden_dim)
        self._dense_1 = nn.Linear(input_dim[0], hidden_dim[0])
        self._dense_2 = nn.Linear(input_dim[1] + hidden_dim[0], hidden_dim[1])
        self._initialize_variables()

    def _initialize_variables(self):
        self._dense_1.weight.data.uniform_(*hidden_init(self._dense_1.weight))
        self._dense_1.bias.data.fill_(0)
        self._dense_2.weight.data.uniform_(*hidden_init(self._dense_2.weight))
        self._dense_2.bias.data.fill_(0)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x = self._activation_fn(self._dense_1(x))
        x = torch.cat((x, u), dim=1)
        x = self._activation_fn(self._dense_2(x))
        return x
