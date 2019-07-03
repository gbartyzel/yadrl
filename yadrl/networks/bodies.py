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
                 input_dim: Tuple[int, ...],
                 hidden_dim: Tuple[int, ...],
                 activation_fn: Callable = F.relu):
        super(DDPGMLPNetwork, self).__init__(input_dim, hidden_dim[-1],
                                             activation_fn)

        self._size = len(hidden_dim)
        layers = (input_dim[0],) + tuple(hidden_dim)
        for i in range(self._size):
            if i == 1:
                exec('self._dense_{} = nn.Linear({}, {})'.format(
                    i, layers[i] + input_dim[i], layers[i + 1]))
            else:
                exec('self._dense_{} = nn.Linear({}, {})'.format(
                    i, layers[i], layers[i + 1]))

        self._initialize_variables()

    def _initialize_variables(self):
        for i in range(self._size):
            weight_vals = hidden_init(eval('self._dense_{}.weight'.format(i)))
            eval('self._dense_{}.weight.data.uniform_(*weight_vals)'.format(i))

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        for i in range(self._size):
            if i == 1:
                x = torch.cat((x, u), dim=1)
            x = self._activation_fn(eval('self._dense_{}(x)'.format(i)))
        return x
