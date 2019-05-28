from typing import Sequence, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(x: nn.Parameter):
    size = x.data.size()[0]
    val = 1 / np.sqrt(size)
    return -val, val


def expand_input(*x: Sequence[torch.Tensor]):
    if all([len(y.shape) == 1 for y in x]):
        x = tuple(y.view(1, -1) for y in x)
    return torch.cat(x, dim=1)


class BaseMLPNetwork(nn.Module):
    def __init__(self,
                 input_dim: Union[int, Tuple[int, ...]],
                 output_dim: int,
                 activation_fn: Callable = F.relu):
        super(BaseMLPNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._activation_fn = activation_fn

    def forward(self, *input):
        return NotImplementedError


class MLPNetwork(BaseMLPNetwork):
    def __init__(self,
                 input_dim: Union[int, Tuple[int, ...]],
                 hidden_dim: Tuple[int, ...],
                 activation_fn: Callable = F.relu):
        super(MLPNetwork, self).__init__(input_dim, hidden_dim[-1], activation_fn)

        self._size = len(hidden_dim)
        layers = (input_dim, ) + hidden_dim
        for i in range(self._size):
            exec('self._dense_{0} = nn.Linear(layers[{0}], layers[{0} + 1])'.format(i))

    def forward(self, *x: Sequence[torch.Tensor]):
        x = expand_input(*x)
        for i in range(self._size):
            x = self._activation_fn(eval('self._dense_{}(x)'.format(i)))
        return x


class DDPGMLPNetwork(BaseMLPNetwork):
    def __init__(self,
                 input_dim: Tuple[int, ...],
                 hidden_dim: Tuple[int, ...],
                 activation_fn: Callable = F.relu):
        super(DDPGMLPNetwork, self).__init__(input_dim, hidden_dim[-1], activation_fn)

        self._size = len(hidden_dim)
        l = (input_dim[0],) + tuple(hidden_dim)
        for i in range(self._size):
            if i == 1:
                exec('self._dense_{0} = nn.Linear(l[{0}] + input_dim[{0}], l[{0} + 1])'.format(i))
            else:
                exec('self._dense_{0} = nn.Linear(l[{0}], l[{0} + 1])'.format(i))

        self._initialize_variables()

    def _initialize_variables(self):
        for i in range(self._size):
            weight_vals = hidden_init(eval('self._dense_{}.weight'.format(i)))
            eval('self._dense_{}.weight.data.uniform_(*weight_vals)'.format(i))
            bias_vals = hidden_init(eval('self._dense_{}.bias'.format(i)))
            eval('self._dense_{0}.bias.data.uniform_(*bias_vals)'.format(i))

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        for i in range(self._size):
            if i == 1:
                x = expand_input(x, u)
            x = self._activation_fn(eval('self._dense_{}(x)'.format(i)))
        return x
