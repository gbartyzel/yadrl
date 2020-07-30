from typing import Callable
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from yadrl.networks.noisy_linear import FactorizedNoisyLinear
from yadrl.networks.noisy_linear import IndependentNoisyLinear


def fan_init(x: nn.Parameter):
    size = x.data.size()[1]
    val = 1 / np.sqrt(size)
    return -val, val


def orthogonal_init(x: nn.Module):
    classname = x.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.orthogonal_(x.weight.data, gain=np.sqrt(2))
        nn.init.constant_(x.bias.data, 0.0)


class BaseMLPNetwork(nn.Module):
    def __init__(self,
                 input_dim: Union[int, Tuple[int, ...]],
                 output_dim: int,
                 activation_fn: Union[Callable, nn.Module] = F.relu):
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
                 activation_fn: nn.Module = F.relu):
        super(MLPNetwork, self).__init__(input_dim, hidden_dim[-1],
                                         activation_fn)

        self._size = len(hidden_dim)
        self._body = nn.ModuleList()
        layers = (input_dim,) + hidden_dim
        for i in range(self._size):
            self._body.append(nn.Linear(layers[i], layers[i + 1]))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self._body:
            orthogonal_init(layer)

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        if isinstance(x, tuple):
            x = torch.cat(x, dim=1)
        for layer in self._body:
            x = self._activation_fn(layer(x))
        return x


class NoisyMlpNetwork(BaseMLPNetwork):
    layer_fn = {'factorized': FactorizedNoisyLinear,
             'independent': IndependentNoisyLinear}

    def __init__(self,
                 input_dim: Union[int, Tuple[int, ...]],
                 hidden_dim: Tuple[int, ...],
                 activation_fn: nn.Module = F.relu,
                 noise_type: str = 'factorized',
                 sigma_init: float = 0.5):
        super().__init__(input_dim, hidden_dim[-1], activation_fn)
        self._size = len(hidden_dim)

        self._body = nn.ModuleList()
        layers = (input_dim,) + hidden_dim
        for i in range(self._size):
            self._body.append(
                self.layer_fn[noise_type](layers[i], layers[i + 1], sigma_init))

    def sample_noise(self):
        for layer in self._body:
            layer.sample_noise()

    def reset_noise(self):
        for layer in self._body:
            layer.reset_noise()

    def forward(self,
                x: Sequence[torch.Tensor],
                sample_noise: bool = False) -> torch.Tensor:
        self.reset_noise()
        if sample_noise:
            self.sample_noise()

        if isinstance(x, tuple):
            x = torch.cat(x, dim=1)
        for layer in self._body:
            x = self._activation_fn(layer(x))
        return x


class BNMLPNetwork(BaseMLPNetwork):
    def __init__(self,
                 input_dim: Union[int, Tuple[int, ...]],
                 hidden_dim: Tuple[int, ...],
                 activation_fn: nn.Module = F.relu):
        super(BNMLPNetwork, self).__init__(input_dim, hidden_dim[-1],
                                           activation_fn)

        self._size = len(hidden_dim)
        self._body = nn.ModuleList()
        layers = (input_dim,) + hidden_dim
        for i in range(self._size):
            self._body.append(nn.Linear(layers[i], layers[i + 1]))
            self._body.append(nn.BatchNorm1d(layers[i + 1]))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self._body:
            orthogonal_init(layer)

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        if isinstance(x, tuple):
            x = torch.cat(x, dim=1)
        for i in range(len(self._body) // 2):
            x = self._body[2 * i + 1](self._body[2 * i](x))
            x = self._activation_fn(x)
        return x


class DDPGMLPNetwork(BaseMLPNetwork):
    def __init__(self,
                 input_dim: Tuple[int, int],
                 hidden_dim: Tuple[int, int],
                 activation_fn: Callable = F.relu):
        super(DDPGMLPNetwork, self).__init__(input_dim, hidden_dim[-1],
                                             activation_fn)

        self._size = len(hidden_dim)
        self._dense_1 = nn.Linear(input_dim[0], hidden_dim[0])
        self._dense_2 = nn.Linear(input_dim[1] + hidden_dim[0], hidden_dim[1])
        self.reset_parameters()

    def reset_parameters(self):
        self._dense_1.weight.data.uniform_(*fan_init(self._dense_1.weight))
        self._dense_1.bias.data.fill_(0)
        self._dense_2.weight.data.uniform_(*fan_init(self._dense_2.weight))
        self._dense_2.bias.data.fill_(0)

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        u = x[1]
        x = self._activation_fn(self._dense_1(x[0]))
        x = torch.cat((x, u), dim=1)
        x = self._activation_fn(self._dense_2(x))
        return x


class DDPGBNMLPNetwork(BaseMLPNetwork):
    def __init__(self,
                 input_dim: Tuple[int, int],
                 hidden_dim: Tuple[int, int],
                 activation_fn: Callable = F.relu):
        super(DDPGBNMLPNetwork, self).__init__(input_dim, hidden_dim[-1],
                                               activation_fn)

        self._size = len(hidden_dim)
        self._dense_1 = nn.Linear(input_dim[0], hidden_dim[0])
        self._bn_1 = nn.BatchNorm1d(hidden_dim[0])
        self._dense_2 = nn.Linear(input_dim[1] + hidden_dim[0], hidden_dim[1])
        self._bn_2 = nn.BatchNorm1d(hidden_dim[1])
        self.reset_parameters()

    def reset_parameters(self):
        self._dense_1.weight.data.uniform_(*fan_init(self._dense_1.weight))
        self._dense_1.bias.data.fill_(0)
        self._dense_2.weight.data.uniform_(*fan_init(self._dense_2.weight))
        self._dense_2.bias.data.fill_(0)

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        u = x[1]
        x = self._activation_fn(self._bn_1(self._dense_1(x[0])))
        x = torch.cat((x, u), dim=1)
        x = self._activation_fn(self._bn_2(self._dense_2(x)))
        return x
