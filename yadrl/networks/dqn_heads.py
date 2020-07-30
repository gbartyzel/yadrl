import copy
from typing import Callable
from typing import Iterator
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

from yadrl.networks.noisy_linear import FactorizedNoisyLinear
from yadrl.networks.noisy_linear import IndependentNoisyLinear
from yadrl.networks.bodies import NoisyMlpNetwork, MLPNetwork


class DQNHead(nn.Module):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 hidden_size: Tuple[int, ...] = ()):
        super(DQNHead, self).__init__()
        self._phi = phi
        self._output_dim = output_dim

        if len(hidden_size):
            self._head_hidden = MLPNetwork(
                input_dim=list(phi.parameters())[-1].shape[0],
                hidden_dim=hidden_size)
            self._head = nn.Linear(hidden_size[-1], output_dim)
        else:
            self._head_hidden = nn.Identity()
            self._head = nn.Linear(list(phi.parameters())[-1].shape[0],
                                   output_dim)
        self.initialize_parameters()

    def initialize_parameters(self):
        self._head.weight.data.uniform_(-3e-3, 3e-3)
        self._head.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self,
                x: torch.Tensor,
                sample_noise: bool = False) -> torch.Tensor:
        x = self._phi(x)
        x = self._head_hidden(x)
        return self._head(x)

    def sample_noise(self):
        pass

    def reset_noise(self):
        pass


class NoisyDQNHead(DQNHead):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 hidden_size: Tuple[int, ...] = (),
                 noise_type: str = 'factorized',
                 sigma_init: float = 0.5):
        super().__init__()
        self._head = NoisyMlpNetwork(

        )


class DuelingDQNHead(DQNHead):
    def __init__(self, **kwargs):
        self._value = nn.Sequential()
        super().__init__(**kwargs)


if __name__ == '__main__':
    mlp = MLPNetwork(3, ())
