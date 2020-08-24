from copy import deepcopy
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from yadrl.networks.commons import get_layer


class ValueHead(nn.Module):
    def __init__(self,
                 phi: nn.Module,
                 noise_type: str = 'none',
                 sigma_init: float = 0.5):
        super().__init__()
        self._enable_noise = noise_type != 'none'
        self._phi = phi
        self._head = self._create_head(list(phi.parameters())[-1].shape[0],
                                       noise_type, sigma_init)
        self._initialize_variables()

    def _create_head(self,
                     input_dim: int,
                     layer_type: str,
                     sigma_init: float) -> nn.Module:
        return get_layer(layer_type=layer_type,
                         input_dim=input_dim,
                         output_dim=1,
                         sigma_init=sigma_init)

    def _initialize_variables(self):
        if not self._enable_noise:
            self._head.weight.data.uniform_(-3e-3, 3e-3)
            self._head.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self,
                x: torch.Tensor,
                sample_noise: bool = False,
                **kwargs) -> torch.Tensor:
        self.reset_noise()
        if sample_noise:
            self.sample_noise()
        x = self._phi(x)
        return self._head(x)

    def sample_noise(self):
        if self._enable_noise:
            self._head.sample_noise()

    def reset_noise(self):
        if self._enable_noise:
            self._head.reset_noise()


class QuantileValueHead(ValueHead):
    def __init__(self, support_dim=int, **kwargs):
        self._support_dim = support_dim
        super().__init__(**kwargs)

    def _create_head(self,
                     input_dim: int,
                     layer_type: str,
                     sigma_init: float) -> nn.Module:
        return get_layer(layer_type=layer_type,
                         input_dim=input_dim,
                         output_dim=self._support_dim,
                         sigma_init=sigma_init)

    def forward(self,
                x: torch.Tensor,
                sample_noise: bool = False,
                **kwargs) -> torch.Tensor:
        out = super().forward(x, sample_noise)
        return out.view(-1, self._support_dim)


class CategoricalValueHead(QuantileValueHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self,
                x: torch.Tensor,
                sample_noise: bool = False,
                log_prob: bool = False) -> torch.Tensor:
        x = super().forward(x)
        return F.log_softmax(x, -1) if log_prob else F.softmax(x, -1)


class DoubleValueHead(nn.Module):
    def __init__(self,
                 phi: Union[nn.Module, Tuple[nn.Module]],
                 noise_type: str = 'none',
                 sigma_init: float = 0.5):
        super().__init__()
        self._head_1 = ValueHead(deepcopy(phi), noise_type, sigma_init)
        self._head_2 = ValueHead(deepcopy(phi), noise_type, sigma_init)

    def forward(self,
                x: Tuple[torch.Tensor, ...],
                sample_noise: bool = False) -> Tuple[torch.Tensor, ...]:
        return self._head_1(x, sample_noise), self._head_2(x, sample_noise)

    def eval_head_1(self,
                    x: torch.Tensor,
                    sample_noise: bool = False) -> torch.Tensor:
        return self._head_1(x, sample_noise)

    def eval_head_2(self,
                    x: Tuple[torch.Tensor, ...],
                    sample_noise: bool = False) -> torch.Tensor:
        return self._head_2(x, sample_noise)

    def head_1_parameters(self):
        return self._head_1.parameters()

    def head_2_parameters(self):
        return self._head_2.parameters()
