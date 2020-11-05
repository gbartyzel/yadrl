from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from yadrl.networks.body import Body
from yadrl.networks.commons import get_layer


class DQNHead(nn.Module):
    def __init__(self,
                 phi: Body,
                 output_dim: int,
                 noise_type: str = 'none',
                 sigma_init: float = 0.5):
        super(DQNHead, self).__init__()
        self._output_dim = output_dim
        self._enable_noise = noise_type != 'none'

        self._phi = phi
        self._head = get_layer(
            layer_type=noise_type,
            input_dim=list(phi.parameters())[-1].shape[0],
            output_dim=output_dim,
            sigma_init=sigma_init)
        self._initialize_variables()

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


class QuantileDQNHead(DQNHead):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 support_dim: int = 1,
                 noise_type: str = 'none',
                 sigma_init: float = 0.5):
        super().__init__(phi, output_dim * support_dim, noise_type, sigma_init)
        self._output_dim = output_dim
        self._support_dim = support_dim

    def forward(self,
                x: torch.Tensor,
                sample_noise: bool = False,
                **kwargs) -> torch.Tensor:
        return super().forward(x).view(-1, self._output_dim, self._support_dim)


class CategoricalDQNHead(QuantileDQNHead):
    def forward(self,
                x: torch.Tensor,
                sample_noise: bool = False,
                log_prob: bool = False,
                **kwargs) -> torch.Tensor:
        x = super().forward(x)
        return F.log_softmax(x, -1) if log_prob else F.softmax(x, -1)


class DuelingDQNHead(DQNHead):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 noise_type: str = 'none',
                 sigma_init: float = 0.5):
        super().__init__(phi, output_dim, noise_type, sigma_init)
        self._value = get_layer(
            layer_type=noise_type,
            input_dim=list(phi.parameters())[-1].shape[0],
            output_dim=1,
            sigma_init=sigma_init)
        self._initialize_variables()

    def _initialize_variables(self):
        super()._initialize_variables()
        if not self._enable_noise and hasattr(self, '_value'):
            self._value.weight.data.uniform_(-3e-3, 3e-3)
            self._value.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self,
                x: torch.Tensor,
                sample_noise: bool = False,
                **kwargs) -> torch.Tensor:
        self.reset_noise()
        if sample_noise:
            self.sample_noise()
        x = self._phi(x)
        # x = scale_gradient(x, 2 ** (-1 / 2))
        out = self._head(x)
        value = self._value(x)
        out += value - out.mean(dim=1, keepdim=True)
        return out

    def sample_noise(self):
        if self._enable_noise:
            self._head.sample_noise()
            self._value.sample_noise()

    def reset_noise(self):
        if self._enable_noise:
            self._head.reset_noise()
            self._value.reset_noise()


class QuantileDuelingDQNHead(DQNHead):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 support_dim: int,
                 noise_type: str = 'none',
                 sigma_init: float = 0.5):
        super().__init__(phi, output_dim * support_dim, noise_type, sigma_init)
        self._output_dim = output_dim
        self._support_dim = support_dim

        self._value = get_layer(
            layer_type=noise_type,
            input_dim=list(phi.parameters())[-1].shape[0],
            output_dim=support_dim,
            sigma_init=sigma_init)
        self._initialize_variables()

    def forward(self,
                x: torch.Tensor,
                sample_noise: bool = False,
                **kwargs) -> torch.Tensor:
        self.reset_noise()
        if sample_noise:
            self.sample_noise()
        x = self._phi(x)
        out = self._head(x).view(-1, self._output_dim, self._support_dim)
        value = self._value(x).view(-1, 1, self._support_dim)
        out += (value - out.mean(dim=-1, keepdim=True))
        return out


class CategoricalDuelingDQNHead(DQNHead):
    def forward(self,
                x: torch.Tensor,
                sample_noise: bool = False,
                log_prob: bool = False,
                **kwargs) -> torch.Tensor:
        x = super().forward(x)
        return F.log_softmax(x, -1) if log_prob else F.softmax(x, -1)


class DoubleDQNHead(nn.Module):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 noise_type: str = 'none',
                 sigma_init: float = 0.5):
        super().__init__()
        self._head_1 = DQNHead(deepcopy(phi), output_dim, noise_type,
                               sigma_init)
        self._head_2 = DQNHead(deepcopy(phi), output_dim, noise_type,
                               sigma_init)

    def forward(self,
                x: torch.Tensor,
                sample_noise: bool = False) -> Tuple[torch.Tensor, ...]:
        return self._head_1(x, sample_noise), self._head_2(x, sample_noise)

    def head_1_parameters(self):
        return self._head_1.parameters()

    def head_2_parameters(self):
        return self._head_2.parameters()


class GradScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input * ctx.scale, None


scale_gradient = GradScaler.apply

if __name__ == '__main__':
    a = [1]
    print(['relu'] * len(a[:-1]))
