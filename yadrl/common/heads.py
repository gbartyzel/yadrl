from copy import deepcopy
from typing import Callable, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


class DQNHead(nn.Module):
    def __init__(self, phi: nn.Module, output_dim: int):
        super(DQNHead, self).__init__()
        self._phi = phi
        self._q_value = nn.Linear(self._phi.output_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = self._phi(x)
        return self._q_value(x)


class DuelingDQNHead(nn.Module):
    def __init__(self, phi: nn.Module, output_dim: int):
        super(DuelingDQNHead, self).__init__()
        self._phi = phi

        self._advantage = nn.Linear(self._phi.output_dim, output_dim)
        self._value = nn.Linear(self._phi.output_dim, 1)

    def forward(self, x: torch.Tensor):
        x = self._phi(x)
        advantage = self._advantage(x)
        value = self._value(x)

        return value + advantage - advantage.mean(dim=1, keepdim=True)


class ValueHead(nn.Module):
    def __init__(self, phi: nn.Module, q_value: bool = False, ddpg_init: bool = False):
        super(ValueHead, self).__init__()
        self._q_value = q_value
        self._phi = phi
        self._value = nn.Linear(self._phi.output_dim, 1)

        if ddpg_init:
            self._initialize_variables()

    def _initialize_variables(self):
        self._value.weight.data.uniform_(-3e-3, 3e-3)
        self._value.bias.data.uniform(-3e-3, 3e-3)

    def forward(self, *x):
        if self._q_value:
            return self._value(self._phi(*x))
        return self._value(self._phi(*x))


class DoubleQValueHead(nn.Module):
    def __init__(self, phi: nn.Module):
        super(DoubleQValueHead, self).__init__()
        self._q1_value = ValueHead(deepcopy(phi), q_value=True)
        self._q2_value = ValueHead(deepcopy(phi), q_value=True)

    def forward(self, *x):
        return self._q1_value(*x), self._q2_value(*x)

    def eval_q1(self, *x):
        return self._q1_value(*x)

    def eval_q2(self, *x):
        return self._q2_value(*x)


class DeterministicPolicyHead(nn.Module):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 ddpg_init: bool = False,
                 activation_fn: Callable = F.tanh):
        super(DeterministicPolicyHead, self).__init__()
        self._activation_fn = activation_fn
        self._phi = phi

        self._action = nn.Linear(self._phi.output_dim, output_dim)

        if ddpg_init:
            self._initialize_variables()

    def _initialize_variables(self):
        self._action.weight.data.uniform_(-3e-3, 3e-3)
        self._action.bias.data.uniform(-3e-3, 3e-3)

    def forward(self, x: torch.Tensor):
        x = self._phi(x)
        if self._activation_fn:
            return self._activation_fn(self._action(x))
        return self._action(x)


class GaussianPolicyHead(nn.Module):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 independent_std: bool = True,
                 squash: bool = False,
                 std_limits: Sequence[float] = (-20.0, 2.0)):
        super(GaussianPolicyHead, self).__init__()
        self._independend_std = independent_std
        self._squash = squash
        self._std_limits = std_limits

        self._phi = phi
        self._mean = nn.Linear(self._phi.output_dim, output_dim)
        if independent_std:
            self._log_std = nn.Parameter(torch.zeros(1, output_dim))
        else:
            self._log_std = nn.Linear(self._phi.output_dim, output_dim)
            self._initialize_parameters()

    def _initialize_parameters(self):
        self._log_std.weight.data.uniform_(-3e-3, 3e-3)
        self._log_std.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        x = self._phi(x)
        mean = self._mean(x)
        log_std = self._log_std.expand_as(mean) if self._independend_std else self._log_std(x)
        log_std = torch.clamp(log_std, *self._std_limits)
        return mean, log_std

    def sample(self, state: torch.Tensor,
               raw_action: Optional[torch.Tensor] = None,
               reparameterize: bool = True):
        mean, log_std = self.forward(state)
        covariance = torch.diag_embed(torch.exp(log_std))
        dist = MultivariateNormal(loc=mean, scale_tril=covariance)

        if not raw_action:
            raw_action = dist.rsample() if reparameterize else dist.sample()

        action = torch.tanh(raw_action) if self._squash else raw_action
        log_prob = dist.log_prob(raw_action).unsqueeze(-1)
        if self._squash:
            log_prob -= self._squash_correction(raw_action)
        entropy = dist.entropy().unsqueeze(-1)

        return action, log_prob, entropy, torch.tanh(mean)

    @staticmethod
    def _squash_correction(action: torch.Tensor, eps: float = 1e-6):
        return torch.log(1.0 - torch.tanh(action).pow(2) + eps).sum(-1, keepdim=True)
