from copy import deepcopy

import torch
import torch.nn as nn


class DQNHead(nn.Module):
    def __init__(self, phi: nn.Module, output_dim: int):
        super(DQNHead, self).__init__()
        self._phi = phi
        self._q_value_head = nn.Linear(self._phi.output_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = self._phi(x)
        return self._q_value_head(x)


class DuelingDQNHead(nn.Module):
    def __init__(self, phi: nn.Module, output_dim: int):
        super(DuelingDQNHead, self).__init__()
        self._phi = phi

        self._advantage_head = nn.Linear(self._phi.output_dim, output_dim)
        self._value_head = nn.Linear(self._phi.output_dim, 1)

    def forward(self, x: torch.Tensor):
        x = self._phi(x)
        advantage = self._advantage_head(x)
        value = self._value_head(x)

        return value + advantage - advantage.mean(dim=1, keepdim=True)


class ValueHead(nn.Module):
    def __init__(self, phi: nn.Module, q_value: bool = False, ddpg_init: bool = False):
        super(ValueHead, self).__init__()
        self._q_value = q_value
        self._phi = phi
        self._value_head = nn.Linear(self._phi.output_dim, 1)

        if ddpg_init:
            self._initialize_variables()

    def _initialize_variables(self):
        self._action_head.weight.data.uniform_(-3e-3, 3e-3)
        self._action_head.bias.data.uniform(-3e-3, 3e-3)

    def forward(self, *x):
        if self._q_value:
            return self._value_head(self._phi(*x))
        return self._value_head(self._phi(*x))


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
                 activation_fn: nn.Module = nn.Tanh):
        super(DeterministicPolicyHead, self).__init__()
        self._activation_fn = activation_fn()
        self._phi = phi

        self._action_head = nn.Linear(self._phi.output_dim, output_dim)

        if ddpg_init:
            self._initialize_variables()

    def _initialize_variables(self):
        self._action_head.weight.data.uniform_(-3e-3, 3e-3)
        self._action_head.bias.data.uniform(-3e-3, 3e-3)

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        x = self._phi(x, u)
        return self._activation_fn(self._action_head(x))


class GaussianPolicyHead(nn.Module):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int):
        pass
