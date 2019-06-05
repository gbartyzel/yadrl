from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from yadrl.networks.heads import ValueHead, DeterministicPolicyHead, \
    GaussianPolicyHead, CategoricalPolicyHead


class DQNModel(nn.Module):
    def __init__(self, phi: nn.Module, output_dim: int, dueling=False):
        super(DQNModel, self).__init__()
        self._dueling = dueling

        self._phi = phi
        self._advantage = nn.Linear(self._phi.output_dim, output_dim)
        if dueling:
            self._value = ValueHead(self._phi.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._phi(x)
        advantage = self._advantage(x)
        if self._dueling:
            value = self._value(x)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        return advantage


class Critic(nn.Module):
    def __init__(self, phi, ddpg_init=False):
        super(Critic, self).__init__()
        self._phi = phi
        self._value = ValueHead(self._phi.output_dim, ddpg_init)

    def forward(self, *x: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self._value(self._phi(x))


class DoubleCritic(nn.Module):
    def __init__(self, phi: Tuple[nn.Module, nn.Module], ddpg_init=False):
        super(DoubleCritic, self).__init__()
        self._phi_1 = phi[0]
        self._phi_2 = phi[1]
        self._value_1 = ValueHead(self._phi_1.output_dim, ddpg_init)
        self._value_2 = ValueHead(self._phi_2.output_dim, ddpg_init)

    def net_1_parameters(self):
        return tuple(self._phi_1.parameters()) \
               + tuple(self._value_1.parameters())

    def net_2_parameters(self):
        return tuple(self._phi_2.parameters()) \
               + tuple(self._value_2.parameters())

    def forward(self, *x: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return self._value_1(self._phi_1(*x)), self._value_2(self._phi_2(*x))

    def eval_v1(self, *x: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self._value_1(self._phi_1(*x))

    def eval_v2(self, *x: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self._value_2(self._phi_1(*x))


class DeterministicActor(nn.Module):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 ddpg_init: bool = False,
                 activation_fn: Callable = torch.tanh):
        super(DeterministicActor, self).__init__()
        self._phi = phi
        self._head = DeterministicPolicyHead(
            self._phi.output_dim, output_dim, ddpg_init, activation_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(self._phi(x))


class GaussianActor(nn.Module):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 independent_std: bool = True,
                 squash: bool = False,
                 std_limits: Tuple[float, float] = (-20.0, 2.0)):
        super(GaussianActor, self).__init__()
        self._phi = phi
        self._head = GaussianPolicyHead(
            self._phi.output_dim, output_dim, independent_std, squash,
            std_limits)

    def forward(self,
                x: torch.Tensor,
                raw_action: Optional[torch.Tensor] = None,
                reparameterize: bool = False) -> Tuple[torch.Tensor, ...]:
        return self._head.sample(self._phi(x), raw_action, reparameterize)


class CategoricalActor(nn.Module):
    def __init__(self, phi: nn.Module, output_dim: int):
        super(CategoricalActor, self).__init__()
        self._phi = phi
        self._head = CategoricalPolicyHead(self._phi.output_dim, output_dim)

    def forward(self,
                x: torch.Tensor,
                action: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, ...]:
        return self._head.sample(self._phi(x), action)


class CategoricalActorCritic(nn.Module):
    def __init__(self, phi: nn.Module, output_dim: int):
        super(CategoricalActorCritic, self).__init__()
        self._phi = phi
        self._value = ValueHead(self._phi.output_dim)
        self._policy = CategoricalPolicyHead(self._phi.output_dim, output_dim)

    def forward(self,
                x: torch.Tensor,
                action: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, ...]:
        x = self._phi(x)
        value = self._value(x)
        action, log_prob, entropy = self._policy.sample(x, action)
        return action, log_prob, entropy, value


class GaussianActorCritic(nn.Module):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 independent_std: bool = True,
                 squash: bool = True,
                 std_limits: Tuple[float, float] = (-20.0, 2.0)):
        super(GaussianActorCritic, self).__init__()
        self._phi = phi
        self._value = ValueHead(self._phi.output_dim)
        self._policy = GaussianPolicyHead(
            self._phi.output_dim, output_dim, independent_std, squash,
            std_limits)

    def forward(self,
                x: torch.Tensor,
                action: Optional[torch.Tensor] = None,
                reparametrize: bool = False) -> Tuple[torch.Tensor, ...]:
        x = self._phi(x)
        value = self._value(x)
        action, log_prob, entropy, raw_action = self._policy.sample(
            x, action, reparametrize)
        return action, log_prob, entropy, raw_action, value
