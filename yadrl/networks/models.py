from copy import deepcopy
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from yadrl.networks.heads import CategoricalPolicyHead
from yadrl.networks.heads import DeterministicPolicyHead
from yadrl.networks.heads import GaussianPolicyHead
from yadrl.networks.heads import ValueHead


class DQNModel(nn.Module):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 dueling: bool = False,
                 noise_type: str = 'none',
                 sigma_init: float = 0.5):
        super(DQNModel, self).__init__()
        self._dueling = dueling
        self._phi = deepcopy(phi)

        self._advantage = ValueHead(input_dim=self._phi.output_dim,
                                    output_dim=output_dim,
                                    noise_type=noise_type,
                                    sigma_init=sigma_init)
        if dueling:
            self._value = ValueHead(input_dim=self._phi.output_dim,
                                    output_dim=1,
                                    noise_type=noise_type,
                                    sigma_init=sigma_init)

    def forward(self,
                x: torch.Tensor,
                sample_noise: bool = False) -> Dict[str, torch.Tensor]:
        self.reset_noise()
        if sample_noise:
            self.sample_noise()
        x = self._phi(x)

        adv = self._advantage(x)
        q_val = adv
        if self._dueling:
            value = self._value(x).expand_as(adv)
            q_val = value + adv - adv.mean(dim=1, keepdim=True).expand_as(adv)
        return q_val

    def sample_noise(self):
        self._advantage.sample_noise()
        if self._dueling:
            self._value.sample_noise()

    def reset_noise(self):
        self._advantage.reset_noise()
        if self._dueling:
            self._value.reset_noise()


class CategoricalDQNModel(DQNModel):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 atoms_dim: int,
                 dueling: bool = False,
                 noise_type: str = 'none',
                 sigma_init: float = 0.5):
        super(CategoricalDQNModel, self).__init__(
            phi, output_dim * atoms_dim, dueling, noise_type, sigma_init)

        self._output_dim = output_dim
        self._atoms_dim = atoms_dim

        if dueling:
            self._value = ValueHead(input_dim=self._phi.output_dim,
                                    output_dim=atoms_dim,
                                    noise_type=noise_type,
                                    sigma_init=sigma_init)

    def forward(self,
                x: torch.Tensor,
                sample_noise: bool = False) -> Dict[str, torch.Tensor]:
        self.reset_noise()
        if sample_noise:
            self.sample_noise()
        x = self._phi(x)

        adv = self._advantage(x).view(-1, self._output_dim, self._atoms_dim)
        probs = adv
        if self._dueling:
            value = self._value(x).view(-1, 1, self._atoms_dim).expand_as(adv)
            probs = value + adv - adv.mean(dim=-1, keepdim=True).expand_as(adv)
        return F.softmax(probs, dim=-1)


class QuantileDQNModel(DQNModel):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 quantiles_dim: int,
                 noise_type: str = 'none',
                 sigma_init: float = 0.5):
        super(QuantileDQNModel, self).__init__(
            phi, output_dim * quantiles_dim, False, noise_type, sigma_init)

        self._output_dim = output_dim
        self._quantiles_dim = quantiles_dim

    def forward(self,
                x: torch.Tensor,
                sample_noise: bool = False) -> Dict[str, torch.Tensor]:
        self.reset_noise()
        if sample_noise:
            self.sample_noise()
        x = self._phi(x)

        return self._advantage(x).view(-1, self._output_dim,
                                       self._quantiles_dim)


class Critic(nn.Module):
    def __init__(self, phi):
        super(Critic, self).__init__()
        self._phi = deepcopy(phi)
        self._value = ValueHead(self._phi.output_dim)

    def forward(self, *x: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self._value(self._phi(x))


class DoubleCritic(nn.Module):
    def __init__(self, phi: Tuple[nn.Module, nn.Module]):
        super(DoubleCritic, self).__init__()
        self._phi_1 = deepcopy(phi[0])
        self._phi_2 = deepcopy(phi[1])
        self._value_1 = ValueHead(self._phi_1.output_dim)
        self._value_2 = ValueHead(self._phi_2.output_dim)

    def q1_parameters(self):
        return tuple(self._phi_1.parameters()) \
               + tuple(self._value_1.parameters())

    def q2_parameters(self):
        return tuple(self._phi_2.parameters()) \
               + tuple(self._value_2.parameters())

    def forward(self,
                *x: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return self._value_1(self._phi_1(x)), self._value_2(self._phi_2(x))

    def eval_v1(self, *x: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self._value_1(self._phi_1(x))

    def eval_v2(self, *x: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self._value_2(self._phi_1(x))


class DeterministicActor(nn.Module):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 fan_init: bool = False,
                 activation_fn: Callable = torch.tanh):
        super(DeterministicActor, self).__init__()
        self._phi = deepcopy(phi)
        self._head = DeterministicPolicyHead(
            self._phi.output_dim, output_dim, fan_init, activation_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(self._phi(x))


class GaussianActor(nn.Module):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 std_limits: Tuple[float, float] = (-20.0, 2.0),
                 independent_std: bool = False,
                 squash: bool = True,
                 reparameterize: bool = True,
                 fan_init: bool = True):
        super(GaussianActor, self).__init__()
        self._phi = phi
        self._head = GaussianPolicyHead(
            self._phi.output_dim, output_dim, std_limits,
            independent_std, squash, reparameterize, fan_init)

    def forward(self,
                x: torch.Tensor,
                raw_action: Optional[torch.Tensor] = None,
                deterministic: bool = False) -> Tuple[torch.Tensor, ...]:
        return self._head.sample(self._phi(x), raw_action, deterministic)


class CategoricalActor(nn.Module):
    def __init__(self, phi: nn.Module, output_dim: int):
        super(CategoricalActor, self).__init__()
        self._phi = deepcopy(phi)
        self._head = CategoricalPolicyHead(self._phi.output_dim, output_dim)

    def forward(self,
                x: torch.Tensor,
                action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor,
                                                                ...]:
        return self._head.sample(self._phi(x), action)


class CategoricalActorCritic(nn.Module):
    def __init__(self, phi: nn.Module, output_dim: int):
        super(CategoricalActorCritic, self).__init__()
        self._phi = deepcopy(phi)
        self._value = ValueHead(self._phi.output_dim)
        self._policy = CategoricalPolicyHead(self._phi.output_dim, output_dim)

    def forward(self,
                x: torch.Tensor,
                action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor,
                                                                ...]:
        x = self._phi(x)
        value = self._value(x)
        action, log_prob, entropy = self._policy.sample(x, action)
        return action, log_prob, entropy, value


class GaussianActorCritic(nn.Module):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 std_limits: Tuple[float, float] = (-20.0, 2.0),
                 independent_std: bool = True,
                 squash: bool = True,
                 fan_init: bool = True):
        super(GaussianActorCritic, self).__init__()
        self._phi = deepcopy(phi)
        self._value = ValueHead(self._phi.output_dim)
        self._policy = GaussianPolicyHead(
            self._phi.output_dim, output_dim, std_limits,
            independent_std, squash, fan_init)

    def forward(self,
                x: torch.Tensor,
                action: Optional[torch.Tensor] = None, ) -> Tuple[torch.Tensor,
                                                                  ...]:
        x = self._phi(x)
        value = self._value(x)
        action, log_prob, entropy = self._policy.sample(x, action)
        return action, log_prob, entropy, value
