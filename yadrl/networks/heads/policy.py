from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical


class DeterministicPolicyHead(nn.Module):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 fan_init: bool = True,
                 activation_fn: Callable = torch.tanh):
        super(DeterministicPolicyHead, self).__init__()
        self._activation_fn = activation_fn

        self._phi = phi
        self._head = nn.Linear(list(phi.parameters())[-1].shape[0], output_dim)

        if fan_init:
            self._initialize_variables()

    def _initialize_variables(self):
        self._head.weight.data.uniform_(-3e-3, 3e-3)
        self._head.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._phi(x)
        action = self._head(x)
        if self._activation_fn:
            action = self._activation_fn(action)
        return action


class GaussianPolicyHead(nn.Module):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 std_limits: Sequence[float] = (-20.0, 2.0),
                 independent_std: bool = True,
                 squash: bool = False,
                 reparameterize: bool = True,
                 fan_init: bool = True):
        super(GaussianPolicyHead, self).__init__()
        self._independend_std = independent_std
        self._squash = squash
        self._std_limits = std_limits
        self._reparameterize = reparameterize

        self._phi = phi
        self._mean = nn.Linear(list(phi.parameters())[-1].shape[0], output_dim)
        if independent_std:
            self._log_std = nn.Parameter(torch.zeros(1, output_dim))
        else:
            self._log_std = nn.Linear(list(phi.parameters())[-1].shape[0],
                                      output_dim)

        if fan_init:
            self._initialize_parameters()

    def _initialize_parameters(self):
        if self._independend_std:
            self._log_std.weight.data.uniform_(-3e-3, 3e-3)
            self._log_std.bias.data.uniform_(-3e-3, 3e-3)
        self._mean.weight.data.uniform_(-3e-3, 3e-3)
        self._mean.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self,
                x: torch.Tensor,
                raw_action: Optional[torch.Tensor] = None,
                deterministic: bool = False) -> Tuple[torch.Tensor, ...]:
        x = self._phi(x)
        mean = self._mean(x)
        if self._independend_std:
            log_std = self._log_std.expand_as(mean)
        else:
            log_std = self._log_std(x)
        log_std = torch.clamp(log_std, *self._std_limits)
        covariance = torch.diag_embed(log_std.exp())
        dist = MultivariateNormal(loc=mean, scale_tril=covariance)

        if not raw_action:
            if self._reparameterize:
                raw_action = dist.rsample()
            else:
                raw_action = dist.sample()

        action = torch.tanh(raw_action) if self._squash else raw_action
        log_prob = dist.log_prob(raw_action).unsqueeze(-1)
        if self._squash:
            log_prob -= self._squash_correction(raw_action)
        entropy = dist.entropy().unsqueeze(-1)

        if deterministic:
            action = torch.tanh(dist.mean)
        return action, log_prob, entropy

    @staticmethod
    def _squash_correction(action: torch.Tensor,
                           eps: float = 1e-6) -> torch.Tensor:
        return torch.log(
            1.0 - torch.tanh(action).pow(2) + eps).sum(-1, keepdim=True)


class GumbelSoftmaxPolicyHead(nn.Module):
    def __init__(self, phi: nn.Module, output_dim: int):
        super(GumbelSoftmaxPolicyHead, self).__init__()
        self._phi = phi
        self._logits = nn.Linear(list(phi.parameters())[-1].shape[0],
                                 output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self._logits.weight.data.uniform_(-3e-3, 3e-3)
        self._logits.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self,
                x: torch.Tensor,
                temperature: int = 1.0,
                action: Optional[torch.Tensor] = None,
                deterministic: bool = False) -> Tuple[torch.Tensor, ...]:
        temperature_tensor = torch.Tensor([temperature]).to(x.device)
        x = self._phi(x)
        x = self._logits(x)
        dist = RelaxedOneHotCategorical(temperature=temperature_tensor,
                                        logits=x)
        if not action:
            raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action).view(-1, 1)
        action = F.one_hot(torch.argmax(raw_action, dim=-1),
                           x.shape[-1]).float()
        action = (action - raw_action).detach() + raw_action
        if deterministic:
            action = F.softmax(x, dim=-1).argmax()
        return action, log_prob
