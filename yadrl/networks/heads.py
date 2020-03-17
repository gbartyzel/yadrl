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


def _get_layer(layer_type, input_dim, output_dim, sigma_init):
    if layer_type == 'none':
        return nn.Linear(input_dim, output_dim)
    elif layer_type == 'factorized':
        return FactorizedNoisyLinear(input_dim, output_dim, sigma_init)
    elif layer_type == 'independent':
        return IndependentNoisyLinear(input_dim, output_dim, sigma_init)
    raise ValueError(
        'Wrong layer type, choose between: none, factorized, independent')


class ValueHead(nn.Module):
    def __init__(self,
                 phi: nn.Module,
                 support_dim: int = 1,
                 distribution_type: str = 'none',
                 noise_type: str = 'none',
                 sigma_init: float = 0.5):
        super(ValueHead, self).__init__()
        assert distribution_type in ('none', 'categorical', 'quantile')
        self._support_dim = 1 if distribution_type == 'none' else support_dim
        self._enable_noise = noise_type != 'none'
        self._distribution_type = distribution_type

        self._phi = phi
        self._value = _get_layer(
            layer_type=noise_type,
            input_dim=list(phi.parameters())[-1].shape[0],
            output_dim=self._support_dim,
            sigma_init=sigma_init)

        if not self._enable_noise:
            self._initialize_variables()

    def _initialize_variables(self):
        self._value.weight.data.uniform_(-3e-3, 3e-3)
        self._value.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self,
                x: torch.Tensor,
                sample_noise: bool = False) -> torch.Tensor:
        self.reset_noise()
        if sample_noise:
            self.sample_noise()
        x = self._phi(x)
        out = self._value(x)
        if self._distribution_type == 'none':
            return out
        else:
            probs = out.view(-1, self._support_dim)
            if self._distribution_type == 'categorical':
                return F.softmax(probs, dim=-1)
            return probs

    def sample_noise(self):
        if self._enable_noise:
            self._value.sample_noise()

    def reset_noise(self):
        if self._enable_noise:
            self._value.reset_noise()


class DQNHead(nn.Module):
    def __init__(self,
                 phi: nn.Module,
                 output_dim: int,
                 dueling: bool = False,
                 support_dim: int = 1,
                 distribution_type: str = 'none',
                 noise_type: str = 'none',
                 sigma_init: float = 0.5):
        super(DQNHead, self).__init__()
        assert distribution_type in ('none', 'categorical', 'quantile')
        self._support_dim = 1 if distribution_type == 'none' else support_dim
        self._output_dim = output_dim
        self._enable_noise = noise_type != 'none'
        self._dueling = dueling
        self._distribution_type = distribution_type

        self._phi = phi
        self._head = _get_layer(
            layer_type=noise_type,
            input_dim=list(phi.parameters())[-1].shape[0],
            output_dim=output_dim * self._support_dim,
            sigma_init=sigma_init)
        if dueling:
            self._value = _get_layer(
                layer_type=noise_type,
                input_dim=list(phi.parameters())[-1].shape[0],
                output_dim=self._support_dim,
                sigma_init=sigma_init)
        if not self._enable_noise:
            self._initialize_variables()

    def _initialize_variables(self):
        self._head.weight.data.uniform_(-3e-3, 3e-3)
        self._head.bias.data.uniform_(-3e-3, 3e-3)
        if self._dueling:
            self._value.weight.data.uniform_(-3e-3, 3e-3)
            self._value.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self,
                x: torch.Tensor,
                sample_noise: bool = False):
        x = self._phi(x)
        self.reset_noise()
        if sample_noise:
            self.sample_noise()
        if self._distribution_type != 'none':
            return self._distributional_forward(x)
        return self._forward(x)

    def _forward(self, x: torch.Tensor):
        out = self._head(x)
        if self._dueling:
            out += (self._value(x) - out.mean(dim=1, keepdim=True))
        return out

    def _distributional_forward(self, x: torch.Tensor):
        out = self._head(x).view(-1, self._output_dim, self._support_dim)
        if self._dueling:
            value = self._value(x).view(-1, 1, self._support_dim)
            out += (value - out.mean(dim=-1, keepdim=True))
        if self._distribution_type == 'categorical':
            return F.softmax(out, dim=-1)
        return out

    def sample_noise(self):
        if self._enable_noise:
            self._head.sample_noise()
            if self._dueling:
                self._value.sample_noise()

    def reset_noise(self):
        if self._enable_noise:
            self._head.reset_noise()
            if self._dueling:
                self._value.reset_noise()


class MultiValueHead(nn.Module):
    def __init__(self,
                 phi: Union[nn.Module, Tuple[nn.Module]],
                 heads_num: int = 2, **kwargs):
        super(MultiValueHead, self).__init__()
        if not isinstance(phi, tuple):
            phi = (phi,) * heads_num
        self._heads = nn.ModuleList([ValueHead(copy.deepcopy(phi[i]), **kwargs)
                                     for i in range(heads_num)])

    def parameters(self, item: Optional[int] = None) -> Iterator[nn.Parameter]:
        if item is not None:
            return self._heads[item].parameters()
        return super().parameters()

    def named_parameters(self,
                         prefix: str = '',
                         recurse: bool = True,
                         item: Optional[int] = None
                         ) -> Iterator[Tuple[str, nn.Parameter]]:
        if item is not None:
            return self._heads[item].named_parameters(prefix, recurse)
        return super().named_parameters(prefix, recurse)

    def forward(self,
                x: Tuple[torch.Tensor, ...],
                train: bool = False,
                unsqueeze: bool = False) -> Tuple[torch.Tensor, ...]:
        if unsqueeze:
            return tuple(head(x, train).unsqueeze(1) for head in self._heads)
        return tuple([head(x, train) for head in self._heads])


class MultiDQNHead(nn.Module):
    def __init__(self,
                 phi: Union[nn.Module, Tuple[nn.Module]],
                 heads_num: int = 2, **kwargs):
        super(MultiDQNHead, self).__init__()
        if not isinstance(phi, tuple):
            phi = (phi,) * heads_num
        self._heads = nn.ModuleList([DQNHead(copy.deepcopy(phi[i]), **kwargs)
                                     for i in range(heads_num)])

    def parameters(self, item: Optional[int] = None) -> Iterator[nn.Parameter]:
        if item is not None:
            return self._heads[item].parameters()
        return super().parameters()

    def named_parameters(self,
                         prefix: str = '',
                         recurse: bool = True,
                         item: Optional[int] = None
                         ) -> Iterator[Tuple[str, nn.Parameter]]:
        if item is not None:
            return self._heads[item].named_parameters(prefix, recurse)
        return super().named_parameters(prefix, recurse)

    def forward(self,
                x: torch.Tensor,
                train: bool = False,
                unsqueeze: bool = False) -> Tuple[torch.Tensor, ...]:
        if unsqueeze:
            return tuple(head(x, train).unsqueeze(1) for head in self._heads)
        return tuple(head(x, train) for head in self._heads)


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

if __name__ == '__main__':
    from yadrl.networks.bodies import MLPNetwork

    heads = GaussianActorCritic(
        phi=MLPNetwork(10, (64, 64)),
        output_dim=2,
        squash=True,
        independent_std=False
    )

