from typing import Sequence, Tuple

import torch as th
import torch.nn as nn

from yadrl.networks.body import Body
from yadrl.networks.layer import Layer


class Head(nn.Module):
    registered_heads = {}

    noise_map = {
        'none': 'linear',
        'factorized': 'factorized_noisy_linear',
        'independent': 'independent_noisy_linear'
    }

    def __init_subclass__(cls, head_type: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if head_type:
            cls.registered_heads[head_type] = cls

    @classmethod
    def build(cls, head_type: str, **kwargs) -> 'Head':
        return cls.registered_heads[head_type](**kwargs)

    def __init__(self,
                 phi: Body,
                 output_dim: int,
                 support_dim: int = 1,
                 noise_type: str = 'none',
                 hidden_activation: str = 'relu',
                 output_activation: str = 'none',
                 hidden_dim: Sequence[int] = None):
        super().__init__()
        self._hidden_dim = () if hidden_dim is None else hidden_dim
        self._output_dim = output_dim
        self._support_dim = support_dim

        self._layer_type = self.noise_map[noise_type]
        self._hidden_act_fn = hidden_activation
        self._output_act_fn = output_activation

        self._phi = phi
        self._heads = nn.ModuleList([
            self._make_module(self._output_dim * self._support_dim)])
        self.reset_noise()

    def forward(self, *input_data: th.Tensor) -> th.Tensor:
        out = self._phi(input_data)
        return out

    def sample_noise(self):
        self._phi.sample_noise()
        for head in self._heads:
            for layer in head:
                layer.sample_noise()

    def reset_noise(self):
        self._phi.reset_noise()
        for head in self._heads:
            for layer in head:
                layer.reset_noise()

    def _make_module(self, output_dim):
        dims = (self._phi.output_dim,) + self._hidden_dim + (output_dim,)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(Layer.build(
                in_dim=dims[i], out_dim=dims[i + 1],
                activation=self._get_act_fn(i, len(dims)),
                layer_type=self._layer_type))
        return nn.Sequential(*layers)

    def _get_act_fn(self, iteration, last_iteration):
        if iteration + 1 == last_iteration - 2:
            return self._output_act_fn
        return self._hidden_act_fn


class SimpleHead(Head, head_type='simple'):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, *input_data: th.Tensor):
        out = super().forward(*input_data)
        return self._heads[0](out)


class QuantileHead(SimpleHead, head_type='quantile'):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, *input_data: th.Tensor):
        out = super().forward(*input_data)
        return out.view(-1, self._output_dim, self._support_dim)


class CategoricalHead(QuantileHead, head_type='categorical'):
    def __init__(self, **kwargs):
        super().__init__(output_activation='log_softmax', **kwargs)


class DuelingHead(Head, head_type='dueling'):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._heads.append(self._make_module(self._support_dim))

    def forward(self, *input_data: th.Tensor):
        out = super().forward(*input_data)
        advantage, value = [module(out) for module in self._moduels]
        advantage += value - advantage.mean(1, True)
        return advantage


class QuantileDuelingHead(DuelingHead, head_type='quantile_dueling'):
    def forward(self, *input_data: th.Tensor):
        out = super().forward(*input_data)
        advantage, value = [module(out) for module in self._moduels]
        advantage = advantage.view(-1, self._output_dim, self._support_dim)
        value = value.view(-1, 1, self._support_dim)
        advantage += value - advantage.mean(1, True)
        return advantage


class CategoricalDuelingHead(QuantileDuelingHead,
                             head_type='categorical_dueling'):
    def __init__(self, **kwargs):
        super().__init__(output_activation='log_softmax', **kwargs)


class MultiHead(Head, head_type='multi'):
    def __init__(self, num_heads: int = 2, **kwargs):
        super().__init__(**kwargs)
        delattr(self, '_phi')
        self._heads = nn.ModuleList([
            Head.build(head_type='simple', **kwargs) for _ in range(num_heads)])

    def forward(self, *input_data: th.Tensor) -> Tuple[th.Tensor, ...]:
        return tuple(head(*input_data) for head in self._heads)

    def sample_noise(self):
        for head in self._heads:
            head.sample_noise()

    def reset_noise(self):
        for head in self._heads:
            head.reset_noise()


class DistributionHead(Head):

    def __init__(self,
                 phi: Body,
                 output_dim: int,
                 reparameterize: bool = False,
                 output_activation: str = 'none'):
        super().__init__(phi=phi, output_dim=output_dim,
                         output_activation=output_activation)
        self._reparameterize: bool = reparameterize
        self._dist: td.Distribution = None

    def sample(self, state: th.Tensor) -> th.Tensor:
        pass

    def sample_deterministic(self, state: th.Tensor) -> th.Tensor:
        pass

    def get_action(self,
                   state: th.Tensor,
                   deterministic: bool = False) -> th.Tensor:
        if deterministic:
            return self.sample_deterministic(state)
        return self.sample(state)

    def log_prob(self, action: th.Tensor) -> th.Tensor:
        pass

    def entropy(self) -> th.Tensor:
        return self._dist.entropy()


class GaussianHead(DistributionHead, head_type='gaussian'):
    def __init__(self, output_dim, **kwargs):
        super().__init__(output_dim=output_dim * 2, reparameterize=True,
                         **kwargs)
        self._output_dim = output_dim

    def forward(self,
                input_data: th.Tensor,
                sample_noise: bool = False) -> th.Tensor:
        out = super().forward(input_data, sample_noise)
        mean, log_std = out.split(self._output_dim, -1)
        return mean, log_std

    def sample(self, state: th.Tensor) -> th.Tensor:
        mean, log_std = self.forward(state)
        self._dist = td.Normal(mean, log_std.exp())
        if self._reparameterize:
            return self._dist.rsample()
        return self._dist.sample()

    def sample_deterministic(self, state: th.Tensor) -> th.Tensor:
        mean, _ = self.forward(state)
        return mean

    def log_prob(self, action: th.Tensor) -> th.Tensor:
        log_prob = self._dist.log_prob(action)
        log_prob = th.clamp(log_prob, -100.0, 100.0)
        if len(log_prob.shape) == 2:
            return th.sum(log_prob, -1, True)
        return log_prob.view(-1, 1)


class SquashedGaussianHead(GaussianHead, head_type='squashed_gaussian'):
    def __init__(self,
                 log_std_limit: Tuple[float, float] = (-20.0, 2.0),
                 **kwargs):
        super().__init__(**kwargs)
        self._log_std_limit = log_std_limit

    def forward(self,
                input_data: th.Tensor,
                sample_noise: bool = False) -> th.Tensor:
        mean, log_std = super().forward(input_data)
        return mean, log_std.clamp(*self._log_std_limit)

    def sample(self, state: th.Tensor) -> th.Tensor:
        action = super().sample(state)
        return th.tanh(action)

    def sample_deterministic(self, state: th.Tensor) -> th.Tensor:
        action = super().sample_deterministic(state)
        return th.tanh(action)

    def log_prob(self, action: th.Tensor) -> th.Tensor:
        eps = th.finfo(action.dtype).eps
        gaussian_action = th.atanh(action.clamp(-1.0 + eps, 1.0 - eps))
        log_prob = super().log_prob(gaussian_action)
        log_prob -= th.log(1.0 - th.tanh(action).pow(2) + eps).sum(-1, True)
        return log_prob
