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
