from typing import Tuple

import torch as th
import torch.distributions as td

from yadrl.networks.body import Body
from yadrl.networks.head import Head


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
