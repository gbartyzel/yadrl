import numpy as np
import torch

import yadrl.common.ops as ops
from yadrl.agents.dqn.dqn import DQN


class QuantileDQN(DQN, agent_type='quantile_regression_dqn'):
    head_types = ['quantile', 'dueling_quantile']

    def __init__(self, support_dim: int = 51, **kwargs):
        self._support_dim = support_dim
        super().__init__(**kwargs)
        self._cumulative_density = torch.from_numpy(
            (np.arange(support_dim) + 0.5) / support_dim
        ).float().unsqueeze(0).to(self._device)

    def _sample_q(self, state: torch.Tensor,
                  train: bool = False) -> torch.Tensor:
        return super()._sample_q(state, train).mean(-1)

    def _compute_loss(self, batch):
        batch_vec = torch.arange(self._batch_size).long()
        with torch.no_grad():
            self.target_model.sample_noise()
            next_quantiles = self.target_model(batch.next_state)
            if self._use_double_q:
                self.model.sample_noise()
                next_q = self.model(batch.next_state).mean(-1)
            else:
                next_q = next_quantiles.mean(-1)
            next_action = next_q.argmax(-1).long()
            next_quantiles = next_quantiles[batch_vec, next_action, :]

            target_quantiles = ops.td_target(
                batch.reward, batch.mask, next_quantiles,
                batch.discount_factor * self._discount)

        action = batch.action.long().squeeze()
        self.model.sample_noise()
        expected_quantiles = self.model(batch.state)
        expected_quantiles = expected_quantiles[batch_vec, action, :]

        loss = ops.quantile_hubber_loss(expected_quantiles, target_quantiles,
                                        self._cumulative_density)
        return loss
