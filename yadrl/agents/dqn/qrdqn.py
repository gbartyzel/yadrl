import numpy as np
import torch

import yadrl.common.utils as utils
from yadrl.agents.dqn.dqn import DQN


class QuantileDQN(DQN, agent_type='quantile_regression_dqn'):
    head_types = ['quantile', 'dueling_quantile']

    def __init__(self, support_dim: int = 51, **kwargs):
        self._support_dim = support_dim
        super().__init__(**kwargs)
        self._cumulative_density = torch.from_numpy(
            (np.arange(support_dim) + 0.5) / support_dim
        ).float().unsqueeze(0).to(self._device)

    def _sample_q_value(self, state, train):
        return super()._sample_q_value(state, train).mean(-1)

    def _compute_loss(self, batch):
        state = self._state_normalizer(batch.state, self._device)
        next_state = self._state_normalizer(batch.next_state, self._device)

        batch_vec = torch.arange(self._batch_size).long()
        with torch.no_grad():
            next_quantiles = self.target_model(next_state, True)
            if self._use_double_q:
                next_q = self.model(next_state, True).mean(-1)
            else:
                next_q = next_quantiles.mean(-1)
            next_action = next_q.argmax(-1).long()
            next_quantiles = next_quantiles[batch_vec, next_action, :]

        target_quantiles = utils.td_target(
            reward=batch.reward,
            mask=batch.mask,
            target=next_quantiles,
            discount=batch.discount_factor * self._discount)

        action = batch.action.long().squeeze()
        expected_quantiles = self.model(state, True)
        expected_quantiles = expected_quantiles[batch_vec, action, :]

        loss = utils.quantile_hubber_loss(
            prediction=expected_quantiles,
            target=target_quantiles,
            cumulative_density=self._cumulative_density)

        return loss
