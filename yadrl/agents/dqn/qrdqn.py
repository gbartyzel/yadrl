import torch as th

import yadrl.common.ops as ops
from yadrl.agents.dqn.dqn import DQN
from yadrl.common.memory import Batch


class QuantileDQN(DQN, agent_type='quantile_regression_dqn'):
    head_types = ['distribution_value', 'distribution_dueling_value']

    def __init__(self, support_dim: int = 51, **kwargs):
        self._support_dim = support_dim
        super().__init__(**kwargs)
        self._cumulative_density = th.arange(
            0.5 / self._support_dim, 1.0, 1.0 / self._support_dim,
            device=self._device, dtype=th.float32).unsqueeze(0)

    def _sample_q(self, state: th.Tensor, train: bool = False) -> th.Tensor:
        return super()._sample_q(state, train).mean(-1)

    def _compute_loss(self, batch: Batch) -> th.Tensor:
        batch_vec = th.arange(self._batch_size).long()
        with th.no_grad():
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
