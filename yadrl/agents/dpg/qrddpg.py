import torch as th

import yadrl.common.ops as ops
from yadrl.agents.dpg.ddpg import DDPG
from yadrl.common.memory import Batch


class QuantileDDPG(DDPG, agent_type='quantile_regression_ddpg'):
    def __init__(self,
                 support_dim: int = 100,
                 **kwargs):
        self._support_dim = support_dim
        super().__init__(**kwargs)
        self._cumulative_density = th.arange(
            0.5 / self._support_dim, 1.0, 1.0 / self._support_dim,
            device=self._device, dtype=th.float32).unsqueeze(0)

    def _sample_q(self, state: th.Tensor, action: th.Tensor,
                  sample_noise: bool = False) -> th.Tensor:
        return super()._sample_q(state, action, sample_noise).mean(-1)

    def _compute_critic_loss(self, batch: Batch) -> th.Tensor:
        with th.no_grad():
            next_action = self.target_pi(batch.next_state)
            self.target_qv.sample_noise()
            next_quantiles = self.target_qv(batch.next_state, next_action)
            target_quantiles = ops.td_target(
                batch.reward, batch.mask, next_quantiles,
                batch.discount_factor * self._discount)

        self.qv.sample_noise()
        expected_quantiles = self.qv(batch.state, batch.action)
        loss = ops.quantile_hubber_loss(expected_quantiles, target_quantiles,
                                        self._cumulative_density)
        return loss
