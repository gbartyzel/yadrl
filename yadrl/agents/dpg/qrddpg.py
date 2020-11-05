"""import numpy as np
import torch

import yadrl.common.utils as utils
from yadrl.agents.dpg.ddpg import DDPG
from yadrl.common.memory import Batch
from yadrl.networks.heads.value import QuantileValueHead


class QuantileDDPG(DDPG):
    def __init__(self,
                 support_dim: int = 100,
                 **kwargs):
        self._support_dim = support_dim
        super().__init__(**kwargs)
        self._cumulative_density = torch.from_numpy(
            (np.arange(support_dim) + 0.5) / support_dim
        ).float().unsqueeze(0).to(self._device)

    def _initialize_critic_networks(self, phi):
        self._qv = QuantileValueHead(
            phi=phi,
            support_dim=self._support_dim).to(self._device)

    def _q_value(self,
                 state: torch.Tensor,
                 action: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        quantiles = super()._q_value(state, action)
        q_value = quantiles.mean(-1)
        return q_value

    def _compute_loss(self, batch: Batch) -> torch.Tensor:
        next_state = self._state_normalizer(batch.next_state, self._device)
        state = self._state_normalizer(batch.primary, self._device)

        with torch.no_grad():
            next_action = self._target_pi(next_state)
            next_quantiles = self._target_qv((next_state, next_action))
        target_quantiles = utils.td_target(
            reward=batch.reward,
            mask=batch.mask,
            target=next_quantiles,
            discount=batch.discount_factor * self._discount).detach()

        expected_quantiles = self._qv((state, batch.secondary))

        loss = utils.quantile_hubber_loss(
            prediction=expected_quantiles,
            target=target_quantiles,
            cumulative_density=self._cumulative_density)
        return loss
"""
