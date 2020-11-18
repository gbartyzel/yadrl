from typing import Tuple

import torch

import yadrl.common.utils as utils
from yadrl.agents.dpg.ddpg import DDPG
from yadrl.common.memory import Batch


class CategoricalDDPG(DDPG):
    def __init__(self,
                 support_dim: int = 51,
                 v_limit: Tuple[float, float] = (-10.0, 10.0),
                 **kwargs):
        self._support_dim = support_dim
        super().__init__(**kwargs)
        self._v_limit = v_limit
        self._atoms = torch.linspace(v_limit[0], v_limit[1], support_dim,
                                     device=self._device).unsqueeze(0)

    def _sample_q(self,
                  state: torch.Tensor,
                  action: torch.Tensor,
                  sample_noise: bool = False) -> torch.Tensor:
        probs = super()._sample_q(state, action, sample_noise).exp()
        return probs.mul(self._atoms.expand_as(probs)).sum(-1)

    def _compute_loss(self, batch: Batch) -> torch.Tensor:
        next_state = self._state_normalizer(batch.next_state, self._device)
        state = self._state_normalizer(batch.state, self._device)

        with torch.no_grad():
            next_action = self.target_pi(next_state)
            self.target_qv.sample_noise()
            next_probs = self.target_qv(next_state, next_action).exp()
        target_atoms = utils.td_target(
            reward=batch.reward,
            mask=batch.mask,
            target=self._atoms,
            discount=batch.discount_factor * self._discount)
        target_probs = utils.l2_projection(
            next_probs=next_probs,
            atoms=self._atoms,
            target_atoms=target_atoms)

        self.qv.sample_noise()
        log_probs = self.qv(state, batch.action)
        loss = -(target_probs * log_probs).sum(-1)
        return loss.mean()
