from typing import Tuple

import torch as th
import torch.nn.functional as F

import yadrl.common.ops as ops
from yadrl.agents.dpg.ddpg import DDPG
from yadrl.common.memory import Batch


class CategoricalDDPG(DDPG, agent_type="categorical_ddpg"):
    def __init__(
        self,
        support_dim: int = 51,
        v_limit: Tuple[float, float] = (-10.0, 10.0),
        **kwargs
    ):
        self._support_dim = support_dim
        super().__init__(**kwargs)
        self._v_limit = v_limit
        self._atoms = th.linspace(
            v_limit[0], v_limit[1], support_dim, device=self._device
        ).unsqueeze(0)

    def _sample_q(
        self, state: th.Tensor, action: th.Tensor, sample_noise: bool = False
    ) -> th.Tensor:
        probs = F.softmax(super()._sample_q(state, action, sample_noise), -1)
        return probs.mul(self._atoms.expand_as(probs)).sum(-1)

    def _compute_critic_loss(self, batch: Batch) -> th.Tensor:
        with th.no_grad():
            next_action = self.target_pi(batch.next_state)
            self.target_qv.sample_noise()
            next_probs = F.softmax(self.target_qv(batch.next_state, next_action), -1)
            target_atoms = ops.td_target(
                batch.reward, batch.mask, self._atoms, self._discount
            )
            target_probs = ops.l2_projection(next_probs, self._atoms, target_atoms)

        self.qv.sample_noise()
        log_probs = F.log_softmax(self.qv(batch.state, batch.action), -1)
        loss = -(target_probs * log_probs).sum(-1)
        return loss.mean()
