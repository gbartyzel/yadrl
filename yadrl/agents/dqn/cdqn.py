from typing import Tuple

import torch
import torch.nn.functional as F

import yadrl.common.ops as ops
from yadrl.agents.dqn.dqn import DQN


class CategoricalDQN(DQN, agent_type='categorical_dqn'):
    head_types = ['categorical', 'distribution_dueling']

    def __init__(self,
                 v_limit: Tuple[float, float] = (-100.0, 100.0),
                 support_dim: int = 51, **kwargs):
        self._support_dim = support_dim
        super().__init__(**kwargs)
        self._v_limit = v_limit
        self._atoms = torch.linspace(v_limit[0], v_limit[1], support_dim,
                                     device=self._device).unsqueeze(0)

    def _sample_q(self, state: torch.Tensor,
                  train: bool = False) -> torch.Tensor:
        probs = F.softmax(super()._sample_q(state, train), -1)
        return probs.mul(self._atoms.expand_as(probs)).sum(-1)

    def _compute_loss(self, batch):
        state = self._state_normalizer(batch.state, self._device)
        next_state = self._state_normalizer(batch.next_state, self._device)
        batch_vec = torch.arange(self._batch_size).long()

        with torch.no_grad():
            self.target_model.sample_noise()
            next_probs = F.softmax(self.target_model(next_state).exp(), -1)
            exp_atoms = self._atoms.expand_as(next_probs)
            if self._use_double_q:
                self.model.sample_noise()
                next_double_probs = F.softmax(self.model(next_state), -1)
                next_q = next_double_probs.mul(exp_atoms).sum(-1)
            else:
                next_q = next_probs.mul(exp_atoms).sum(-1)
            next_action = next_q.argmax(-1).long()
            next_probs = next_probs[batch_vec, next_action, :]

        target_atoms = ops.td_target(batch.reward, batch.mask, self._atoms,
                                     batch.discount_factor * self._discount)
        target_probs = ops.l2_projection(next_probs, self._atoms, target_atoms)

        self.model.sample_noise()
        log_probs = F.log_softmax(self.model(state), -1)
        log_probs = log_probs[batch_vec, batch.action.squeeze().long(), :]
        loss = torch.mean(-(target_probs * log_probs).sum(-1))
        return loss
