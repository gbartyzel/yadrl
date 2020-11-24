from typing import Tuple

import torch

import yadrl.common.ops as utils
from yadrl.agents.dqn.dqn import DQN


class CategoricalDQN(DQN, agent_type='categorical_dqn'):
    head_types = ['categorical', 'dueling_categorical']

    def __init__(self,
                 v_limit: Tuple[float, float] = (-100.0, 100.0),
                 support_dim: int = 51, **kwargs):
        super().__init__(**kwargs)
        self._v_limit = v_limit
        self._atoms = torch.linspace(v_limit[0], v_limit[1], support_dim,
                                     device=self._device).unsqueeze(0)

    def _sample_q(self, state: torch.Tensor,
                  train: bool = False) -> torch.Tensor:
        probs = super()._sample_q(state, train).exp()
        return probs.mul(self._atoms.expand_as(probs)).sum(-1)

    def _compute_loss(self, batch):
        state = self._state_normalizer(batch.state, self._device)
        next_state = self._state_normalizer(batch.next_state, self._device)
        batch_vec = torch.arange(self._batch_size).long()

        with torch.no_grad():
            self.target_model.sample_noise()
            next_probs = self.target_model(next_state).exp()
            exp_atoms = self._atoms.expand_as(next_probs)
            if self._use_double_q:
                self.model.sample_noise()
                next_q = self.model(next_state).exp().mul(exp_atoms).sum(-1)
            else:
                next_q = next_probs.mul(exp_atoms).sum(-1)
            next_action = next_q.argmax(-1).long()
            next_probs = next_probs[batch_vec, next_action, :]

        target_atoms = utils.td_target(
            reward=batch.reward,
            mask=batch.mask,
            target=self._atoms,
            discount=batch.discount_factor * self._discount)
        target_probs = utils.l2_projection(
            next_probs=next_probs,
            atoms=self._atoms,
            target_atoms=target_atoms)

        self.model.sample_noise()
        log_probs = self.model(state)
        log_probs = log_probs[batch_vec, batch.action.squeeze().long(), :]
        loss = torch.mean(-(target_probs * log_probs).sum(-1))
        return loss
