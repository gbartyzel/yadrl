from typing import Tuple

import torch

import yadrl.common.utils as utils
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

    def _sample_q_value(self, state, train):
        probs = super()._sample_q_value(state).exp()
        return probs.mul(self._atoms.expand_as(probs)).sum(-1)

    def _compute_loss(self, batch):
        state = self._state_normalizer(batch.state, self._device)
        next_state = self._state_normalizer(batch.next_state, self._device)

        batch_vec = torch.arange(self._batch_size).long()

        with torch.no_grad():
            next_probs = self.target_model(next_state, True)
            exp_atoms = self._atoms.expand_as(next_probs)
            if self._use_double_q:
                next_q = self.model(next_state, True).mul(exp_atoms).sum(-1)
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
        action = batch.secondary.squeeze().long()
        log_probs = self.model(state, True, True)[batch_vec, action, :]
        loss = torch.mean(-(target_probs * log_probs).sum(-1))
        return loss
