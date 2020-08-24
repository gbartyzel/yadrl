import numpy as np
import torch

import yadrl.common.utils as utils
from yadrl.agents.sac.sac import SAC
from yadrl.common.memory import Batch
from yadrl.networks.dqn_heads import DoubleDQNHead
from yadrl.networks.policy_heads import GumbelSoftmaxPolicyHead


class SACDiscrete(SAC):
    def __init__(self,
                 noise_type: str = 'none', **kwargs):
        self._noise_type = noise_type
        super().__init__(**kwargs)

    def _initialize_online_networks(self, pi_phi, qv_phi):
        self._pi = GumbelSoftmaxPolicyHead(pi_phi, self._action_dim)
        self._qv = DoubleDQNHead(phi=qv_phi,
                                 output_dim=self._action_dim,
                                 noise_type=self._noise_type)

    def _act(self, state: np.ndarray, train: bool = False) -> np.ndarray:
        return super()._act(state, train).argmax()

    def _compute_loses(self, batch: Batch):
        state = self._state_normalizer(batch.state)
        next_state = self._state_normalizer(batch.next_state)

        next_action, log_prob = self._pi(next_state)
        target_next_qs = self._target_qv(next_state, True, True)
        target_next_q = torch.min(torch.cat(target_next_qs, 1), 1)[0]
        target_next_q = torch.sum(target_next_q * next_action, -1, True)

        target_next_v = target_next_q - self._temperature * log_prob
        target_q = utils.td_target(
            reward=batch.reward,
            mask=batch.mask,
            target=target_next_v,
            discount=batch.discount_factor * self._discount).detach()

        qs_loss = (utils.mse_loss(q.gather(1, batch.action.long()), target_q)
                   for q in self._qv(state, True))

        action, log_prob = self._pi(state)
        pi_q = torch.min(torch.cat(self._qv(state, True, True), 1), 1)[0]
        target_log_prob = torch.sum(pi_q * action, -1, True)
        policy_loss = torch.mean(self._temperature * log_prob - target_log_prob)

        if self._temperature_tuning:
            temperature_loss = (
                    -self._log_temperature
                    * (log_prob + self._target_entropy).detach()).mean()
        else:
            temperature_loss = 0.0

        return qs_loss, policy_loss, temperature_loss
