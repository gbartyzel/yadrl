import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import yadrl.common.utils as utils
from yadrl.agents.dqn.dqn import DQN


class MDQN(DQN):
    def __init__(self,
                 alpha: float,
                 lower_clamp: float,
                 temperature_tuning: bool = True,
                 temperature_lrate: float = 0.0005,
                 **kwargs):
        super().__init__(**kwargs)
        self._alpha = alpha
        self._lower_clamp = lower_clamp

        self._temperature_tuning = temperature_tuning
        if temperature_tuning:
            self._target_entropy = -np.prod(self._action_dim)
            self._log_temperature = torch.zeros(
                1, requires_grad=True, device=self._device)
            self._temperature_optim = optim.Adam([self._log_temperature],
                                                 lr=temperature_lrate)
        self._temperature = 1.0 / self._reward_scaling

    def _compute_loss(self, batch):
        state = self._state_normalizer(batch.state, self._device)
        next_state = self._state_normalizer(batch.next_state, self._device)

        with torch.no_grad():
            target_q_next = self._target_qv(next_state, True)
            next_pi = F.softmax(target_q_next / self._temperature, -1)
            next_log_pi = self._log_sum_exp_trick(target_q_next)
            m_log_pi = self._log_sum_exp_trick(
                self._target_qv(state, True)).gather(1, batch.action.long())

            target_q = utils.td_target(
                reward=batch.reward + self._alpha
                       * m_log_pi.clamp(self._lower_clamp, 0.0),
                mask=batch.mask,
                target=(next_pi * (target_q_next - next_log_pi)).sum(-1, True),
                discount=batch.discount_factor * self._discount)

        expected_q = self._qv(state, True).gather(1, batch.action.long())
        loss = utils.huber_loss(expected_q, target_q)

        if self._temperature_tuning:
            self._update_temperature(expected_q)
        return loss

    def _log_sum_exp_trick(self, q_value):
        diff = q_value - q_value.max(-1, True)[0]
        return diff - self._temperature * torch.log(
            torch.exp(diff / self._temperature).sum(-1, True))

    def _update_temperature(self, q_value):
        log_pi = self._log_sum_exp_trick(q_value)
        temperature_loss = torch.mean(
            -self._log_temperature
            * (log_pi + self._target_entropy).detach())

        self._temperature_optim.zero_grad()
        temperature_loss.backward()
        self._temperature_optim.step()
        self._temperature = self._log_temperature.exp().detach()
