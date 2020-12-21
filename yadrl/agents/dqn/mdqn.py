import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import yadrl.common.ops as ops
from yadrl.agents.dqn.dqn import DQN


class MDQN(DQN, agent_type='munchausen_dqn'):
    def __init__(self,
                 alpha: float = 0.9,
                 lower_clamp: float = -1.0,
                 temperature_tuning: bool = True,
                 temperature_learning_rate: float = 0.0005,
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
                                                 lr=temperature_learning_rate)
        self._temperature = 1.0 / self._reward_scaling

    def _compute_loss(self, batch) -> torch.Tensor:
        state = self._state_normalizer(batch.state, self._device)
        next_state = self._state_normalizer(batch.next_state, self._device)

        if self._temperature_tuning:
            self._update_temperature(state)

        with torch.no_grad():
            self.target_model.sample_noise()
            target_q_next = self.target_model(next_state)
            next_pi = F.softmax(target_q_next / self._temperature, -1)
            next_log_pi = ops.scaled_logsoftmax(target_q_next,
                                                self._temperature)
            m_log_pi = ops.scaled_logsoftmax(self.target_model(state),
                                             self._temperature)
            m_log_pi = m_log_pi.gather(1, batch.action.long())

            m_reward = batch.reward + self._alpha * m_log_pi.clamp(
                self._lower_clamp, 0.0)
            target_q = ops.td_target(
                reward=m_reward,
                mask=batch.mask,
                target=(next_pi * (target_q_next - next_log_pi)).sum(-1, True),
                discount=batch.discount_factor * self._discount)

        expected_q = self.model(state).gather(1, batch.action.long())
        if self._use_huber_loss_fn:
            return ops.huber_loss(expected_q, target_q)
        return ops.mse_loss(expected_q, target_q)

    def _update_temperature(self, state: torch.Tensor):
        q_value = self._sample_q(state, True)
        log_pi = ops.scaled_logsoftmax(q_value, self._temperature)
        loss = torch.mean(-self._log_temperature
                          * (log_pi + self._target_entropy).detach())

        self._temperature_optim.zero_grad()
        loss.backward()
        self._temperature_optim.step()
        self._temperature = self._log_temperature.exp().detach()

        self._writer.add_scalar('loss/entropy', loss, self._env_step)
