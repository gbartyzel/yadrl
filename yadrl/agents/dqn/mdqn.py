import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim

import yadrl.common.ops as ops
from yadrl.agents.dqn.dqn import DQN
from yadrl.common.memory import Batch


class MDQN(DQN, agent_type="munchausen_dqn"):
    def __init__(
        self,
        alpha: float = 0.9,
        lower_clamp: float = -1.0,
        temperature_tuning: bool = True,
        entropy_learning_rate: float = 0.0005,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._alpha = alpha
        self._lower_clamp = lower_clamp

        self._temperature_tuning = temperature_tuning
        if temperature_tuning:
            self._target_entropy = -0.98 * np.log(1 / np.prod(self._action_dim))
            self._log_temperature = th.zeros(1, requires_grad=True, device=self._device)
            self._temperature_optim = optim.Adam(
                [self._log_temperature], lr=entropy_learning_rate
            )
        self._temperature = 1.0 / self._reward_scaling

    def _compute_loss(self, batch: Batch) -> th.Tensor:
        with th.no_grad():
            self.target_model.sample_noise()
            target_q_next = self.target_model(batch.next_state)
            next_pi = F.softmax(target_q_next / self._temperature, -1)
            next_log_pi = ops.scaled_logsoftmax(target_q_next, self._temperature)
            m_log_pi = ops.scaled_logsoftmax(
                self.target_model(batch.state), self._temperature
            )
            m_log_pi = m_log_pi.gather(1, batch.action.long())

            m_reward = batch.reward + self._alpha * m_log_pi.clamp(
                self._lower_clamp, 0.0
            )
            m_target = (next_pi * (target_q_next - next_log_pi)).sum(-1, True)
            target_q = ops.td_target(
                m_reward, batch.mask, m_target, batch.discount_factor * self._discount
            )

        expected_q = self.model(batch.state).gather(1, batch.action.long())
        if self._use_huber_loss_fn:
            loss = ops.huber_loss(expected_q, target_q)
        else:
            loss = ops.mse_loss(expected_q, target_q)
        if self._temperature_tuning:
            self._update_temperature(batch.state)

        return loss

    def _update_temperature(self, state: th.Tensor):
        q_value = self._sample_q(state, True)
        log_pi = ops.scaled_logsoftmax(q_value, self._temperature)
        loss = th.mean(
            -self._log_temperature * (log_pi + self._target_entropy).detach()
        )

        self._temperature_optim.zero_grad()
        loss.backward()
        self._temperature_optim.step()
        self._temperature = self._log_temperature.exp().detach()

        self._writer.add_scalar("loss/entropy", loss, self._env_step)
