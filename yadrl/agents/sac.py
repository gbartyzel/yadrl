from typing import NoReturn
from typing import Optional

import numpy as np
import torch

from yadrl.agents.base import BaseOffPolicy
from yadrl.common.memory import Batch
from yadrl.common.utils import mse_loss
from yadrl.networks import GaussianActor, DoubleCritic


class SAC(BaseOffPolicy):
    def __init__(self,
                 policy_phi: torch.nn.Module,
                 q_values_phi: torch.nn.Module,
                 policy_lrate: float,
                 q_values_lrate: float,
                 alpha_lrate: float,
                 reward_scaling: Optional[float] = 1.0,
                 alpha_tuning: bool = True,
                 **kwargs):

        super(SAC, self).__init__(agent_type='sac', **kwargs)
        self._policy = GaussianActor(
            policy_phi, self._action_dim).to(self._device)
        self._policy_optim = torch.optim.Adam(self._policy.parameters(),
                                              policy_lrate)

        self._q_values = DoubleCritic(
            (q_values_phi, q_values_phi), fan_init=True).to(self._device)
        self._target_q_values = DoubleCritic(
            (q_values_phi, q_values_phi), fan_init=True).to(self._device)
        self._q_value_1_optim = torch.optim.Adam(self._q_values.q1_parameters(),
                                                 q_values_lrate)
        self._q_value_2_optim = torch.optim.Adam(self._q_values.q2_parameters(),
                                                 q_values_lrate)

        self._alpha_tuning = alpha_tuning
        if alpha_tuning:
            self._target_entropy = -np.prod(self._action_dim)
            self._log_alpha = torch.zeros(
                1, requires_grad=True, device=self._device)
            self._alpha_optim = torch.optim.Adam([self._log_alpha],
                                                 lr=alpha_lrate)
        self._alpha = 1.0 / reward_scaling
        self._reward_scaling = reward_scaling

        self.load()
        self._target_q_values.load_state_dict(self._q_values.state_dict())

    def act(self, state: np.ndarray, train: bool = False) -> np.ndarray:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._policy.eval()
        with torch.no_grad():
            if train:
                action = self._policy(state, deterministic=False)[0]
            else:
                action = self._policy(state, deterministic=True)[0]
        self._policy.train()
        return action[0].cpu().numpy()

    def update(self):
        batch = self._memory.sample(self._batch_size, self._device)
        self._update_parameters(*self._compute_loses(batch))
        self._soft_update(self._q_values.parameters(),
                          self._target_q_values.parameters())

    def _compute_loses(self, batch: Batch):
        mask = 1.0 - batch.done
        next_action, log_prob, _ = self._policy(batch.next_state)
        target_next_q = torch.min(*self._target_q_values(batch.next_state,
                                                         next_action))
        target_next_v = target_next_q - self._alpha * log_prob
        target_q = self._td_target(batch.reward, mask, target_next_v).detach()
        expected_q1, expected_q2 = self._q_values(batch.state, batch.action)

        q1_loss = mse_loss(expected_q1, target_q)
        q2_loss = mse_loss(expected_q2, target_q)

        action, log_prob, _ = self._policy(batch.state)
        target_log_prob = torch.min(*self._q_values(batch.state, action))
        policy_loss = torch.mean(self._alpha * log_prob - target_log_prob)

        if self._alpha_tuning:
            alpha_loss = torch.mean(
                -self._log_alpha * (log_prob + self._target_entropy).detach())
        else:
            alpha_loss = 0.0

        return q1_loss, q2_loss, policy_loss, alpha_loss

    def _update_parameters(self, q1_loss, q2_loss, policy_loss, alpha_loss):

        self._q_value_1_optim.zero_grad()
        q1_loss.backward()
        self._q_value_1_optim.step()

        self._q_value_2_optim.zero_grad()
        q2_loss.backward()
        self._q_value_2_optim.step()

        self._policy_optim.zero_grad()
        policy_loss.backward()
        self._policy_optim.step()

        if self._alpha_tuning:
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()

            self._alpha = torch.exp(self._log_alpha)

    def load(self) -> NoReturn:
        model = self._checkpoint_manager.load()
        if model:
            self._policy.load_state_dict(model['actor'])
            self._q_values.load_state_dict(model['critic'])

    def save(self):
        state_dicts = {
            'actor': self._policy.state_dict(),
            'critic': self._q_values.state_dict()
        }
        self._checkpoint_manager.save(state_dicts, self.step)
