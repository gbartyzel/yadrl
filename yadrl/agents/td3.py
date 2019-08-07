from typing import NoReturn
from typing import Sequence
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from yadrl.agents.base import BaseOffPolicy
from yadrl.common.exploration_noise import GaussianNoise
from yadrl.common.memory import Batch
from yadrl.common.utils import mse_loss
from yadrl.networks.models import DeterministicActor, DoubleCritic


class TD3(BaseOffPolicy):
    def __init__(self,
                 pi_phi: nn.Module,
                 qvs_phi: nn.Module,
                 action_limit: Union[np.ndarray, Sequence[float]],
                 target_noise_limit: Union[np.ndarray, Sequence[float]],
                 noise_std: float,
                 target_noise_std: float,
                 policy_update_frequency: int,
                 pi_lrate: float,
                 qvs_lrate: float,
                 pi_grad_norm_value: float,
                 qvs_grad_norm_value: float,
                 **kwargs):
        super(TD3, self).__init__(agent_type='td3', **kwargs)
        GaussianNoise.TORCH_BACKEND = True
        if np.shape(action_limit) != (2,):
            raise ValueError
        self._action_limit = action_limit
        self._target_noise_limit = target_noise_limit
        self._policy_update_frequency = policy_update_frequency
        self._pi_grad_norm_value = pi_grad_norm_value
        self._qvs_grad_norm_value = qvs_grad_norm_value

        self._pi = DeterministicActor(pi_phi, self._action_dim).to(self._device)
        self._target_pi = DeterministicActor(
            pi_phi, self._action_dim).to(self._device)
        self._pi_optim = optim.Adam(self._pi.parameters(), pi_lrate)

        self._qvs = DoubleCritic((qvs_phi, qvs_phi)).to(self._device)
        self._target_qvs = DoubleCritic((qvs_phi, qvs_phi)).to(self._device)
        self._qv_1_optim = optim.Adam(self._qvs.q1_parameters(), qvs_lrate)
        self._qv_2_optim = optim.Adam(self._qvs.q2_parameters(), qvs_lrate)

        self.load()
        self._target_pi.load_state_dict(self._pi.state_dict())
        self._target_qvs.load_state_dict(self._qvs.state_dict())
        self._target_pi.eval()
        self._target_qvs.eval()

        self._noise = GaussianNoise(self._action_dim, sigma=noise_std)
        self._target_noise = GaussianNoise(
            self._action_dim, sigma=target_noise_std)

    def act(self, state, train):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._pi.eval()
        with torch.no_grad():
            action = self._pi(state).cpu()
        self._pi.train()

        if train:
            action = torch.clamp(action + self._noise(), *self._action_limit)
        return action[0].numpy()

    def update(self):
        batch = self._memory.sample(self._batch_size, self._device)
        self._update_critic(batch)

        if self.step % self._policy_update_frequency == 0:
            self._update_actor(batch)
            self._soft_update(self._qvs.parameters(),
                              self._target_qvs.parameters())
            self._soft_update(self._pi.parameters(),
                              self._target_pi.parameters())

    def _update_critic(self, batch: Batch):
        state = self._state_normalizer(batch.state)
        next_state = self._state_normalizer(batch.next_state)

        noise = self._target_noise().clamp(
            *self._target_noise_limit).to(self._device)
        next_action = self._target_pi(next_state) + noise
        next_action = next_action.clamp(*self._action_limit)

        target_next_qs = self._target_qvs(next_state, next_action)
        target_next_q = torch.min(*target_next_qs).view(-1, 1).detach()
        target_q = self._td_target(batch.reward, batch.mask, target_next_q)
        expected_q1, expected_q2 = self._qvs(state, batch.action)

        q1_loss = mse_loss(expected_q1, target_q)
        q2_loss = mse_loss(expected_q2, target_q)

        self._qv_1_optim.zero_grad()
        q1_loss.backward()
        if self._qvs_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._qvs.q1_parameters(),
                                     self._qvs_grad_norm_value)
        self._qv_1_optim.step()

        self._qv_2_optim.zero_grad()
        q2_loss.backward()
        if self._qvs_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._qvs.q2_parameters(),
                                     self._qvs_grad_norm_value)
        self._qv_2_optim.step()

    def _update_actor(self, batch: Batch):
        loss = -self._qvs.eval_v1(
            batch.state, self._pi(batch.state)).mean()
        self._pi_optim.zero_grad()
        loss.backward()
        if self._pi_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._pi.parameters(),
                                     self._pi_grad_norm_value)
        self._pi_optim.step()

    def load(self) -> NoReturn:
        model = self._checkpoint_manager.load()
        if model:
            self._pi.load_state_dict(model['actor'])
            self._qvs.load_state_dict(model['critic'])
            self._reward_normalizer.load(model['reward_norm'])
            self._state_normalizer.load(model['state_norm'])

    def save(self):
        state_dict = dict()
        state_dict['actor'] = self._pi.state_dict(),
        state_dict['critic'] = self._qvs.state_dict()
        if self._use_reward_normalization:
            state_dict['reward_norm'] = self._reward_normalizer.state_dict()
        if self._use_state_normalization:
            state_dict['state_norm'] = self._state_normalizer.state_dict()
        self._checkpoint_manager.save(state_dict, self.step)
