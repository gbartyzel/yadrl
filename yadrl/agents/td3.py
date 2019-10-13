import copy
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
                 qv_phi: nn.Module,
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
        self._target_pi = copy.deepcopy(self._pi).to(self._device)
        self._pi_optim = optim.Adam(self._pi.parameters(), pi_lrate)

        self._qv = DoubleCritic(qv_phi).to(self._device)
        self._target_qv = copy.deepcopy(self._qv).to(self._device)
        self._qv_optim = optim.Adam(self._qv.parameters(), qvs_lrate)

        self.load()

        self._noise = GaussianNoise(self._action_dim, sigma=noise_std)
        self._target_noise = GaussianNoise(
            self._action_dim, sigma=target_noise_std)

    def _act(self, state, train):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._pi.eval()
        with torch.no_grad():
            action = self._pi(state)
        self._pi.train()

        if train:
            noise = self._noise().to(self._device)
            action = torch.clamp(action + noise, *self._action_limit)
        return action[0].cpu().numpy()

    def _update(self):
        batch = self._memory.sample(self._batch_size, self._device)
        self._update_critic(batch)

        if self._step % (self._policy_update_frequency *
                         self._update_frequency) == 0:
            self._update_actor(batch)
            self._update_target(self._pi, self._target_pi)
            self._update_target(self._qv, self._target_qv)

    def _update_critic(self, batch: Batch):
        state = self._state_normalizer(batch.state)
        next_state = self._state_normalizer(batch.next_state)

        with torch.no_grad():
            noise = self._target_noise().clamp(
                *self._target_noise_limit).to(self._device)
            next_action = self._target_pi(next_state) + noise
            next_action = next_action.clamp(*self._action_limit)

            target_next_qs = self._target_qv(next_state, next_action)
            target_next_q = torch.min(*target_next_qs).view(-1, 1)
            target_q = self._td_target(batch.reward, batch.mask, target_next_q)
        expected_q1, expected_q2 = self._qv(state, batch.action)

        loss = mse_loss(expected_q1, target_q) + mse_loss(expected_q2, target_q)

        self._qv_optim.zero_grad()
        loss.backward()
        if self._qvs_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._qv.q1_parameters(),
                                     self._qvs_grad_norm_value)
        self._qv_optim.step()

    def _update_actor(self, batch: Batch):
        actions = self._pi(batch.state)
        loss = -self._qv(batch.state, actions)[0].mean()
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
            self._qv.load_state_dict(model['critic'])
            self._step = model['step']
            if 'state_norm' in model:
                self._state_normalizer.load(model['state_norm'])

    def save(self):
        state_dict = dict()
        state_dict['actor'] = self._pi.state_dict(),
        state_dict['critic'] = self._qv.state_dict()
        state_dict['step'] = self._step
        if self._use_state_normalization:
            state_dict['state_norm'] = self._state_normalizer.state_dict()
        self._checkpoint_manager.save(state_dict, self._step)

    @property
    def parameters(self):
        return list(self._qv.named_parameters()) + \
               list(self._pi.named_parameters())

    @property
    def target_parameters(self):
        return list(self._target_qv.named_parameters()) + \
               list(self._target_pi.named_parameters())
