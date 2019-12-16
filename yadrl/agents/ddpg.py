import copy
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import yadrl.common.exploration_noise as noise
from yadrl.agents.base import BaseOffPolicy
from yadrl.common.memory import Batch
from yadrl.common.utils import mse_loss
from yadrl.networks.models import Critic
from yadrl.networks.models import DeterministicActor


class DDPG(BaseOffPolicy):
    def __init__(self,
                 pi_phi: nn.Module,
                 qv_phi: nn.Module,
                 pi_lrate: float,
                 qv_lrate: float,
                 l2_reg_value: float,
                 pi_grad_norm_value: float,
                 qv_grad_norm_value: float,
                 action_limit: Union[Sequence[float], np.ndarray],
                 noise_type: Optional[str] = "ou",
                 mean: Optional[float] = 0.0,
                 sigma: Optional[float] = 0.2,
                 sigma_min: Optional[float] = 0.0,
                 n_step_annealing: Optional[float] = 1e6,
                 theta: Optional[float] = 0.15,
                 dt: Optional[float] = 0.01,
                 **kwargs):
        super(DDPG, self).__init__(agent_type='ddpg', **kwargs)
        if np.shape(action_limit) != (2,):
            raise ValueError
        self._action_limit = action_limit
        self._pi_grad_norm_value = pi_grad_norm_value
        self._qv_grad_norm_value = qv_grad_norm_value

        self._pi = DeterministicActor(
            phi=pi_phi,
            output_dim=self._action_dim,
            fan_init=True).to(self._device)
        self._pi_optim = optim.Adam(self._pi.parameters(), lr=pi_lrate)
        self._target_pi = copy.deepcopy(self._pi).to(self._device)

        self._qv = Critic(phi=qv_phi).to(self._device)
        self._qv_optim = optim.Adam(
            self._qv.parameters(), qv_lrate, weight_decay=l2_reg_value)
        self._target_qv = copy.deepcopy(self._qv).to(self._device)

        self.load()
        self._target_pi.load_state_dict(self._pi.state_dict())
        self._target_qv.load_state_dict(self._qv.state_dict())

        self._noise = self._get_noise(noise_type, mean, sigma, sigma_min,
                                      theta, n_step_annealing, dt)

    def _act(self, state: np.ndarray, train: bool = False) -> np.ndarray:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        state = self._state_normalizer(state, self._device)
        self._pi.eval()
        with torch.no_grad():
            action = self._pi(state)
        self._pi.train()

        if train:
            noise = self._noise().to(self._device)
            action = torch.clamp(action + noise, *self._action_limit)
        return action[0].cpu().numpy()

    def reset(self):
        self._noise.reset()

    def _update(self):
        batch = self._memory.sample(self._batch_size, self._device)
        self._update_critic(batch)
        self._update_actor(batch)

        self._update_target(self._pi, self._target_pi)
        self._update_target(self._qv, self._target_qv)

    def _update_critic(self, batch: Batch):
        state = self._state_normalizer(batch.state, self._device)
        next_state = self._state_normalizer(batch.next_state, self._device)

        with torch.no_grad():
            next_action = self._target_pi(batch.next_state)
            target_next_q = self._target_qv(next_state, next_action).view(-1, 1)

            target_q = self._td_target(batch.reward, batch.mask, target_next_q)
        expected_q = self._qv(state, batch.action)

        loss = mse_loss(expected_q, target_q)
        self._qv_optim.zero_grad()
        loss.backward()
        if self._qv_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._qv.parameters(),
                                     self._qv_grad_norm_value)
        self._qv_optim.step()

    def _update_actor(self, batch: Batch):
        state = self._state_normalizer(batch.state, self._device)
        loss = torch.mean(-self._qv(state, self._pi(state)))
        self._pi_optim.zero_grad()

        loss.backward()
        if self._pi_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._pi.parameters(),
                                     self._pi_grad_norm_value)
        self._pi_optim.step()

    def load(self, path: str) -> NoReturn:
        model = torch.load(path)
        if model:
            self._pi.load_state_dict(model['actor'])
            self._qv.load_state_dict(model['critic'])
            self._target_qv.load_state_dict(model['target_critic'])
            self._step = model['step']
            if 'state_norm' in model:
                self._state_normalizer.load(model['state_norm'])

    def save(self):
        state_dict = dict()
        state_dict['actor'] = self._pi.state_dict(),
        state_dict['critic'] = self._qv.state_dict()
        state_dict['target_critic'] = self._target_qv.state_dict()
        state_dict['step'] = self._step
        if self._use_state_normalization:
            state_dict['state_norm'] = self._state_normalizer.state_dict()
        torch.save(state_dict, 'model_{}.pth'.format(self._step))

    def _get_noise(self, noise_type: str,
                   mean: float,
                   sigma: float,
                   sigma_min: float,
                   theta: float,
                   n_step_annealing: float,
                   dt: float) -> noise.GaussianNoise:

        if noise_type == "gaussian":
            return noise.GaussianNoise(
                self._action_dim, mean=mean, sigma=sigma, sigma_min=sigma_min,
                n_step_annealing=n_step_annealing)
        elif noise_type == "ou":
            return noise.OUNoise(
                self._action_dim, mean=mean, theta=theta, sigma=sigma,
                sigma_min=sigma_min, n_step_annealing=n_step_annealing, dt=dt)
        else:
            raise ValueError

    @property
    def parameters(self):
        return list(self._qv.named_parameters()) + \
               list(self._pi.named_parameters())

    @property
    def target_parameters(self):
        return list(self._target_qv.named_parameters()) + \
               list(self._target_pi.named_parameters())
