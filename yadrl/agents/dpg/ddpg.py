"""from copy import deepcopy
from typing import Any, NoReturn, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import yadrl.common.utils as utils
from yadrl.agents.agent import OffPolicyAgent
from yadrl.common.exploration_noise import GaussianNoise
from yadrl.common.memory import Batch
from yadrl.networks.heads.policy import DeterministicPolicyHead
from yadrl.networks.heads.value import ValueHead


class DDPG(OffPolicyAgent):
    def __init__(self,
                 pi_phi: nn.Module,
                 qv_phi: nn.Module,
                 pi_lrate: float,
                 qv_lrate: float,
                 action_limit: Union[Sequence[float], np.ndarray],
                 noise: GaussianNoise,
                 noise_scale_factor: float = 1.0,
                 l2_reg_value: float = 0.01,
                 pi_grad_norm_value: float = 0.0,
                 qv_grad_norm_value: float = 0.0,
                 policy_update_frequency: int = 2,
                 **kwargs):
        super().__init__(**kwargs)
        assert np.shape(action_limit) == (2,), "Wrong action limit!"

        self._action_limit = action_limit
        self._policy_update_frequency = policy_update_frequency
        self._pi_grad_norm_value = pi_grad_norm_value
        self._qv_grad_norm_value = qv_grad_norm_value

        self._initialize_online_networks(pi_phi, qv_phi)
        self._initialize_target_networks()

        self._pi_optim = optim.Adam(self._pi.parameters(), lr=pi_lrate)
        self._qv_optim = optim.Adam(
            self._qv.parameters(), qv_lrate, weight_decay=l2_reg_value)

        self._noise = noise
        self._noise_scale_factor = noise_scale_factor

    def _initialize_online_networks(self, pi_phi, qv_phi):
        self._initialize_actor_networks(pi_phi)
        self._initialize_critic_networks(qv_phi)

    def _initialize_target_networks(self):
        self._target_pi = deepcopy(self._pi).to(self._device)
        self._target_pi.eval()
        self._target_qv = deepcopy(self._qv).to(self._device)
        self._target_qv.eval()

    def _initialize_actor_networks(self, phi: nn.Module):
        self._pi = DeterministicPolicyHead(
            phi=deepcopy(phi),
            output_dim=self._action_dim,
            fan_init=True).to(self._device)

    def _initialize_critic_networks(self, phi):
        self._qv = ValueHead(phi=deepcopy(phi)).to(self._device)
        self._target_qv = ValueHead(phi=deepcopy(phi)).to(self._device)

    def _act(self, state: np.ndarray, train: bool = False) -> np.ndarray:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        state = self._state_normalizer(state, self._device)
        self._pi.eval()
        with torch.no_grad():
            action = self._pi(state)
            qv = self._q_value(state, action)
        self._pi.train()
        self._writer.add_scalar('train/q_value', qv, self._env_step)
        if train:
            noise = self._noise_scale_factor * self._noise().to(self._device)
            action = torch.clamp(action + noise, *self._action_limit)
        return action[0].cpu().numpy()

    def _q_value(self,
                 state: torch.Tensor,
                 action: torch.Tensor) -> torch.Tensor:
        return self._qv((state, action))

    def _observe(self,
                 state: Union[np.ndarray, torch.Tensor],
                 action: Union[np.ndarray, torch.Tensor],
                 reward: Union[float, torch.Tensor],
                 next_state: Union[np.ndarray, torch.Tensor],
                 done: Any):
        super()._observe(state, action, reward, next_state, done)
        if done:
            self._noise.reset()

    def _update(self):
        batch = self._memory.sample(self._batch_size)
        self._update_critic(batch)
        if self._env_step % (self._policy_update_frequency *
                             self._update_frequency) == 0:
            self._update_actor(batch)
            self._update_target(self._pi, self._target_pi)
            self._update_target(self._qv, self._target_qv)

    def _update_critic(self, batch: Batch):
        loss = self._compute_loss(batch)
        self._qv_optim.zero_grad()
        loss.backward()
        if self._qv_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._qv.parameters(),
                                     self._qv_grad_norm_value)
        self._qv_optim.step()
        self._writer.add_scalar('train/loss/qv', loss.item(), self._env_step)

    def _compute_loss(self, batch: Batch) -> torch.Tensor:
        next_state = self._state_normalizer(batch.next_state, self._device)
        state = self._state_normalizer(batch.primary, self._device)

        with torch.no_grad():
            next_action = self._target_pi(next_state).detach()
            target_next_q = self._target_qv((next_state, next_action))

        target_q = utils.td_target(
            reward=batch.reward,
            mask=batch.mask,
            target=target_next_q.view(-1, 1),
            discount=batch.discount_factor * self._discount)
        expected_q = self._q_value(state, batch.secondary)
        return utils.mse_loss(expected_q, target_q)

    def _update_actor(self, batch: Batch):
        state = self._state_normalizer(batch.primary, self._device)
        q_value = self._q_value(state, self._pi(state))
        loss = -q_value.mean()
        self._pi_optim.zero_grad()

        loss.backward()
        if self._pi_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._pi.parameters(),
                                     self._pi_grad_norm_value)
        self._pi_optim.step()
        self._writer.add_scalar('train/loss/pi', loss.item(), self._env_step)

    def load(self, path: str) -> NoReturn:
        model = torch.load(path)
        if model:
            self._pi.load_state_dict(model['actor'])
            self._qv.load_state_dict(model['critic'])
            self._target_qv.load_state_dict(model['target_critic'])
            self._env_step = model['step']
            if 'state_norm' in model:
                self._state_normalizer.load(model['state_norm'])

    def save(self):
        state_dict = dict()
        state_dict['actor'] = self._pi.state_dict(),
        state_dict['critic'] = self._qv.state_dict()
        state_dict['target_critic'] = self._target_qv.state_dict()
        state_dict['step'] = self._env_step
        if self._use_state_normalization:
            state_dict['state_norm'] = self._state_normalizer.state_dict()
        torch.save(state_dict, 'model_{}.pth'.format(self._env_step))

    @property
    def parameters(self):
        return list(self._qv.named_parameters()) + \
               list(self._pi.named_parameters())

    @property
    def target_parameters(self):
        return list(self._target_qv.named_parameters()) + \
               list(self._target_pi.named_parameters())
"""
