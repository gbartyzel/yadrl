import copy
from typing import NoReturn

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import yadrl.common.utils as utils
from yadrl.agents.base import BaseOffPolicyAgent
from yadrl.common.memory import Batch
from yadrl.networks.policy_heads import GaussianPolicyHead
from yadrl.networks.value_heads import DoubleValueHead


class SAC(BaseOffPolicyAgent):
    def __init__(self,
                 pi_phi: nn.Module,
                 qv_phi: nn.Module,
                 pi_lrate: float,
                 qv_lrate: float,
                 temperature_lrate: float,
                 pi_grad_norm_value: float = 0.0,
                 qv_grad_norm_value: float = 0.0,
                 temperature_tuning: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._pi_grad_norm_value = pi_grad_norm_value
        self._qvs_grad_norm_value = qv_grad_norm_value

        self._initialize_online_networks(pi_phi, qv_phi)
        self._initialize_target_networks()

        self._pi_optim = optim.Adam(self._pi.parameters(), pi_lrate)
        self._qv_optims = [optim.Adam(self._qv.head_1_parameters(), qv_lrate),
                           optim.Adam(self._qv.head_2_parameters(), qv_lrate)]

        self._temperature_tuning = temperature_tuning
        if temperature_tuning:
            self._target_entropy = -np.prod(self._action_dim)
            self._log_temperature = torch.zeros(
                1, requires_grad=True, device=self._device)
            self._temperature_optim = optim.Adam([self._log_temperature],
                                                 lr=temperature_lrate)
        self._temperature = 1.0 / self._reward_scaling

    def _initialize_online_networks(self, pi_phi, qv_phi):
        self._pi = GaussianPolicyHead(phi=pi_phi,
                                      output_dim=self._action_dim,
                                      independent_std=False,
                                      squash=True).to(self._device)
        self._qv = DoubleValueHead(phi=qv_phi).to(self._device)

    def _initialize_target_networks(self):
        self._target_qv = copy.deepcopy(self._qv).to(self._device)
        self._target_qv.eval()

    def _act(self, state: np.ndarray, train: bool = False) -> np.ndarray:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._pi.eval()
        with torch.no_grad():
            action = self._pi(state, deterministic=not train)[0]
        self._pi.train()
        return action[0].cpu().numpy()

    def _update(self):
        batch = self._memory.sample(self._batch_size)
        self._update_parameters(*self._compute_loses(batch))
        self._update_target(self._qv, self._target_qv)

    def _compute_loses(self, batch: Batch):
        state = self._state_normalizer(batch.state)
        next_state = self._state_normalizer(batch.next_state)

        next_action, log_prob, _ = self._pi(next_state)
        target_next_q = torch.min(
            *self._target_qv((next_state, next_action), train=True))
        target_next_v = target_next_q - self._temperature * log_prob
        target_q = utils.td_target(
            reward=batch.reward,
            mask=batch.mask,
            target=target_next_v,
            discount=batch.discount_factor * self._discount).detach()
        expected_qs = self._qv((state, batch.action), train=True)

        qs_loss = (utils.mse_loss(q, target_q) for q in expected_qs)

        action, log_prob, _ = self._pi(state)
        target_log_prob = torch.min(*self._qv((state, action), train=True))
        policy_loss = torch.mean(self._temperature * log_prob - target_log_prob)

        if self._temperature_tuning:
            temperature_loss = torch.mean(
                -self._log_temperature
                * (log_prob + self._target_entropy).detach())
        else:
            temperature_loss = 0.0

        return qs_loss, policy_loss, temperature_loss

    def _update_parameters(self, qs_loss, policy_loss, alpha_loss):
        for i, (loss, optim) in enumerate(zip(qs_loss, self._qv_optims)):
            optim.zero_grad()
            loss.backward()
            if self._qvs_grad_norm_value > 0.0:
                nn.utils.clip_grad_norm_(self._qv.parameters(item=i),
                                         self._qvs_grad_norm_value)
            optim.step()

        self._pi_optim.zero_grad()
        policy_loss.backward()
        if self._pi_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._pi.parameters(),
                                     self._pi_grad_norm_value)
        self._pi_optim.step()

        if self._temperature_tuning:
            self._temperature_optim.zero_grad()
            alpha_loss.backward()
            self._temperature_optim.step()
            self._temperature = self._log_temperature.exp().detach()

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

    @property
    def parameters(self):
        return list(self._qv.named_parameters()) + \
               list(self._pi.named_parameters())

    @property
    def target_parameters(self):
        return self._target_qv.named_parameters()
