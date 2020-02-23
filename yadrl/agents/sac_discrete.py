from typing import NoReturn

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from yadrl.agents.base import BaseOffPolicy
from yadrl.common.memory import Batch
from yadrl.common.utils import mse_loss
from yadrl.networks.heads import GumbelSoftmaxPolicyHead
from yadrl.networks.heads import MultiValueHead


class SACDiscrete(BaseOffPolicy):
    def __init__(self,
                 pi_phi: nn.Module,
                 qv_phi: nn.Module,
                 pi_lrate: float,
                 qv_lrate: float,
                 temperature_lrate: float,
                 pi_grad_norm_value: float = 0.0,
                 qvs_grad_norm_value: float = 0.0,
                 temperature_tuning: bool = True,
                 **kwargs):

        super(SACDiscrete, self).__init__(**kwargs)
        self._pi_grad_norm_value = pi_grad_norm_value
        self._qv_grad_norm_value = qvs_grad_norm_value

        self._pi = GumbelSoftmaxPolicyHead(
            pi_phi, self._action_dim).to(self._device)
        self._pi_optim = optim.Adam(self._pi.parameters(), pi_lrate)

        self._qv = MultiValueHead(qv_phi, self._action_dim).to(self._device)
        self._target_qv = MultiValueHead(
            qv_phi, self._action_dim).to(self._device)
        self._target_qv.load_state_dict(self._qv.state_dict())
        self._target_qv.eval()
        self._qv_optims = [optim.Adam(self._qv.parameters(item=i), qv_lrate)
                           for i in range(2)]

        self._temperature_tuning = temperature_tuning
        if temperature_tuning:
            self._target_entropy = -np.prod(self._action_dim)
            self._log_temperature = torch.zeros(
                1, requires_grad=True, device=self._device)
            self._temperature_optim = optim.Adam([self._log_temperature],
                                                 lr=temperature_lrate)
        self._temperature = 1.0 / self._reward_scaling

    def _act(self, state: np.ndarray, train: bool = False) -> np.ndarray:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._pi.eval()
        with torch.no_grad():
            action = self._pi.sample(state)[0].argmax()
        self._pi.train()
        return action.cpu().numpy()

    def _update(self):
        batch = self._memory.sample(self._batch_size)
        self._update_parameters(*self._compute_loses(batch))
        self._update_target(self._qv, self._target_qv)

    def _compute_loses(self, batch: Batch):
        state = self._state_normalizer(batch.state)
        next_state = self._state_normalizer(batch.next_state)

        next_action, log_prob = self._pi.sample(next_state)
        target_next_qs = self._target_qv(next_state)
        target_next_q = torch.min(
            (target_next_qs[0] * next_action).sum(-1, True),
            (target_next_qs[1] * next_action).sum(-1, True))
        target_next_v = target_next_q - self._temperature * log_prob
        target_q = self._td_target(batch.reward, batch.mask,
                                   target_next_v).detach()

        expected_q1, expected_q2 = self._qv(state)

        q1_loss = mse_loss(expected_q1.gather(1, batch.action.long()), target_q)
        q2_loss = mse_loss(expected_q2.gather(1, batch.action.long()), target_q)

        action, log_prob, _ = self._pi.sample(state)
        qs = self._qv(state)
        target_log_prob = torch.min((qs[0] * action).sum(-1, True),
                                    (qs[1] * action).sum(-1, True))
        policy_loss = torch.mean(self._temperature * log_prob - target_log_prob)

        if self._temperature_tuning:
            alpha_loss = torch.mean(
                -self._log_temperature * (
                        log_prob + self._target_entropy).detach())
        else:
            alpha_loss = 0.0

        return q1_loss, q2_loss, policy_loss, alpha_loss

    def _update_parameters(self, q1_loss, q2_loss, policy_loss, alpha_loss):
        self._qv_optims[0].zero_grad()
        q1_loss.backward()
        if self._qv_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._qv.parameters(item=0),
                                     self._qv_grad_norm_value)
        self._qv_optims[0].step()

        self._qv_optims[1].zero_grad()
        q2_loss.backward()
        if self._qv_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._qv.parameters(item=1),
                                     self._qv_grad_norm_value)
        self._qv_optims[1].step()

        self._pi_optim.zero_grad()
        policy_loss.backward()
        if self._pi_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._pi.parameters(),
                                     self._pi_grad_norm_value)
        self._pi_optim.step()

        if self._temperature_tuning:
            self._temperature_optim.zero_grad()
            alpha_loss.backward()
            self._temperature.step()

            self._temperature = self._log_temperature.exp().detach()

        self._data_to_log['alpha'] = self._temperature

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
