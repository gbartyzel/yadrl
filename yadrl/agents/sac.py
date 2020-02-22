from typing import NoReturn

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from yadrl.agents.base import BaseOffPolicy
from yadrl.common.memory import Batch
from yadrl.common.utils import mse_loss
from yadrl.networks.models import DoubleCritic
from yadrl.networks.models import GaussianActor


class SAC(BaseOffPolicy):
    def __init__(self,
                 pi_phi: nn.Module,
                 qv_phi: nn.Module,
                 pi_lrate: float,
                 qv_lrate: float,
                 alpha_lrate: float,
                 pi_grad_norm_value: float = 0.0,
                 qv_grad_norm_value: float = 0.0,
                 alpha_tuning: bool = True,
                 **kwargs):

        super(SAC, self).__init__(**kwargs)
        self._pi_grad_norm_value = pi_grad_norm_value
        self._qvs_grad_norm_value = qv_grad_norm_value

        self._pi = GaussianActor(
            pi_phi, self._action_dim).to(self._device)
        self._pi_optim = optim.Adam(self._pi.parameters(), pi_lrate)

        self._qv = DoubleCritic(qv_phi).to(self._device)
        self._target_qv = DoubleCritic(qv_phi).to(self._device)
        self._qv_1_optim = optim.Adam(self._qv.q1_parameters(), qv_lrate)
        self._qv_2_optim = optim.Adam(self._qv.q2_parameters(), qv_lrate)
        self._target_qv.load_state_dict(self._qv.state_dict())

        self._alpha_tuning = alpha_tuning
        if alpha_tuning:
            self._target_entropy = -np.prod(self._action_dim)
            self._log_alpha = torch.zeros(
                1, requires_grad=True, device=self._device)
            self._alpha_optim = optim.Adam([self._log_alpha], lr=alpha_lrate)
        self._alpha = 1.0 / self._reward_scaling

    def _act(self, state: np.ndarray, train: bool = False) -> np.ndarray:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._pi.eval()
        with torch.no_grad():
            if train:
                action = self._pi(state, deterministic=False)[0]
            else:
                action = self._pi(state, deterministic=True)[0]
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
        target_next_q = torch.min(*self._target_qv(next_state, next_action))
        target_next_v = target_next_q - self._alpha * log_prob
        target_q = self._td_target(batch.reward, batch.mask,
                                   target_next_v).detach()
        expected_q1, expected_q2 = self._qv(state, batch.action)

        q1_loss = mse_loss(expected_q1, target_q)
        q2_loss = mse_loss(expected_q2, target_q)

        action, log_prob, _ = self._pi(state)
        target_log_prob = torch.min(*self._qv(state, action))
        policy_loss = torch.mean(self._alpha * log_prob - target_log_prob)

        if self._alpha_tuning:
            alpha_loss = torch.mean(
                -self._log_alpha * (log_prob + self._target_entropy).detach())
        else:
            alpha_loss = 0.0

        return q1_loss, q2_loss, policy_loss, alpha_loss

    def _update_parameters(self, q1_loss, q2_loss, policy_loss, alpha_loss):
        self._qv_1_optim.zero_grad()
        q1_loss.backward()
        if self._qvs_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._qv.q1_parameters(),
                                     self._qvs_grad_norm_value)
        self._qv_1_optim.step()

        self._qv_2_optim.zero_grad()
        q2_loss.backward()
        if self._qvs_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._qv.q2_parameters(),
                                     self._qvs_grad_norm_value)
        self._qv_2_optim.step()

        self._pi_optim.zero_grad()
        policy_loss.backward()
        if self._pi_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._pi.parameters(),
                                     self._pi_grad_norm_value)
        self._pi_optim.step()

        if self._alpha_tuning:
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()

            self._alpha = torch.exp(self._log_alpha)

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
