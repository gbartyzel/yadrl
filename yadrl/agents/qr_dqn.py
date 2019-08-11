import random
from copy import deepcopy
from typing import NoReturn

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from yadrl.agents.base import BaseOffPolicy
from yadrl.common.scheduler import LinearScheduler
from yadrl.common.utils import quantile_hubber_loss
from yadrl.networks.models import QuantileDQNModel


class QRDQN(BaseOffPolicy):
    def __init__(self,
                 phi: nn.Module,
                 action_dim: int,
                 lrate: float,
                 grad_norm_value: float,
                 epsilon_annealing_steps: float,
                 epsilon_min: float,
                 quantiles_dim: int = 100,
                 noise_type: str = 'none',
                 use_double_q: bool = False,
                 use_soft_update: bool = False, **kwargs):
        super(QRDQN, self).__init__(agent_type='qrdqn', action_dim=1, **kwargs)
        self._action_dim = action_dim
        self._quantiles_dim = quantiles_dim
        self._cumulative_probs = torch.from_numpy(
            (np.arange(quantiles_dim) + 0.5) / quantiles_dim
        ).float().unsqueeze(0).to(self._device)

        self._grad_norm_value = grad_norm_value

        self._use_soft_update = use_soft_update
        self._use_noise = noise_type != 'none'
        self._use_double_q = use_double_q

        if not use_soft_update:
            self._polyak = int(1.0 / self._polyak)

        self._epsilon_scheduler = LinearScheduler(
            1.0, epsilon_min, epsilon_annealing_steps)

        self._qv = QuantileDQNModel(
            phi=phi,
            output_dim=self._action_dim,
            quantiles_dim=quantiles_dim,
            noise_type=noise_type).to(self._device)
        self._target_qv = deepcopy(self._qv)
        self.load()
        self._target_qv.load_state_dict(self._qv.state_dict())

        self._optim = optim.Adam(self._qv.parameters(), lr=lrate,
                                 eps=0.01 / self._batch_size)

    def act(self, state: int, train: bool = False) -> np.ndarray:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        state = self._state_normalizer(state, self._device)

        self._qv.eval()
        with torch.no_grad():
            action = self._qv(state, train).mean(-1).argmax(-1)
        self._qv.train()
        print(action)
        eps_flag = random.random() > self._epsilon_scheduler.step()
        if eps_flag or self._use_noise or not train:
            return action[0].cpu().numpy()
        return random.randint(0, self._action_dim - 1)

    def update(self):
        batch = self._memory.sample(self._batch_size, self._device)

        loss = self._compute_quantile_loss(batch)

        self._optim.zero_grad()
        loss.backward()
        if self._grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._qv.parameters(),
                                     self._grad_norm_value)
        self._optim.step()

        if self._use_soft_update:
            self._soft_update(self._qv.parameters(),
                              self._target_qv.parameters())

        if not self._use_soft_update and self.step % self._polyak == 0:
            self._hard_update(self._qv, self._target_qv)

    def _compute_quantile_loss(self, batch):
        state = self._state_normalizer(batch.state, self._device)
        next_state = self._state_normalizer(batch.next_state, self._device)
        reward = self._reward_normalizer(batch.reward, self._device)

        next_quantiles = self._target_qv(next_state, True)
        next_action = next_quantiles.mean(-1).argmax(-1).long().view(-1, 1, 1)
        next_action = next_action.repeat(1, 1, self._quantiles_dim)

        next_quantiles = next_quantiles.gather(1, next_action).squeeze()
        target_quantiles = reward + batch.mask * self._discount * next_quantiles

        action = batch.action.view(-1, 1, 1).repeat(1, 1, self._quantiles_dim)
        expected_quantiles = self._qv(state, True).gather(1, action.long())
        expected_quantiles = expected_quantiles.squeeze(1)

        loss = quantile_hubber_loss(expected_quantiles, target_quantiles,
                                    self._cumulative_probs, 1.0)

        return loss

    def load(self) -> NoReturn:
        model = self._checkpoint_manager.load()
        if model:
            self._reward_normalizer.load(model['reward_norm'])
            self._state_normalizer.load(model['state_norm'])
            self._qv.load_state_dict(model['model'])

    def save(self):
        state_dict = dict()
        state_dict['model'] = self._qv.state_dict()
        if self._use_reward_normalization:
            state_dict['reward_norm'] = self._reward_normalizer.state_dict()
        if self._use_state_normalization:
            state_dict['state_norm'] = self._state_normalizer.state_dict()
        self._checkpoint_manager.save(state_dict, self.step)
