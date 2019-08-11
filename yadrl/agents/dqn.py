import random
from copy import deepcopy
from typing import NoReturn

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from yadrl.agents.base import BaseOffPolicy
from yadrl.common.scheduler import LinearScheduler
from yadrl.common.utils import huber_loss
from yadrl.networks.models import CategoricalDQNModel
from yadrl.networks.models import DQNModel


class DQN(BaseOffPolicy):
    def __init__(self,
                 phi: nn.Module,
                 action_dim: int,
                 lrate: float,
                 grad_norm_value: float,
                 epsilon_annealing_steps: float,
                 epsilon_min: float,
                 v_min: float = -100.0,
                 v_max: float = 100.0,
                 atoms_dim: int = 51,
                 noise_type: str = 'none',
                 use_soft_update: bool = False,
                 use_double_q: bool = False,
                 use_dueling: bool = False,
                 use_categorical: bool = False, **kwargs):
        super(DQN, self).__init__(agent_type='dqn', action_dim=1, **kwargs)

        self._action_dim = action_dim
        self._atoms_dim = atoms_dim

        self._grad_norm_value = grad_norm_value

        self._use_double_q = use_double_q
        self._use_soft_update = use_soft_update
        self._use_noise = noise_type != 'none'
        self._use_catgorical = use_categorical

        if not use_soft_update:
            self._polyak = int(1.0 / self._polyak)

        self._epsilon_scheduler = LinearScheduler(
            1.0, epsilon_min, epsilon_annealing_steps)

        if use_categorical:
            self._qv = CategoricalDQNModel(
                phi=phi,
                output_dim=self._action_dim,
                atoms_dim=atoms_dim,
                dueling=use_dueling,
                noise_type=noise_type).to(self._device)

            self._v_limit = (v_min, v_max)
            self._z_delta = (v_max - v_min) / (self._atoms_dim - 1)
            self._atoms = torch.from_numpy(
                v_min + np.arange(atoms_dim) * self._z_delta
            ).float().unsqueeze(0).to(self._device)
        else:
            self._qv = DQNModel(
                phi=phi,
                output_dim=self._action_dim,
                dueling=use_dueling,
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
            if self._use_catgorical:
                probs = self._qv(state, train)
                action = probs.mul(self._atoms).sum(-1).argmax(-1)
            else:
                action = self._qv(state, train).argmax(dim=1)
        self._qv.train()

        eps_flag = random.random() > self._epsilon_scheduler.step()
        if eps_flag or self._use_noise or not train:
            return action[0].cpu().numpy()
        return random.randint(0, self._action_dim - 1)

    def update(self):
        batch = self._memory.sample(self._batch_size, self._device)

        if self._use_catgorical:
            loss = self._compute_distributional_loss(batch)
        else:
            loss = self._compute_td_loss(batch)

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

    def _compute_td_loss(self, batch):
        state = self._state_normalizer(batch.state, self._device)
        next_state = self._state_normalizer(
            batch.next_state, self._device)

        target_next_q = self._target_qv(next_state, True)
        if self._use_double_q:
            next_action = self._qv(next_state, True).argmax(1).unsqueeze(1)
            target_next_q = target_next_q.gather(1, next_action)
        else:
            target_next_q = target_next_q.max(1)[0].unsqueeze(1)

        target_q = self._td_target(batch.reward, batch.mask,
                                   target_next_q).detach()
        expected_q = self._qv(state, True).gather(1, batch.action.long())
        loss = huber_loss(expected_q, target_q)

        return loss

    def _compute_distributional_loss(self, batch):
        state = self._state_normalizer(batch.state, self._device)
        next_state = self._state_normalizer(batch.next_state, self._device)
        reward = self._reward_normalizer(batch.reward, self._device)

        next_probs = self._target_qv(next_state, True)
        if self._use_double_q:
            probs = self._qv(next_state, True)
            next_action = probs.mul(self._atoms).sum(-1).argmax(-1).view(-1, 1)
        else:
            next_action = next_probs.mul(self._atoms).sum(-1).argmax(1, True)
        next_action = next_action.unsqueeze(-1).repeat(1, 1, self._atoms_dim)

        next_probs = next_probs.gather(1, next_action).squeeze()
        target_probs = torch.zeros(next_probs.shape).to(self._device)

        new_atoms = reward + self._discount * batch.mask * self._atoms
        new_atoms = torch.clamp(new_atoms, *self._v_limit)

        bj = (new_atoms - self._v_limit[0]) / self._z_delta
        l = bj.floor()
        u = bj.ceil()

        delta_l_prob = next_probs * (u + (u == l).float() - bj)
        delta_u_prob = next_probs * (bj - l)

        for i in range(self._batch_size):
            target_probs[i].index_add_(0, l[i].long(), delta_l_prob[i])
            target_probs[i].index_add_(0, u[i].long(), delta_u_prob[i])

        action = batch.action.unsqueeze(-1).repeat(1, 1, self._atoms_dim).long()
        log_probs = self._qv(state, True).log().gather(1, action).squeeze()
        loss = -torch.sum(target_probs * log_probs, 1).mean()

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
