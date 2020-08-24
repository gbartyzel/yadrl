import copy
import random
from typing import NoReturn

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import yadrl.common.utils as utils
from yadrl.agents.base import BaseOffPolicyAgent
from yadrl.common.scheduler import BaseScheduler
from yadrl.networks.heads.dqn import DQNHead, DuelingDQNHead


class DQN(BaseOffPolicyAgent):
    def __init__(self,
                 phi: nn.Module,
                 learning_rate: float,
                 grad_norm_value: float,
                 epsilon_scheduler: BaseScheduler = BaseScheduler(),
                 noise_type: str = 'none',
                 use_double_q: bool = False,
                 use_dueling: bool = False, **kwargs):
        super(DQN, self).__init__(**kwargs)
        self._grad_norm_value = grad_norm_value
        self._use_double_q = use_double_q
        self._use_noise = noise_type != 'none'

        self._epsilon_scheduler = epsilon_scheduler

        self._initialize_online_networks(phi, noise_type, use_dueling)
        self._initialize_target_networks()

        self._optim = optim.Adam(self._qv.parameters(), lr=learning_rate,
                                 eps=0.01 / self._batch_size)

    def _initialize_online_networks(self, phi, noise_type, use_dueling):
        head = DuelingDQNHead if use_dueling else DQNHead
        self._qv = head(phi=phi,
                        output_dim=self._action_dim,
                        noise_type=noise_type).to(self._device)

    def _initialize_target_networks(self):
        self._target_qv = copy.deepcopy(self._qv).to(self._device)
        self._target_qv.eval()

    def _act(self, state: int, train: bool = False) -> np.ndarray:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        state = self._state_normalizer(state, self._device)

        self._qv.eval()
        with torch.no_grad():
            q_value = self._sample_q_value(state, train)
        self._qv.train()

        eps_flag = random.random() > self._epsilon_scheduler.step()
        if eps_flag or self._use_noise or not train:
            return q_value.argmax(-1)[0].cpu().numpy()
        return random.randint(0, self._action_dim - 1)

    def _sample_q_value(self, state, train):
        return self._qv(state, train)

    def _update(self):
        batch = self._memory.sample(self._batch_size)
        loss = self._compute_loss(batch)

        self._writer.add_scalar('loss', loss, self._env_step)
        self._optim.zero_grad()
        loss.backward()
        if self._grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._qv.parameters(),
                                     self._grad_norm_value)
        self._optim.step()
        self._update_target(self._qv, self._target_qv)

    def _compute_loss(self, batch):
        state = self._state_normalizer(batch.state, self._device)
        next_state = self._state_normalizer(batch.next_state, self._device)

        with torch.no_grad():
            target_next_q = self._target_qv(next_state, True).detach()
            if self._use_double_q:
                next_action = self._qv(next_state, True).argmax(1, True)
                target_next_q = target_next_q.gather(1, next_action)
            else:
                target_next_q = target_next_q.max(1)[0].view(-1, 1)

        target_q = utils.td_target(
            reward=batch.reward,
            mask=batch.mask,
            target=target_next_q,
            discount=batch.discount_factor * self._discount)
        expected_q = self._qv(state, True).gather(1, batch.action.long())
        loss = utils.huber_loss(expected_q, target_q)
        return loss

    def load(self, path: str) -> NoReturn:
        model = torch.load(path)
        if model:
            self._qv.load_state_dict(model['model'])
            self._target_qv.load_state_dict(model['target_model'])
            self._step = model['step']
            if 'state_norm' in model:
                self._state_normalizer.load(model['state_norm'])

    def save(self):
        state_dict = dict()
        state_dict['model'] = self._qv.state_dict()
        state_dict['target_model'] = self._target_qv.state_dict()
        state_dict['step'] = self._step
        if self._use_state_normalization:
            state_dict['state_norm'] = self._state_normalizer.state_dict()
        torch.save(state_dict, 'model_{}.pth'.format(self._step))

    @property
    def parameters(self):
        return self._qv.named_parameters()

    @property
    def target_parameters(self):
        return self._target_qv.named_parameters()
