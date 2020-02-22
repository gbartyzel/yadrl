import random
from copy import deepcopy
from typing import NoReturn

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import yadrl.common.utils as utils
from yadrl.agents.base import BaseOffPolicy
from yadrl.common.scheduler import BaseScheduler
from yadrl.networks.models import MultiDQN


class MaxminDQN(BaseOffPolicy):
    def __init__(self,
                 phi: nn.Module,
                 heads_num: int,
                 learning_rate: float,
                 grad_norm_value: float,
                 epsilon_scheduler: BaseScheduler = BaseScheduler(),
                 noise_type: str = 'none',
                 use_dueling: bool = False, **kwargs):
        super(MaxminDQN, self).__init__(**kwargs)

        self._grad_norm_value = grad_norm_value
        self._use_noise = noise_type != 'none'
        self._heads_num = heads_num

        self._epsilon_scheduler = epsilon_scheduler

        self._qv = MultiDQN(
            phi=phi,
            output_dim=self._action_dim,
            heads_num=heads_num,
            dueling=use_dueling,
            noise_type=noise_type).to(self._device)
        self._target_qv = deepcopy(self._qv)
        self._optims = [
            optim.RMSprop(self._qv.parameters(item=i), lr=learning_rate)
            for i in range(heads_num)
        ]

    def _act(self, state: int, train: bool = False) -> np.ndarray:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        state = self._state_normalizer(state, self._device)

        self._qv.eval()
        with torch.no_grad():
            q_value = torch.cat(self._qv(state, train, True), 1).min(1)[0]
        self._qv.train()

        eps_flag = random.random() > self._epsilon_scheduler.step()
        if eps_flag or self._use_noise or not train:
            return q_value.argmax(-1)[0].cpu().numpy()
        return random.randint(0, self._action_dim - 1)

    def _update(self):
        batch = self._memory.sample(self._batch_size)

        losses = self._compute_td_loss(batch)

        for i in range(self._heads_num):
            self._writer.add_scalar('loss/{}'.format(i),
                                    losses[i], self._env_step)

        for optim, loss in zip(self._optims, losses):
            optim.zero_grad()
            loss.backward()
            if self._grad_norm_value > 0.0:
                nn.utils.clip_grad_norm_(self._qv.parameters(),
                                         self._grad_norm_value)
            optim.step()

        self._update_target(self._qv, self._target_qv)

    def _compute_td_loss(self, batch):
        state = self._state_normalizer(batch.state, self._device)
        next_state = self._state_normalizer(batch.next_state, self._device)

        target_next_qs = self._target_qv(next_state, True, True)
        target_next_q = torch.min(torch.cat(target_next_qs, 1), 1)[0]
        target_next_q = target_next_q.max(1)[0].view(-1, 1)

        target_q = utils.td_target(reward=batch.reward,
                                   mask=batch.mask,
                                   next_value=target_next_q,
                                   discount=self._discount).detach()

        loss = [utils.huber_loss(q.gather(1, batch.action.long()), target_q)
                for q in self._qv(state, True)]
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
