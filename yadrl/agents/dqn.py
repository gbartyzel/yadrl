import random
from typing import NoReturn

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from yadrl.agents.base import BaseOffPolicy
from yadrl.common.scheduler import LinearScheduler
from yadrl.networks import DQNModel


class DQN(BaseOffPolicy):
    def __init__(self,
                 phi: nn.Module,
                 action_dim: int,
                 lrate: float,
                 epsilon_annealing_steps: float,
                 epsilon_min: float,
                 use_soft_update: bool = False,
                 use_double_q: bool = False,
                 use_dueling: bool = False, **kwargs):
        super(DQN, self).__init__(agent_type='dqn', action_dim=1, **kwargs)
        self._action_dim = action_dim

        self._use_double_q = use_double_q
        self._use_soft_update = use_soft_update

        if not use_soft_update:
            self._polyak = int(1.0 / self._polyak)

        self._epsilon_scheduler = LinearScheduler(
            1.0, epsilon_min, epsilon_annealing_steps)

        self._qv = DQNModel(
            phi, self._action_dim, use_dueling).to(self._device)
        self._target_qv = DQNModel(
            phi, self._action_dim, use_dueling).to(self._device)

        self.load()
        self._target_qv.load_state_dict(self._qv.state_dict())

        self._optim = optim.Adam(self._qv.parameters(), lr=lrate,
                                 eps=0.01 / self._batch_size)

        self._writer = SummaryWriter()

    def act(self, state: int, train: bool = False) -> np.ndarray:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._qv.eval()
        with torch.no_grad():
            action = torch.argmax(self._qv(state))
        self._qv.train()

        if random.random() > self._epsilon_scheduler.step() or not train:
            return action.cpu().numpy()
        return random.randint(0, self._action_dim - 1)

    def update(self):
        batch = self._memory.sample(self._batch_size, self._device)
        mask = 1.0 - batch.done

        target_next_q = self._target_qv(batch.next_state)
        if self._use_double_q:
            next_action = self._qv(batch.next_state).argmax(dim=1, keepdim=True)
            target_next_q = target_next_q.gather(1, next_action)
        else:
            target_next_q = target_next_q.max(dim=1, keepdim=True)[0]

        target_q = self._td_target(batch.reward, mask, target_next_q).detach()
        expected_q = self._qv(batch.state).gather(1, batch.action.long())
        loss = self._mse_loss(expected_q, target_q)

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        if self._use_soft_update:
            self._soft_update(self._qv.parameters(),
                              self._target_qv.parameters())

        if not self._use_soft_update and self.step % self._polyak == 0:
            self._hard_update(self._qv, self._target_qv)

    def load(self) -> NoReturn:
        model = self._checkpoint_manager.load()
        if model:
            self._qv.load_state_dict(model)

    def save(self):
        self._checkpoint_manager.save(self._qv.state_dict(), self.step)
