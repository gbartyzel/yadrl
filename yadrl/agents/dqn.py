import random

import numpy as np
import torch

from copy import deepcopy
from yadrl.common.heads import DQNHead, DuelingDQNHead
from yadrl.common.replay_memory import ReplayMemory


class DQN(object):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 phi: torch.nn.Module,
                 lrate: float,
                 discount_factor: float,
                 polyak_factor: float,
                 memory_capacity: int,
                 batch_size: int,
                 warm_up_steps: int,
                 epsilon_decay_factor: float,
                 epsilon_min: float,
                 use_soft_update: bool = True,
                 use_double_q: bool = False,
                 use_dueling: bool = False):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._discount = discount_factor
        self._polyak = polyak_factor if use_soft_update else 1.0 / polyak_factor

        self._use_double_q = use_double_q
        self._use_soft_update = use_soft_update

        self._output_dim = action_dim

        self._step = 0.0
        self._batch_size = batch_size
        self._warm_up_steps = warm_up_steps

        self._eps = 1.0
        self._esp_decay_factor = epsilon_decay_factor
        self._eps_min = epsilon_min

        self._model = DuelingDQNHead(phi, action_dim) if use_dueling else DQNHead(phi, action_dim)
        self._model.to(self._device)

        self._target_model = deepcopy(self._model).to(self._device)
        self._target_model.eval()

        self._optim = torch.optim.Adam(self._model.parameters(), lr=lrate)

        self._memory = ReplayMemory(memory_capacity, state_dim, action_dim, True)

    def act(self, state: np.ndarray, train: bool = False):
        self._eps = max(self._eps * self._esp_decay_factor, self._eps_min)
        state = torch.from_numpy(state).float().to(self._device)

        self._model.eval()
        with torch.no_grad():
            action = torch.argmax(self._model(state))

        if random.random() > self._eps or not train:
            return action.cpu().numpy()
        return random.randint(0, self._output_dim)

    def observe(self, state, action, reward, next_state, done):
        self._memory.push(state, action, reward, next_state, done)
        if self._memory.size > self._warm_up_steps:
            self._step += 1
            self.update()

    def update(self):
        batch = self._memory.sample(self._batch_size, self._device)
        state_batch = batch['obs_1']
        action_batch = batch['u']
        reward_batch = batch['r']
        next_state_batch = batch['obs_2']
        done_batch = batch['d']

        self._model.train()

        if self._use_double_q:
            next_action = self._model(next_state_batch).argmax(1).view(-1, 1)
            target_next_q = self._target_model(next_state_batch).gather(1, next_action).detach()
        else:
            target_next_q = self._target_model(next_state_batch).max(1)[0].view(-1, 1).detach()

        target_q = reward_batch + (1.0 - done_batch) * self._discount * target_next_q
        expected_q = self._model(state_batch).gather(1, action_batch)
        loss = 0.5 * (expected_q - target_q) ** 2

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        if self._use_soft_update:
            self._soft_update(self._model.parameters(), self._target_model.parameters())

        if not self._use_soft_update and self._step % self._polyak == 0:
            self._hard_update(self._model, self._target_model)

    def _soft_update(self, params: torch.nn.parameter, target_params: torch.nn.parameter):
        for param, t_param in zip(params, target_params):
            t_param.data.copy_(t_param.data * (1.0 - self._polyak) + param.data * self._polyak)

    @staticmethod
    def _hard_update(model: torch.nn.Module, target_model: torch.nn.Module):
        target_model.load_state_dict(model.state_dict())
