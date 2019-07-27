import random
from typing import NoReturn

import numpy as np
import torch

from yadrl.agents.base import BaseOffPolicy
from yadrl.common.scheduler import LinearScheduler
from yadrl.common.utils import huber_loss
from yadrl.networks import DQNModel


class DQN(BaseOffPolicy):
    def __init__(self,
                 phi: torch.nn.Module,
                 action_dim: int,
                 lrate: float,
                 epsilon_annealing_steps: float,
                 epsilon_min: float,
                 use_soft_update: bool = False,
                 use_double_q: bool = False,
                 use_dueling: bool = False,
                 use_noisy_layer: bool = False, **kwargs):
        super(DQN, self).__init__(agent_type='dqn', action_dim=1, **kwargs)
        self._action_dim = action_dim

        self._use_double_q = use_double_q
        self._use_soft_update = use_soft_update
        self._use_noisy_layer = use_noisy_layer

        if not use_soft_update:
            self._polyak = int(1.0 / self._polyak)

        self._epsilon_scheduler = LinearScheduler(
            1.0, epsilon_min, epsilon_annealing_steps)

        self._qv = DQNModel(phi, self._action_dim, use_dueling,
                            use_noisy_layer).to(self._device)
        self._target_qv = DQNModel(phi, self._action_dim, use_dueling,
                                   use_noisy_layer).to(self._device)

        self.load()
        self._target_qv.load_state_dict(self._qv.state_dict())

        self._optim = torch.optim.Adam(self._qv.parameters(), lr=lrate)

    def act(self, state: int, train: bool = False) -> np.ndarray:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._qv.eval()
        with torch.no_grad():
            if train:
                self._qv.sample_noise()
            else:
                self._qv.reset_noise()
            action = torch.argmax(self._qv(state))
        self._qv.train()

        eps_flag = random.random() > self._epsilon_scheduler.step()
        if eps_flag or self._use_noisy_layer or not train:
            return action.cpu().numpy()
        return random.randint(0, self._action_dim - 1)

    def update(self):
        batch = self._memory.sample(self._batch_size, self._device)
        mask = 1.0 - batch.done

        self._target_qv.sample_noise()
        target_next_q = self._target_qv(batch.next_state)
        if self._use_double_q:
            self._qv.sample_noise()
            next_action = self._qv(batch.next_state).argmax(1).unsqueeze(1)
            target_next_q = target_next_q.gather(1, next_action)
        else:
            target_next_q = target_next_q.max(1)[0].unsqueeze(1)

        target_q = self._td_target(batch.reward, mask, target_next_q).detach()
        self._qv.sample_noise()
        expected_q = self._qv(batch.state).gather(1, batch.action.long())
        loss = huber_loss(expected_q, target_q)

        self._optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._qv.parameters(), 10.0)
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
