import os
from copy import deepcopy
from typing import NoReturn, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from yadrl.agents.base import BaseOffPolicy
from yadrl.common.heads import DoubleQValueHead, DeterministicPolicyHead
from yadrl.common.exploration_noise import GaussianNoise
from yadrl.common.replay_memory import Batch


class TD3(BaseOffPolicy):
    def __init__(self,
                 actor_phi: nn.Module,
                 critic_phi: nn.Module,
                 action_limit: Union[np.ndarray, Sequence[float]],
                 target_noise_limit: Union[np.ndarray, Sequence[float]],
                 noise_std: float,
                 target_noise_std: float,
                 policy_update_frequency: int,
                 actor_lrate: float,
                 critic_lrate: float,
                 **kwargs):
        super(TD3, self).__init__(**kwargs)
        if np.shape(action_limit) != (2,):
            raise ValueError
        self._action_limit = action_limit
        self._target_noise_limit = target_noise_limit
        self._policy_update_frequency = policy_update_frequency

        self._actor = DeterministicPolicyHead(actor_phi, self._action_dim).to(self._device)
        self._target_actor = deepcopy(self._actor).to(self._device)
        self._actor_optim = optim.Adam(self._actor.parameters(), actor_lrate)

        self._critic = DoubleQValueHead(critic_phi).to(self._device)
        self._target_critic = deepcopy(self._critic).to(self._device)
        self._critic_optim = optim.Adam(self._critic.parameters(), critic_lrate)

        self._noise = GaussianNoise(self._action_dim, sigma=noise_std)
        self._target_noise = GaussianNoise(self._action_dim, sigma=target_noise_std)

    def act(self, state, train):
        state = torch.from_numpy(state).float().to(self._device)
        self._actor.eval()
        with torch.no_grad():
            action = self._actor(state)
        self._actor.train()

        if train:
            return torch.clamp(action + self._noise(), *self._action_limit)
        return action

    def update(self):
        batch = self._memory.sample(self._batch_size, self._device)
        self._update_critic(batch)

        if self.step % self._policy_update_frequency == 0:
            self._update_actor(batch)
            self._soft_update(self._critic.parameters(), self._target_critic.parameters())
            self._soft_update(self._actor.parameters(), self._target_actor.parameters())

    def _update_critic(self, batch: Batch):
        noise = torch.clamp(self._target_noise(), *self._target_noise_limit)
        next_action = self._target_actor(batch.next_state)
        next_action = torch.clamp(next_action + noise, *self._action_limit)

        target_next_q1, target_next_q2 = self._target_critic(batch.next_state, next_action)
        target_next_q = torch.min(target_next_q1, target_next_q2).view(-1, 1).detach()
        target_q = batch.reward + (1.0 - batch.done) * self._discount * target_next_q
        expected_q1, expected_q2 = self._critic(batch.state, batch.action)

        loss = self._mse_loss(expected_q1, target_q) + self._mse_loss(expected_q2, target_q)
        self._critic_optim.zero_grad()
        loss.backward()
        self._critic_optim.step()

    def _update_actor(self, batch: Batch):
        loss = -self._critic(batch.state, self._actor(batch.state))
        self._actor_optim.zero_grad()
        loss.backward()
        self._actor_optim.step()

    def _load(self) -> NoReturn:
        if os.path.isfile(self._checkpoint):
            models = torch.load(self._checkpoint)
            self._actor.load_state_dict(models['actor'])
            self._critic.load_state_dict(models['critic'])
            print('Model found and loaded!')
            return
        if not os.path.isdir(os.path.split(self._checkpoint)[0]):
            os.mkdir(os.path.split(self._checkpoint)[0])
        print('Model not found!')

    def _save(self):
        state_dicts = {
            'actor': self._actor.state_dict(),
            'critic': self._critic.state_dict()
        }
        torch.save(state_dicts, self._checkpoint)
