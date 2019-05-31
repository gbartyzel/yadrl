import os
from typing import Union, Sequence, NoReturn

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from yadrl.agents.base import BaseOffPolicy
from yadrl.common.exploration_noise import GaussianNoise
from yadrl.common.heads import DeterministicPolicyHead, ValueHead
from yadrl.common.replay_memory import Batch


class DDPG(BaseOffPolicy):
    def __init__(self,
                 actor_phi: nn.Module,
                 critic_phi: nn.Module,
                 noise: GaussianNoise,
                 actor_lrate: float,
                 critic_lrate: float,
                 l2_reg_value: float,
                 action_bounds: Union[Sequence[float], np.ndarray],
                 **kwargs):
        super(DDPG, self).__init__(**kwargs)
        if np.shape(action_bounds) != (2,):
            raise ValueError
        self._action_bounds = action_bounds

        self._actor = DeterministicPolicyHead(actor_phi, self._action_dim, True).to(self._device)
        self._actor_optim = optim.Adam(self._actor.parameters(), lr=actor_lrate)
        self._target_actor = DeterministicPolicyHead(
            actor_phi, self._action_dim, True).to(self._device)

        self._critic = ValueHead(critic_phi, True, True)
        self._critic_optim = optim.Adam(self._critic.parameters(), lr=critic_lrate,
                                        weight_decay=l2_reg_value)
        self._target_critic = ValueHead(critic_phi, True, True).to(self._device)

        self.load()
        self._target_actor.load_state_dict(self._actor.state_dict())
        self._target_critic.load_state_dict(self._critic.state_dict())

        self._noise = noise

    def act(self, state: np.ndarray, train: bool = False) -> np.ndarray:
        state = torch.from_numpy(state).float().to(self._device)
        self._actor.eval()
        with torch.no_grad():
            action = self._actor(state)
        self._actor.eval()

        if train:
            return torch.clamp(action + self._noise(), *self._action_bounds)
        return action

    def reset(self):
        self._noise.reset()

    def update(self):
        batch = self._memory(self._batch_size, self._device)
        self._update_critic(batch)
        self._update_actor(batch)

        self._soft_update(self._actor.parameters(), self._target_actor.parameters())
        self._soft_update(self._critic.parameters(), self._target_critic.parameters())

    def _update_critic(self, batch: Batch):
        mask = 1.0 - batch.done
        next_action = self._target_actor(batch.next_state)
        target_next_q = self._target_critic(batch.next_state, next_action).view(-1, 1).detach()

        target_q = batch.reward + mask * self._discount * target_next_q
        expected_q = self._critic(batch.state, batch.action)

        loss = self._mse_loss(expected_q, target_q)
        self._critic_optim.zero_grad()
        loss.backward()
        self._critic_optim.step()

    def _update_actor(self, batch: Batch):
        loss = -self._critic(batch.state, self._actor(batch.state))
        self._actor_optim.zero_grad()
        loss.backward()
        self._actor_optim.step()

    def load(self) -> NoReturn:
        if os.path.isfile(self._checkpoint):
            models = torch.load(self._checkpoint)
            self._actor.load_state_dict(models['actor'])
            self._critic.load_state_dict(models['critic'])
            print('Model found and loaded!')
            return
        if not os.path.isdir(os.path.split(self._checkpoint)[0]):
            os.mkdir(os.path.split(self._checkpoint)[0])
        print('Model not found!')

    def save(self):
        state_dicts = {
            'actor': self._actor.state_dict(),
            'critic': self._critic.state_dict()
        }
        torch.save(state_dicts, self._checkpoint)
