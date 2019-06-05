import os
from typing import NoReturn, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from yadrl.agents.base import BaseOffPolicy
from yadrl.common.exploration_noise import GaussianNoise
from yadrl.common.replay_memory import Batch
from yadrl.networks import DeterministicActor, DoubleCritic


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
        GaussianNoise.TORCH_BACKEND = True
        if np.shape(action_limit) != (2,):
            raise ValueError
        self._action_limit = action_limit
        self._target_noise_limit = target_noise_limit
        self._policy_update_frequency = policy_update_frequency

        self._actor = DeterministicActor(
            actor_phi, self._action_dim).to(self._device)
        self._target_actor = DeterministicActor(
            actor_phi, self._action_dim).to(self._device)
        self._actor_optim = optim.Adam(self._actor.parameters(), actor_lrate)

        self._critic = DoubleCritic(
            (critic_phi, critic_phi)).to(self._device)
        self._target_critic = DoubleCritic(
            (critic_phi, critic_phi)).to(self._device)
        self._critic_optim = optim.Adam(self._critic.parameters(), critic_lrate)

        self.load()
        self._target_actor.load_state_dict(self._actor.state_dict())
        self._target_critic.load_state_dict(self._critic.state_dict())

        self._noise = GaussianNoise(self._action_dim, sigma=noise_std)
        self._target_noise = GaussianNoise(
            self._action_dim, sigma=target_noise_std)

    def act(self, state, train):
        state = torch.from_numpy(state).float().to(self._device)
        self._actor.eval()
        with torch.no_grad():
            action = self._actor(state).cpu()
        self._actor.train()

        if train:
            action = torch.clamp(action + self._noise(), *self._action_limit)
        return action[0].numpy()

    def update(self):
        batch = self._memory.sample(self._batch_size, self._device)
        self._update_critic(batch)

        if self.step % self._policy_update_frequency == 0:
            self._update_actor(batch)
            self._soft_update(self._critic.parameters(),
                              self._target_critic.parameters())
            self._soft_update(self._actor.parameters(),
                              self._target_actor.parameters())

    def _update_critic(self, batch: Batch):
        mask = 1.0 - batch.done

        noise = self._target_noise().clamp(
            *self._target_noise_limit).to(self._device)
        next_action = self._target_actor(batch.next_state)
        next_action = torch.clamp(next_action + noise, *self._action_limit)

        target_next_qs = self._target_critic(batch.next_state, next_action)
        target_next_q = torch.min(target_next_qs).view(-1, 1).detach()
        target_q = (batch.reward + mask * self._discount ** self._n_step
                    * target_next_q)
        expected_q1, expected_q2 = self._critic(batch.state, batch.action)

        q1_loss = self._mse_loss(expected_q1, target_q)
        q2_loss = self._mse_loss(expected_q2, target_q)
        loss = q1_loss + q2_loss

        self._critic_optim.zero_grad()
        loss.backward()
        self._critic_optim.step()

    def _update_actor(self, batch: Batch):
        loss = -torch.mean(
            self._critic.eval_v1(batch.state, self._actor(batch.state)))
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
