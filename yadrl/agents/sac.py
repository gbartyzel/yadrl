import os
from copy import deepcopy
from typing import NoReturn

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from yadrl.agents.base import BaseOffPolicy
from yadrl.common.heads import GaussianPolicyHead, DoubleQValueHead
from yadrl.common.replay_memory import Batch


class SAC(BaseOffPolicy):
    def __init__(self,
                 actor_phi: nn.Module,
                 critic_phi: nn.Module,
                 lrate: float,
                 **kwargs):

        super(SAC, self).__init__(**kwargs)
        self._actor = GaussianPolicyHead(
            actor_phi, self._action_dim, False, True, (-10.0, 10.0)).to(self._device)
        self._critic = DoubleQValueHead(critic_phi).to(self._device)
        self.load()
        self._target_critic = deepcopy(self._critic).to(self._device)

        self._actor_optim = optim.Adam(self._actor.parameters(), lr=lrate)
        self._critic_optim = optim.Adam(self._critic.parameters(), lr=lrate)

        self._target_entropy = -np.prod(self._action_dim)
        self._log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
        self._alpha_optim = optim.Adam([self._log_alpha], lr=lrate)

    def act(self, state: np.ndarray, train: bool = False):
        state = torch.from_numpy(state).float().to(self._device)
        self._actor.eval()
        with torch.no_grad():
            if train:
                action, _, _ = self._actor.sample(state)
                action = action[0]
            else:
                action = self._actor(state)
        self._actor.train()
        return action.cpu().numpy()

    def update(self):
        batch = self._memory.sample(self._batch_size, self._device)
        self._update_critic(batch)
        self._update_actor_and_alpha(batch)

        self._soft_update(self._critic.parameters(), self._target_critic.parameters())

    def _update_critic(self, batch: Batch):
        alpha = torch.exp(self._log_alpha)
        mask = 1.0 - batch.done

        next_action, log_prob, _ = self._actor.sample(batch.next_state)
        target_next_q = torch.min(*self._target_critic(batch.next_state, next_action))
        target_next_v = target_next_q.view(-1, 1) - alpha * log_prob.view(-1, 1)
        target_q = batch.reward + mask * self._discount * target_next_v.detach()
        expected_q1, expected_q2 = self._critic(batch.state, batch.action)

        q1_loss = self._mse_loss(expected_q1, target_q)
        self._critic_optim.zero_grad()
        q1_loss.backward()
        self._critic_optim.step()

        q2_loss = self._mse_loss(expected_q2, target_q)
        self._critic_optim.zero_grad()
        q2_loss.backward()
        self._critic_optim.step()

    def _update_actor_and_alpha(self, batch: Batch):
        alpha = torch.exp(self._log_alpha)
        action, log_prob, _ = self._actor.sample(batch.state)
        log_prob = log_prob.view(-1, 1)

        target_log_prob = torch.min(*self._critic(batch.state, action)).detach()
        policy_loss = torch.mean(alpha * log_prob - target_log_prob)
        self._actor_optim.zero_grad()
        policy_loss.backward()
        self._actor_optim.step()

        alpha_loss = torch.mean(-self._log_alpha * (log_prob + self._target_entropy).detach())
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

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
