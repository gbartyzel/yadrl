from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import yadrl.common.ops as utils
from yadrl.agents.agent import OffPolicyAgent
from yadrl.common.memory import Batch
from yadrl.networks.head import Head


class SAC(OffPolicyAgent, agent_type='sac'):
    def __init__(self,
                 pi_lrate: float,
                 qv_lrate: float,
                 temperature_lrate: float,
                 pi_grad_norm_value: float = 0.0,
                 qv_grad_norm_value: float = 0.0,
                 temperature_tuning: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._pi_grad_norm_value = pi_grad_norm_value
        self._qv_grad_norm_value = qv_grad_norm_value

        self._pi_optim = optim.Adam(self.pi.parameters(), pi_lrate)
        self._qv_optim = optim.Adam(self.qv.parameters(), qv_lrate)

        self._temperature_tuning = temperature_tuning
        if temperature_tuning:
            self._target_entropy = -np.prod(self._action_dim)
            self._log_temperature = torch.zeros(
                1, requires_grad=True, device=self._device)
            self._temperature_optim = optim.Adam([self._log_temperature],
                                                 lr=temperature_lrate)
        self._temperature = 1.0 / self._reward_scaling

    @property
    def pi(self):
        return self._networks['actor']

    @property
    def qv(self):
        return self._networks['critic']

    @property
    def target_qv(self):
        return self._networks['target_critic']

    def _initialize_networks(self, phi):
        actor_net = Head.build(head_type='squashed_gaussian',
                               body=phi['actor'],
                               output_dim=self._action_dim)
        critic_net = Head.build(head_type='multi', body=phi['critic'])
        target_critic_net = deepcopy(critic_net)
        critic_net.to(self._device)
        target_critic_net.to(self._device)
        target_critic_net.eval()

        return {'actor': actor_net,
                'critic': critic_net,
                'target_critic': target_critic_net}

    def _act(self, state: np.ndarray, train: bool = False) -> np.ndarray:
        state = super()._act(state)
        self.pi.eval()
        with torch.no_grad():
            action = self.pi.get_action(state, not train)
        self.pi.train()
        return action[0].cpu().numpy()

    def _update(self):
        batch = self._memory.sample(self._batch_size)
        self._update_critic(batch)
        if self._env_step % (self._policy_update_frequency *
                             self._update_frequency) == 0:
            self._update_actor_and_temperature(batch)
            self._update_target(self.qv, self.target_qv)

    def _update_critic(self, batch: Batch):
        state = self._state_normalizer(batch.state)
        next_state = self._state_normalizer(batch.next_state)

        next_action = self.pi.sample()
        log_prob = self.pi.log_prob(next_action)
        self.target_qv.sample_noise()
        target_next_q = torch.min(*self.target_qv(next_state, next_action))
        target_next_v = target_next_q - self._temperature * log_prob
        target_q = utils.td_target(
            reward=batch.reward,
            mask=batch.mask,
            target=target_next_v,
            discount=batch.discount_factor * self._discount).detach()
        self.qv.sample_noise()
        expected_qs = self.qv(state, batch.action)

        loss = sum(utils.mse_loss(q, target_q) for q in expected_qs)

        self._qv_optim.zero_grad()
        loss.backward()
        if self._qv_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self.qv.parameters(),
                                     self._qv_grad_norm_value)
        self._qv_optim.step()

    def _update_actor_and_temperature(self, batch: Batch):
        state = self._state_normalizer(batch.state)

        action = self.pi.sample()
        log_prob = self.pi.log_prob(action)
        self.qv.sample_noise()
        target_log_prob = torch.min(*self.qv(state, action))
        policy_loss = torch.mean(self._temperature * log_prob - target_log_prob)

        if self._temperature_tuning:
            temperature_loss = torch.mean(
                -self._log_temperature
                * (log_prob + self._target_entropy).detach())
        else:
            temperature_loss = 0.0

        self._pi_optim.zero_grad()
        policy_loss.backward()
        if self._pi_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self.pi.parameters(),
                                     self._pi_grad_norm_value)
        self._pi_optim.step()

        if self._temperature_tuning:
            self._temperature_optim.zero_grad()
            temperature_loss.backward()
            self._temperature_optim.step()
            self._temperature = self._log_temperature.exp().detach()

    @property
    def parameters(self):
        return list(self.qv.named_parameters()) + \
               list(self.pi.named_parameters())

    @property
    def target_parameters(self):
        return self.target_qv.named_parameters()
