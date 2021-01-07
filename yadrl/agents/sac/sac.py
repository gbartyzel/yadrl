from copy import deepcopy
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import yadrl.common.ops as ops
import yadrl.common.types as t
from yadrl.agents.agent import OffPolicyAgent
from yadrl.common.memory import Batch
from yadrl.networks.head import Head


class SAC(OffPolicyAgent, agent_type="sac"):
    def __init__(
        self,
        pi_learning_rate: float,
        qv_learning_rate: float,
        entropy_learning_rate: float,
        pi_grad_norm_value: float = 0.0,
        qv_grad_norm_value: float = 0.0,
        temperature_tuning: bool = True,
        policy_update_frequency: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._pi_grad_norm_value = pi_grad_norm_value
        self._qv_grad_norm_value = qv_grad_norm_value

        self._pi_optim = optim.Adam(self.pi.parameters(), pi_learning_rate)
        self._qv_optim = optim.Adam(self.qv.parameters(), qv_learning_rate)

        self._policy_update_frequency = policy_update_frequency

        self._temperature_tuning = temperature_tuning
        if temperature_tuning:
            self._target_entropy = -np.prod(self._action_dim)
            self._log_temperature = torch.zeros(
                1, requires_grad=True, device=self._device
            )
            self._entropy_optim = optim.Adam(
                [self._log_temperature], lr=entropy_learning_rate
            )
        self._temperature = 1.0 / self._reward_scaling

    @property
    def pi(self) -> nn.Module:
        return self._networks["actor"]

    @property
    def qv(self) -> nn.Module:
        return self._networks["critic"]

    @property
    def target_qv(self) -> nn.Module:
        return self._networks["target_critic"]

    def _initialize_networks(self, phi: t.TModuleDict) -> t.TModuleDict:
        actor_net = Head.build(
            head_type="squashed_gaussian", phi=phi["actor"], output_dim=self._action_dim
        )
        critic_net = Head.build(
            head_type="multi", phi=phi["critic"], output_dim=1, num_heads=2
        )
        target_critic_net = deepcopy(critic_net)
        actor_net.to(self._device)
        critic_net.to(self._device)
        target_critic_net.to(self._device)
        target_critic_net.eval()

        return {
            "actor": actor_net,
            "critic": critic_net,
            "target_critic": target_critic_net,
        }

    def _act(self, state: np.ndarray, train: bool = False) -> np.ndarray:
        self.pi.eval()
        with torch.no_grad():
            action = self.pi.get_action(super()._act(state), not train)
        self.pi.train()
        return action[0].cpu().numpy()

    def _update(self):
        batch = self._memory.sample(self._batch_size, self._state_normalizer)
        self._update_critic(batch)
        if (
            self._env_step % (self._policy_update_frequency * self._update_frequency)
            == 0
        ):
            self._update_actor_and_entropy(batch)
            self._update_target(self.qv, self.target_qv)

    def _update_critic(self, batch: Batch):
        with torch.no_grad():
            next_action = self.pi.sample(batch.next_state)
            log_prob = self.pi.log_prob(next_action)
            self.target_qv.sample_noise()
            target_next_q = torch.min(*self.target_qv(batch.next_state, next_action))
            target_next_v = target_next_q - self._temperature * log_prob
            target_q = ops.td_target(
                batch.reward,
                batch.mask,
                target_next_v,
                batch.discount_factor * self._discount,
            )
        self.qv.sample_noise()
        expected_qs = self.qv(batch.state, batch.action)

        loss = sum(ops.mse_loss(q, target_q) for q in expected_qs)

        self._qv_optim.zero_grad()
        loss.backward()
        if self._qv_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self.qv.parameters(), self._qv_grad_norm_value)
        self._qv_optim.step()

    def _update_actor_and_entropy(self, batch: Batch):
        action = self.pi.sample(batch.state)
        log_prob = self.pi.log_prob(action)
        self.qv.sample_noise()
        target_log_prob = torch.min(*self.qv(batch.state, action))
        policy_loss = torch.mean(self._temperature * log_prob - target_log_prob)

        self._pi_optim.zero_grad()
        policy_loss.backward()
        if self._pi_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self.pi.parameters(), self._pi_grad_norm_value)
        self._pi_optim.step()

        if self._temperature_tuning:
            entropy_loss = torch.mean(
                -self._log_temperature * (log_prob + self._target_entropy).detach()
            )

            self._entropy_optim.zero_grad()
            entropy_loss.backward()
            self._entropy_optim.step()
            self._temperature = self._log_temperature.exp().detach()

    @property
    def parameters(self) -> t.TNamedParameters:
        return chain(self.qv.named_parameters(), self.pi.named_parameters())

    @property
    def target_parameters(self) -> t.TNamedParameters:
        return self.target_qv.named_parameters()
