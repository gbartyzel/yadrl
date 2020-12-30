import copy
import random
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

import yadrl.common.ops as ops
from yadrl.agents.agent import OffPolicyAgent
from yadrl.common.scheduler import BaseScheduler
from yadrl.networks.head import Head


class DQN(OffPolicyAgent, agent_type='dqn'):
    head_types = ['simple', 'dueling']

    def __init__(self,
                 learning_rate: float,
                 grad_norm_value: float = 0.0,
                 exploration_strategy: BaseScheduler = BaseScheduler(),
                 noise_type: str = 'none',
                 adam_eps: float = 0.0003125,
                 use_double_q: bool = False,
                 use_dueling: bool = False,
                 use_huber_loss_fn: bool = True,
                 **kwargs):
        self._noise_type = noise_type
        self._use_dueling = use_dueling
        super().__init__(**kwargs)
        self._grad_norm_value = grad_norm_value

        self._use_double_q = use_double_q
        self._use_huber_loss_fn = use_huber_loss_fn

        self._epsilon_scheduler = exploration_strategy
        self._optim = optim.Adam(self.model.parameters(), lr=learning_rate,
                                 eps=adam_eps)

    @property
    def model(self) -> nn.Module:
        return self._networks['model']

    @property
    def target_model(self) -> nn.Module:
        return self._networks['target_model']

    @property
    def parameters(self):
        return self.model.named_parameters()

    @property
    def target_parameters(self):
        return self.target_model.named_parameters()

    def _initialize_networks(self, phi: nn.Module) -> Dict[str, nn.Module]:
        head_type = self.head_types[int(self._use_dueling)]
        support_dim = self._support_dim if hasattr(self, '_support_dim') else 1
        network = Head.build(head_type=head_type,
                             phi=phi,
                             hidden_dim=(32,),
                             support_dim=support_dim,
                             output_dim=self._action_dim,
                             noise_type=self._noise_type)
        target_network = copy.deepcopy(network)
        network.to(self._device)
        target_network.to(self._device)
        target_network.eval()
        return {'model': network, 'target_model': target_network}

    def _act(self, state: int, train: bool = False) -> int:
        state = super()._act(state)

        self.model.eval()
        with torch.no_grad():
            q_value = self._sample_q(state, train)
        self.model.train()

        eps_flag = random.random() > self._epsilon_scheduler.step()
        if eps_flag or (self._noise_type != 'none') or not train:
            return q_value.argmax(-1)[0].cpu().numpy()
        return random.randint(0, self._action_dim - 1)

    def _sample_q(self, state: torch.Tensor,
                  train: bool = False) -> torch.Tensor:
        self.model.reset_noise()
        if train:
            self.model.sample_noise()
        return self.model(state)

    def _update(self):
        batch = self._memory.sample(self._batch_size, self._state_normalizer)
        loss = self._compute_loss(batch)

        self._writer.add_scalar('loss/q_value', loss, self._env_step)
        self._optim.zero_grad()
        loss.backward()
        if self._grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self._grad_norm_value)
        self._optim.step()
        self._update_target(self.model, self.target_model)

    def _compute_loss(self, batch):
        with torch.no_grad():
            self.target_model.sample_noise()
            target_next_q = self.target_model(batch.next_state)
            if self._use_double_q:
                self.model.sample_noise()
                next_action = self.model(batch.next_state).argmax(1, True)
                target_next_q = target_next_q.gather(1, next_action)
            else:
                target_next_q = target_next_q.max(1)[0].view(-1, 1)

            target_q = ops.td_target(batch.reward, batch.mask, target_next_q,
                                     batch.discount_factor * self._discount)

        self.model.sample_noise()
        expected_q = self.model(batch.state).gather(1, batch.action.long())
        if self._use_huber_loss_fn:
            return ops.huber_loss(expected_q, target_q)
        return ops.mse_loss(expected_q, target_q)
