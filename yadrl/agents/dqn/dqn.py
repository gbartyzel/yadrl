import copy
import random
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

import yadrl.common.utils as utils
from yadrl.agents.agent import OffPolicyAgent
from yadrl.common.scheduler import BaseScheduler
from yadrl.networks.head import Head


class DQN(OffPolicyAgent, agent_type='dqn'):
    head_types = ['simple', 'dueling_simple']

    def __init__(self,
                 learning_rate: float,
                 grad_norm_value: float = 0.0,
                 exploration_strategy: BaseScheduler = BaseScheduler(),
                 noise_type: str = 'none',
                 use_double_q: bool = False,
                 use_dueling: bool = False,
                 **kwargs):
        self._noise_type = noise_type
        self._use_dueling = use_dueling
        super().__init__(**kwargs)
        self._grad_norm_value = grad_norm_value
        self._use_double_q = use_double_q
        self._epsilon_scheduler = exploration_strategy
        self._optim = optim.Adam(self.model.parameters(), lr=learning_rate,
                                 eps=0.01 / self._batch_size)

        print(self.model)

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
        batch = self._memory.sample(self._batch_size)
        loss = self._compute_loss(batch)

        self._writer.add_scalar('loss', loss, self._env_step)
        self._optim.zero_grad()
        loss.backward()
        if self._grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self._grad_norm_value)
        self._optim.step()
        self._update_target(self.model, self.target_model)

    def _compute_loss(self, batch):
        state = self._state_normalizer(batch.state, self._device)
        next_state = self._state_normalizer(batch.next_state, self._device)

        with torch.no_grad():
            self.target_model.sample_noise()
            target_next_q = self.target_model(next_state, True)
            if self._use_double_q:
                self.model.sample_noise()
                next_action = self.model(next_state).argmax(1, True)
                target_next_q = target_next_q.gather(1, next_action)
            else:
                target_next_q = target_next_q.max(1)[0].view(-1, 1)

        target_q = utils.td_target(
            reward=batch.reward,
            mask=batch.mask,
            target=target_next_q,
            discount=batch.discount_factor * self._discount)

        self.model.sample_noise()
        expected_q = self.model(state).gather(1, batch.action.long())
        loss = utils.huber_loss(expected_q, target_q)
        return loss
