import copy
from typing import Sequence

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim

import yadrl.common.ops as ops
import yadrl.common.types as t
from yadrl.agents.dqn.dqn import DQN
from yadrl.common.memory import Batch
from yadrl.common.scheduler import BaseScheduler
from yadrl.networks.head import Head


class MaxminDQN(DQN, agent_type="maxmin_dqn"):
    def __init__(self, num_heads: int = 2, update_subset_size: int = 1, **kwargs):
        self._num_heads = num_heads
        self._update_subset_size = update_subset_size
        super().__init__(**kwargs)

    def _initialize_networks(self, phi: t.TModuleDict) -> t.TModuleDict:
        head_type = self.head_types[int(self._use_dueling)]
        support_dim = self._support_dim if hasattr(self, "_support_dim") else 1
        network = Head.build(
            head_type="multi",
            inner_head_type=head_type,
            phi=phi,
            hidden_dim=self._head_hidden_dim,
            support_dim=support_dim,
            output_dim=self._action_dim,
            noise_type=self._noise_type,
        )
        target_network = copy.deepcopy(network)
        network.to(self._device)
        target_network.to(self._device)
        target_network.eval()
        return {"model": network, "target_model": target_network}

    def _sample_q(self, state: th.Tensor, train: bool = False) -> th.Tensor:
        self.model.reset_noise()
        if train:
            self.model.sample_noise()
        q_values = self.model(state)
        return th.cat([q.unsqueeze(1) for q in q_values], dim=1).min(1)[0]

    def _compute_loss(self, batch: Batch) -> th.Tensor:
        heads = np.random.randint(0, self._num_heads, size=(self._update_subset_size,))
        with th.no_grad():
            self.target_model.sample_noise()
            target_next_qs = self.target_model(batch.next_state)
            target_next_qs = th.cat([q.unsqueeze(1) for q in target_next_qs], 1)
            target_next_q = target_next_qs.min(1)[0].max(1)[0].view(-1, 1)
            target_q = ops.td_target(
                batch.reward,
                batch.mask,
                target_next_q,
                batch.discount_factor * self._discount,
            )

        self.model.sample_noise()

        loss = 0.0
        for i in heads:
            expected_q = self.model.evaluate_head(batch.state, idx=i).gather(
                1, batch.action.long()
            )
            if self._use_huber_loss_fn:
                loss += ops.huber_loss(expected_q, target_q)
            else:
                loss += ops.mse_loss(expected_q, target_q)
        return loss / self._update_subset_size
