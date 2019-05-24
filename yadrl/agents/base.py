import abc
from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn

from yadrl.common.replay_memory import ReplayMemory


class BaseOffPolicy(metaclass=abc.ABC):
    def __init__(self,
                 state_dim,
                 action_dim,
                 discount_factor,
                 polyak_factor,
                 memory_capacity,
                 batch_size,
                 warm_up_steps):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._state_dim = state_dim
        self._action_dim = action_dim
        self._discount = discount_factor
        self._polyak = polyak_factor

        self._step = 0
        self._batch_size = batch_size
        self._warm_up_steps = warm_up_steps

        self._memory = ReplayMemory(memory_capacity, state_dim, action_dim, True)

    @abc.abstractmethod
    def act(self, *args):
        return NotImplementedError

    @abc.abstractmethod
    def observe(self, *args):
        return NotImplementedError

    @abc.abstractmethod
    def update(self):
        return NotImplementedError

    def _soft_update(self, params: nn.parameter, target_params: nn.parameter):
        for param, t_param in zip(params, target_params):
            t_param.data.copy_(t_param.data * (1.0 - self._polyak) + param.data * self._polyak)

    @staticmethod
    def _hard_update(model: nn.Module, target_model: nn.Module):
        target_model.load_state_dict(model.state_dict())
