import abc
from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn

from yadrl.common.replay_memory import ReplayMemory
from yadrl.common.checkpoint_manager import CheckpointManager


class BaseOffPolicy(abc.ABC):
    def __init__(self,
                 agent_type: str,
                 state_dim: int,
                 action_dim: int,
                 discount_factor: float,
                 polyak_factor: float,
                 n_step: int,
                 memory_capacity: int,
                 batch_size: int,
                 warm_up_steps: int,
                 update_frequency: int,
                 logdir: str):
        super(BaseOffPolicy, self).__init__()
        self.step = 0

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self._state_dim = state_dim
        self._action_dim = action_dim
        self._discount = discount_factor
        self._polyak = polyak_factor
        self._n_step = n_step

        self._batch_size = batch_size
        self._warm_up_steps = warm_up_steps
        self._update_frequency = update_frequency

        self._checkpoint_manager = CheckpointManager(agent_type, logdir)

        self._memory = ReplayMemory(memory_capacity, state_dim, action_dim,
                                    True)

    @abc.abstractmethod
    def act(self, *args):
        return NotImplementedError

    @abc.abstractmethod
    def update(self):
        return NotImplementedError

    @abc.abstractmethod
    def load(self):
        return NotImplementedError

    @abc.abstractmethod
    def save(self):
        return NotImplementedError

    def observe(self,
                state: Union[np.ndarray, torch.Tensor],
                action: Union[np.ndarray, torch.Tensor],
                reward: Union[float, torch.Tensor],
                next_state: Union[np.ndarray, torch.Tensor],
                done: Any):
        self._memory.push(state, action, reward, next_state, done)
        if self._memory.size > self._warm_up_steps:
            self.step += 1
            if self.step % self._update_frequency == 0:
                self.update()

    def _soft_update(self, params: nn.parameter, target_params: nn.parameter):
        for param, t_param in zip(params, target_params):
            t_param.data.copy_(
                t_param.data * (1.0 - self._polyak) + param.data * self._polyak)

    def _td_target(self,
                   reward: torch.Tensor,
                   mask: torch.Tensor,
                   next_value: torch.Tensor) -> torch.Tensor:
        return reward + mask * self._discount ** self._n_step * next_value

    @staticmethod
    def _hard_update(model: nn.Module, target_model: nn.Module):
        target_model.load_state_dict(model.state_dict())

    @staticmethod
    def _mse_loss(prediction: torch.Tensor,
                  target: torch.Tensor) -> torch.Tensor:
        return torch.mean(0.5 * (prediction - target).pow(2))
