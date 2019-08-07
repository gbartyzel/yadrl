import abc
from typing import Any
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn as nn

import yadrl.common.normalizer as normalizer
from yadrl.common.checkpoint_manager import CheckpointManager
from yadrl.common.memory import ReplayMemory, Rollout


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
                 logdir: str,
                 seed: int = 1337,
                 use_combined_experience_replay: bool = False,
                 use_reward_normalization: bool = False,
                 use_state_normalization: bool = False,
                 reward_norm_clip: Tuple[float, float] = (-1.0, 1.0),
                 state_norm_clip: Tuple[float, float] = (-5.0, 5.0)):
        super(BaseOffPolicy, self).__init__()
        self.step = 0
        self._set_seeds(seed)

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self._use_reward_normalization = use_reward_normalization
        self._use_state_normalization = use_state_normalization

        self._state_dim = state_dim
        self._action_dim = action_dim
        self._discount = discount_factor ** n_step
        self._polyak = polyak_factor
        self._n_step = n_step

        self._batch_size = batch_size
        self._warm_up_steps = warm_up_steps
        self._update_frequency = update_frequency

        self._checkpoint_manager = CheckpointManager(agent_type, logdir)

        self._memory = ReplayMemory(memory_capacity, state_dim, action_dim,
                                    use_combined_experience_replay, True)

        self._rollout = Rollout(n_step, state_dim, action_dim, discount_factor)

        if use_reward_normalization:
            self._reward_normalizer = normalizer.RMSNormalizer(
                (1,), *reward_norm_clip)
        else:
            self._reward_normalizer = normalizer.DummyNormalizer()

        if use_state_normalization:
            self._state_normalizer = normalizer.RMSNormalizer(
                (state_dim,), *state_norm_clip)
        else:
            self._state_normalizer = normalizer.DummyNormalizer()

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
        self._state_normalizer.update(state)
        self._reward_normalizer.update(reward)
        transition = self._rollout.get_transition(state, action, reward,
                                                  next_state, done)
        if transition is None:
            return
        self._memory.push(*transition)
        if self._memory.size > self._warm_up_steps:
            self.step += 1
            if self.step % self._update_frequency == 0:
                self.update()
        if done:
            self._rollout.reset()

    def _soft_update(self, params: nn.parameter, target_params: nn.parameter):
        for param, t_param in zip(params, target_params):
            t_param.data.copy_(
                t_param.data * (1.0 - self._polyak) + param.data * self._polyak)

    def _td_target(self,
                   reward: torch.Tensor,
                   mask: torch.Tensor,
                   next_value: torch.Tensor) -> torch.Tensor:
        reward = self._reward_normalizer(reward, self._device)
        return reward + mask * self._discount * next_value

    @staticmethod
    def _hard_update(model: nn.Module, target_model: nn.Module):
        target_model.load_state_dict(model.state_dict())

    @staticmethod
    def _set_seeds(seed):
        torch.random.manual_seed(seed)
        np.random.seed(seed)
