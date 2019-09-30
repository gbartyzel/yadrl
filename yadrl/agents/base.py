import abc
from typing import Any
from typing import Tuple
from typing import Union

import gym
import numpy as np
import torch
import torch.nn as nn
import tqdm
from gym.spaces.box import Box
from torch.utils.tensorboard import SummaryWriter

import yadrl.common.normalizer as normalizer
from yadrl.common.checkpoint_manager import CheckpointManager
from yadrl.common.memory import ReplayMemory, Rollout


class BaseOffPolicy(abc.ABC):
    def __init__(self,
                 env: gym.Env,
                 agent_type: str,
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
                 use_state_normalization: bool = False,
                 state_norm_clip: Tuple[float, float] = (-5.0, 5.0)):
        super(BaseOffPolicy, self).__init__()
        self._env = env
        self._state = None
        self._step = 0
        self._set_seeds(seed)

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self._use_state_normalization = use_state_normalization

        self._state_dim = self._env.observation_space.shape[0]
        if isinstance(self._env.action_space, Box):
            self._action_dim = self._env.action_space.shape[0]
            memory_action = self._action_dim
        else:
            self._action_dim = self._env.action_space.n
            memory_action = 1

        self._discount = discount_factor ** n_step
        self._polyak = polyak_factor
        self._n_step = n_step

        self._batch_size = batch_size
        self._warm_up_steps = warm_up_steps
        self._update_frequency = update_frequency

        self._checkpoint_manager = CheckpointManager(agent_type, logdir)

        self._memory = ReplayMemory(
            capacity=memory_capacity,
            state_dim=self._state_dim,
            action_dim=memory_action,
            combined=use_combined_experience_replay,
            torch_backend=True)

        self._rollout = Rollout(
            capacity=n_step,
            state_dim=self._state_dim,
            action_dim=memory_action,
            discount_factor=discount_factor)

        if use_state_normalization:
            self._state_normalizer = normalizer.RMSNormalizer(
                (self._state_dim,), *state_norm_clip)
        else:
            self._state_normalizer = normalizer.DummyNormalizer()

        self._writer = SummaryWriter()

    def step(self, train: bool):
        if train and self._memory.size < self._warm_up_steps:
            action = self._env.action_space.sample()
        else:
            action = self._act(self._state, train)
        next_state, reward, done, _ = self._env.step(action)
        self._observe(self._state, action, reward, next_state, done)
        self._state = next_state
        return reward, done

    def train(self, max_steps: int):
        self._state = self._env.reset()
        total_reward = []
        pb = tqdm.tqdm(total=max_steps)
        while self._step < max_steps:
            reward, done = self.step(True)
            total_reward.append(reward)
            pb.update(1)
            if done:
                self._state = self._env.reset()
                if self._step > 0:
                    self._log(sum(total_reward))
                    self._loss = 0.0
                total_reward = []
        pb.close()

    def eval(self, render: bool = False):
        self._state = self._env.reset()
        while True:
            if render:
                self._env.render()
            _, done = self.step(False)
            if done:
                break

    def _observe(self,
                 state: Union[np.ndarray, torch.Tensor],
                 action: Union[np.ndarray, torch.Tensor],
                 reward: Union[float, torch.Tensor],
                 next_state: Union[np.ndarray, torch.Tensor],
                 done: Any):
        self._state_normalizer.update(state)
        transition = self._rollout.get_transition(state, action, reward,
                                                  next_state, done)
        if transition is None:
            return
        self._memory.push(*transition)
        if self._memory.size >= self._warm_up_steps:
            self._step += 1
            if self._step % self._update_frequency == 0:
                self._update()
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
        return reward + mask * self._discount * next_value

    @staticmethod
    def _hard_update(model: nn.Module, target_model: nn.Module):
        target_model.load_state_dict(model.state_dict())

    @staticmethod
    def _set_seeds(seed):
        torch.random.manual_seed(seed)
        np.random.seed(seed)

    def _log(self, reward):
        self._writer.add_scalar('reward', reward, self._step)
        for name, param in self.parameters:
            self._writer.add_histogram(
                'main/{}'.format(name), param, self._step)
        for name, param in self.target_parameters:
            self._writer.add_histogram(
                'target/{}'.format(name), param, self._step)

    @property
    @abc.abstractmethod
    def parameters(self):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def target_parameters(self):
        return NotImplementedError

    @abc.abstractmethod
    def _act(self, *args):
        return NotImplementedError

    @abc.abstractmethod
    def _update(self):
        return NotImplementedError

    @abc.abstractmethod
    def load(self):
        return NotImplementedError

    @abc.abstractmethod
    def save(self):
        return NotImplementedError
