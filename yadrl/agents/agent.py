import abc
import datetime
import os
from typing import Any, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import tqdm
from gym.spaces.box import Box
from torch.utils.tensorboard import SummaryWriter

from yadrl.common.memory import ReplayMemory, Rollout
from yadrl.common.normalizer import DummyNormalizer


class BaseAgent(abc.ABC):
    registered_agents = {}

    def __init_subclass__(cls, agent_type: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if agent_type is not None:
            cls.registered_agents[agent_type] = cls

    @classmethod
    def build(cls, agent_type: str, **kwargs):
        return cls.registered_agents[agent_type](**kwargs)

    def __init__(self,
                 env: gym.Env,
                 body,
                 state_normalizer: DummyNormalizer = DummyNormalizer(),
                 reward_scaling: float = 1.0,
                 discount_factor: float = 0.99,
                 batch_size: int = 64,
                 n_step: int = 1,
                 update_steps: int = 1,
                 experiment_name: str = './output',
                 seed: int = 1337):
        super().__init__()
        self._env = env
        self._state = None
        self._env_step = 0
        self._optimizer_step = 0
        self._set_seeds(seed)
        self._data_to_log = dict()

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self._state_dim = int(np.prod(self._env.observation_space.shape))
        if isinstance(self._env.action_space, Box):
            self._action_dim = self._env.action_space.shape[0]
        else:
            self._action_dim = self._env.action_space.n

        self._discount = discount_factor
        self._n_step = n_step
        self._batch_size = batch_size
        self._reward_scaling = reward_scaling
        self._update_steps = update_steps

        self._rollout = Rollout(length=n_step, discount_factor=discount_factor)
        self._state_normalizer = state_normalizer

        self._writer = SummaryWriter(self._create_logdir(experiment_name))

        self._networks = self._initialize_networks(body)

    def train(self, max_steps: int):
        pass

    def eval(self, render: bool = False):
        self._state = self._env.reset()
        while True:
            if render:
                self._env.render()
            transition = self.step(False, False)
            if transition[-1]:
                break

    def step(self, train: bool, random_action: bool = False):
        if random_action:
            action = self._env.action_space.sample()
        else:
            action = self._act(self._state, train)
        next_state, reward, done, _ = self._env.step(action)
        transition = (self._state, action, reward, next_state, done)
        self._observe(*transition)
        self._state = next_state
        return transition

    def load(self, path: str):
        model = torch.load(path)
        if model:
            for k in self._networks.keys():
                self._networks[k].load_state_dict(model[k])
            self._env_step = model['step']

    def save(self):
        state_dict = {k: net.state_dict() for k, net in self._networks.items()}
        state_dict['step'] = self._env_step
        torch.save(state_dict, 'model_{}.pth'.format(self._env_step))

    def _act(self, state: int, *args) -> np.ndarray:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        state = self._state_normalizer(state, self._device)
        return state

    def _observe(self,
                 state: Union[np.ndarray, torch.Tensor],
                 action: Union[np.ndarray, torch.Tensor],
                 reward: Union[float, torch.Tensor],
                 next_state: Union[np.ndarray, torch.Tensor],
                 done: Any):
        pass

    def _log(self, reward):
        self._writer.add_scalar('train/reward', reward, self._env_step)
        for k, v in self._data_to_log.items():
            self._writer.add_scalar(k, v, self._env_step)

        """
        for name, param in self.parameters:
            self._writer.add_histogram(
                'main/{}'.format(name), param, self._env_step)
        for name, param in self.target_parameters:
            self._writer.add_histogram(
                'target/{}'.format(name), param, self._env_step)
        """

    @property
    @abc.abstractmethod
    def parameters(self):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def target_parameters(self):
        return NotImplementedError

    @abc.abstractmethod
    def _update(self):
        return NotImplementedError

    @abc.abstractmethod
    def _initialize_networks(self, *args, **kwargs):
        return NotImplementedError

    @staticmethod
    def _set_seeds(seed):
        torch.random.manual_seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _create_logdir(log_dir: str) -> str:
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        now = datetime.datetime.now().strftime('%d_%m_%y_%H_%M_%S')
        return os.path.join(log_dir, now)


class OffPolicyAgent(BaseAgent):

    def __init__(self,
                 memory: ReplayMemory,
                 warm_up_steps: int = 64,
                 polyak_factor: float = 0.0,
                 update_frequency: int = 1,
                 target_update_frequency: int = 1000,
                 **kwargs):
        super().__init__(**kwargs)
        self._warm_up_steps = warm_up_steps

        self._use_soft_update = polyak_factor > 0.0
        self._polyak = polyak_factor
        self._update_frequency = update_frequency
        self._target_update_frequency = target_update_frequency
        self._memory = memory

    def train(self, max_steps: int):
        self._state = self._env.reset()
        total_reward = []
        pb = tqdm.tqdm(total=max_steps)
        while self._env_step < max_steps:
            exploration_flag = self._memory.size < self._warm_up_steps
            transition = self.step(True, exploration_flag)
            total_reward.append(transition[2])
            pb.update(1)
            if transition[-1]:
                self._state = self._env.reset()
                if self._env_step > 0:
                    self._log(sum(total_reward))
                total_reward = []
        pb.close()

    def _observe(self,
                 state: Union[np.ndarray, torch.Tensor],
                 action: Union[np.ndarray, torch.Tensor],
                 reward: Union[float, torch.Tensor],
                 next_state: Union[np.ndarray, torch.Tensor],
                 done: Any):
        self._state_normalizer.update(state)
        transition = self._rollout(state, action, reward, next_state, done)
        if transition is None:
            return
        for t in transition:
            self._memory.push(*t)
        if self._memory.size >= self._warm_up_steps:
            self._env_step += 1
            if self._env_step % self._update_frequency == 0:
                for _ in range(self._update_steps):
                    self._optimizer_step += 1
                    self._update()
        if done:
            self._rollout.reset()

    def _update_target(self, model: nn.Module, target_model: nn.Module):
        if self._use_soft_update:
            self._soft_update(model.parameters(), target_model.parameters())
        else:
            if self._env_step / self._update_frequency \
                    % self._target_update_frequency == 0:
                target_model.load_state_dict(model.state_dict())

    def _soft_update(self, params: nn.parameter, target_params: nn.parameter):
        for param, t_param in zip(params, target_params):
            t_param.data.copy_(
                t_param.data * (1.0 - self._polyak) + param.data * self._polyak)

    @staticmethod
    def _hard_update(model: nn.Module, target_model: nn.Module):
        target_model.load_state_dict(model.state_dict())

    @abc.abstractmethod
    def _update(self):
        return NotImplementedError
