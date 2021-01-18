import abc
from copy import deepcopy
from typing import Dict, Union

import gym
import numpy as np
import torch as th
import torch.nn as nn
import tqdm
from gym.spaces.box import Box
from torch.utils.tensorboard import SummaryWriter

import yadrl.common.ops as ops
import yadrl.common.types as t
from yadrl.common.wrappers import apply_wrappers
from yadrl.common.memory import ReplayMemory, Rollout
from yadrl.common.normalizer import DummyNormalizer


class Agent(abc.ABC):
    registered_agents: Dict[str, "Agent"] = {}

    def __init_subclass__(cls, agent_type: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if agent_type is not None:
            cls.registered_agents[agent_type] = cls

    @classmethod
    def build(cls, agent_type: str, **kwargs) -> "Agent":
        import os

        for file in os.listdir(os.path.dirname(__file__)):
            if "__" in file:
                continue
            if os.path.isdir(os.path.join(os.path.dirname(__file__), file)):
                exec("import yadrl.agents." + file)
        return cls.registered_agents[agent_type](**kwargs)

    def __init__(
        self,
        env_id: str,
        env_wrappers: list,
        body: nn.Module,
        state_normalizer: DummyNormalizer = DummyNormalizer(),
        reward_scaling: float = 1.0,
        discount_factor: float = 0.99,
        batch_size: int = 64,
        n_step: int = 1,
        update_steps: int = 1,
        experiment_name: str = "",
        log_path: str = "./output",
        seed: int = 1337,
    ):
        super().__init__()
        self._env = apply_wrappers(gym.make(env_id), env_wrappers)
        self._state = None
        self._env_step = 0
        self._gradient_step = 0
        ops.set_seeds(seed)
        self._data_to_log = dict()

        self._device = th.device("cuda" if th.cuda.is_available() else "cpu")

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

        self._rollout = Rollout(
            n_step, self._env.action_space, self._env.observation_space
        )
        self._state_normalizer = state_normalizer

        self._writer = SummaryWriter(ops.create_log_dir(log_path, experiment_name))

        self._networks = self._initialize_networks(body)

    def train(self, max_steps: int):
        raise NotImplementedError

    def eval(self, render: bool = False):
        self._state = self._env.reset()
        while True:
            if render:
                self._env.render()
            transition = self.step(False, False)
            if transition[-1]:
                break

    def step(self, train: bool, random_action: bool = False) -> t.TTransition:
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
        model = th.load(path)
        if model:
            for k in self._networks.keys():
                self._networks[k].load_state_dict(model[k])
            self._env_step = model["step"]

    def save(self):
        state_dict = {k: net.state_dict() for k, net in self._networks.items()}
        state_dict["step"] = self._env_step
        th.save(state_dict, "model_{}.pth".format(self._env_step))

    def _act(self, state: np.ndarray, *args) -> t.TData:
        state = ops.to_tensor(state, self._device).unsqueeze(0)
        return self._state_normalizer(state, self._device)

    def _observe(
        self,
        state: t.TData,
        action: Union[np.ndarray, int],
        reward: float,
        next_state: t.TData,
        done: bool,
    ):
        pass

    def _log(self, reward: float):
        self._writer.add_scalar("train/reward", reward, self._env_step)
        for k, v in self._data_to_log.items():
            self._writer.add_scalar(k, v, self._env_step)

        for name, module in self._networks.items():
            for p_name, param in module.named_parameters():
                self._writer.add_histogram(
                    "{}/{}".format(name, p_name), param, self._env_step
                )

    @property
    @abc.abstractmethod
    def parameters(self) -> t.TNamedParameters:
        pass

    @property
    @abc.abstractmethod
    def target_parameters(self) -> t.TNamedParameters:
        pass

    @abc.abstractmethod
    def _update(self):
        return NotImplementedError

    @abc.abstractmethod
    def _initialize_networks(self, phi: t.TModuleDict) -> t.TModuleDict:
        pass


class OffPolicyAgent(Agent):
    def __init__(
        self,
        memory: ReplayMemory,
        warm_up_steps: int = 64,
        polyak_factor: float = 0.0,
        update_frequency: int = 1,
        target_update_frequency: int = 1000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._warm_up_steps = warm_up_steps

        self._use_soft_update = polyak_factor > 0.0
        self._polyak = polyak_factor
        self._update_frequency = update_frequency
        self._target_update_frequency = target_update_frequency
        self._memory = memory

    def train(self, max_steps: int):
        self._state = self._env.reset()
        print(self._state.shape)
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

    def _observe(
        self,
        state: t.TData,
        action: t.TActionOption,
        reward: float,
        next_state: t.TData,
        done: bool,
    ):
        self._state_normalizer.update(state)
        self._rollout.push(state, action, reward, next_state, done, self._discount)
        transition = self._rollout.sample()
        if transition is None:
            return
        for t in transition:
            self._memory.push(*t)
        if self._memory.size >= self._warm_up_steps:
            self._env_step += 1
            if self._env_step % self._update_frequency == 0:
                for _ in range(self._update_steps):
                    self._gradient_step += 1
                    self._update()
        if done:
            self._rollout.reset()

    def _update_target(self, model: nn.Module, target_model: nn.Module):
        if self._use_soft_update:
            ops.soft_update(model.parameters(), target_model.parameters(), self._polyak)
        else:
            if (
                self._env_step / self._update_frequency % self._target_update_frequency
                == 0
            ):
                target_model.load_state_dict(model.state_dict())

    @abc.abstractmethod
    def _update(self):
        return NotImplementedError
