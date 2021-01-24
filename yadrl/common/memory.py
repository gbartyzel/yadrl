from collections import namedtuple
from typing import Sequence, Tuple, Union

import gym
import numpy as np
import torch as th

import yadrl.common.types as t
from yadrl.common.normalizer import DummyNormalizer
from yadrl.common.ops import to_tensor

Batch = namedtuple(
    "Batch", ["state", "action", "reward", "next_state", "mask", "discount_factor"]
)


class BaseMemory:
    def __init__(
        self,
        capacity: int,
        action_space: gym.spaces.Space,
        observation_space: gym.spaces.Space,
    ):
        self._capacity = capacity
        self._size = 0
        self._transition_idx = 0

        self._observation_buffer = np.zeros(
            (capacity,) + observation_space.shape, observation_space.dtype
        )
        if isinstance(action_space, gym.spaces.Box):
            self._action_buffer = np.zeros(
                (capacity,) + action_space.shape, action_space.dtype
            )
        if isinstance(action_space, gym.spaces.Discrete):
            self._action_buffer = np.zeros((capacity, 1), action_space.dtype)
        self._reward_buffer = np.zeros((capacity, 1), np.float32)
        self._next_observation_buffer = np.zeros(
            (capacity,) + observation_space.shape, observation_space.dtype
        )
        self._terminal_buffer = np.zeros((capacity, 1), np.bool)
        self._discount_buffer = np.zeros((capacity, 1), np.float32)

    def push(
        self,
        state: np.ndarray,
        action: t.TActionOption,
        reward: float,
        next_state: np.ndarray,
        terminal: bool,
        discount_factor: float,
    ):
        self._observation_buffer[self._transition_idx, ...] = state
        self._action_buffer[self._transition_idx, ...] = action
        self._reward_buffer[self._transition_idx, ...] = reward
        self._next_observation_buffer[self._transition_idx, ...] = next_state
        self._terminal_buffer[self._transition_idx, ...] = terminal
        self._discount_buffer[self._transition_idx, ...] = discount_factor
        self._size = min(self._size + 1, self._capacity)

    def popleft(self):
        self._observation_buffer = self._np_popleft(self._observation_buffer)
        self._action_buffer = self._np_popleft(self._action_buffer)
        self._reward_buffer = self._np_popleft(self._reward_buffer)
        self._next_observation_buffer = self._np_popleft(self._next_observation_buffer)
        self._terminal_buffer = self._np_popleft(self._terminal_buffer)
        self._discount_buffer = self._np_popleft(self._discount_buffer)
        self._size -= 1

    @staticmethod
    def _np_popleft(buffer: np.ndarray) -> np.ndarray:
        temp_buffer = np.roll(buffer, -1, 0)
        temp_buffer[-1] = np.zeros(buffer.shape[1:])
        return temp_buffer

    @property
    def size(self) -> int:
        return self._size

    def reset(self):
        self._observation_buffer = np.zeros_like(self._observation_buffer)
        self._action_buffer = np.zeros_like(self._action_buffer)
        self._reward_buffer = np.zeros_like(self._reward_buffer)
        self._next_observation_buffer = np.zeros_like(self._next_observation_buffer)
        self._terminal_buffer = np.zeros_like(self._terminal_buffer)
        self._discount_buffer = np.zeros_like(self._discount_buffer)
        self._size = 0
        self._transition_idx = 0

    def __getitem__(self, item: int) -> t.TTransition:
        state = self._observation_buffer[item]
        action = self._action_buffer[item]
        reward = self._reward_buffer[item]
        next_state = self._next_observation_buffer[item]
        terminal = self._terminal_buffer[item]
        discount_factor = self._discount_buffer[item]
        return state, action, reward, next_state, terminal, discount_factor


class ReplayMemory(BaseMemory):
    def __init__(self, combined: bool = False, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self._combined = combined
        self._device = th.device(device)

    def push(
        self,
        state: np.ndarray,
        action: t.TActionOption,
        reward: float,
        next_state: np.ndarray,
        terminal: bool,
        discount_factor: float,
    ):
        super().push(state, action, reward, next_state, terminal, discount_factor)
        self._transition_idx = (self._transition_idx + 1) % self._capacity

    def sample(
        self, batch_size: int, state_normalizer: DummyNormalizer = None
    ) -> Batch:
        if state_normalizer is None:
            state_normalizer = DummyNormalizer()
        if self._combined:
            batch_size -= 1
        idxs = np.random.randint((self._size - 1), size=batch_size)
        if self._combined:
            idxs = np.append(idxs, np.array(self._transition_idx))
        state_b = state_normalizer(
            to_tensor(self._observation_buffer[idxs, ...], self._device), self._device
        )
        action_b = to_tensor(self._action_buffer[idxs, ...], self._device)
        reward_b = to_tensor(self._reward_buffer[idxs, ...], self._device)
        next_state_b = state_normalizer(
            to_tensor(self._next_observation_buffer[idxs, ...], self._device),
            self._device,
        )
        mask_b = to_tensor(self._terminal_buffer[idxs, ...], self._device)
        discount_b = to_tensor(self._discount_buffer[idxs, ...], self._device)
        return Batch(
            state=state_b,
            action=action_b,
            reward=reward_b,
            next_state=next_state_b,
            mask=mask_b,
            discount_factor=discount_b,
        )


class Rollout(BaseMemory):
    @property
    def ready(self) -> int:
        return self._size == self._capacity

    def push(
        self,
        state: np.ndarray,
        action: Union[np.ndarray, int],
        reward: float,
        next_state: np.ndarray,
        terminal: bool,
        discount_factor: float,
    ):
        if self._size == self._capacity:
            self.popleft()
        super().push(state, action, reward, next_state, terminal, discount_factor)
        self._transition_idx = min(self._transition_idx + 1, self._capacity - 1)

    def sample(self) -> Sequence[t.TTransition]:
        if self.ready:
            if self._terminal_buffer[-1]:
                transitions = list()
                for _ in range(self._size):
                    transitions.append(self._get_transition())
                    self.popleft()
                return transitions
            return (self._get_transition(),)
        return None

    def _get_transition(self) -> t.TTransition:
        reward, discount_factor = self._compute_cumulative_reward()
        state: np.ndarray = self._observation_buffer[0]
        action: np.ndarray = self._action_buffer[0]
        terminal: bool = self._terminal_buffer[self._size - 1]
        next_state: np.ndarray = self._next_observation_buffer[self._size - 1]
        return state, action, reward, next_state, terminal, discount_factor

    def _compute_cumulative_reward(self) -> Tuple[float, float]:
        cum_reward = 0
        discount = 0.0
        for t in range(self._size):
            discount = self._discount_buffer[t] ** t
            cum_reward += discount * self._reward_buffer[t]
        return cum_reward, discount
