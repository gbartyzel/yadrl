from collections import deque
from collections import namedtuple
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch

Batch = namedtuple('Batch', ['state', 'action', 'reward', 'next_state', 'mask',
                             'discount_factor'])
_TRANSITION = Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]


class ReplayMemory(object):
    def __init__(self,
                 capacity: int,
                 combined: bool = False,
                 torch_backend: bool = False,
                 device: torch.device = torch.device('cpu')):
        self._combined = combined
        self._buffer = deque(maxlen=capacity)
        self._torch_backend = torch_backend
        self._device = device

    def push(self,
             state: np.ndarray,
             action: Union[np.ndarray, int, float],
             reward: Union[np.ndarray, float],
             next_state: np.ndarray,
             terminal: Union[np.ndarray, float, bool],
             discount_factor: Union[float]):
        self._buffer.append((state, action, reward, next_state,
                             terminal, discount_factor))

    def sample(self, batch_size: int) -> Batch:
        if self._combined:
            batch_size -= 1
        idxs = np.random.randint((self.size - 1), size=batch_size)
        if self._combined:
            idxs = np.append(idxs, np.array(self.size - 1, dtype=np.int32))

        return self._encode_batch(idxs)

    def _encode_batch(self, idxs: np.ndarray) -> Batch:
        state, action, reward, next_state, mask, gamma = [], [], [], [], [], []
        for idx in idxs:
            transition = self._buffer[idx]
            state.append(np.array(transition[0]))
            action.append(np.array(transition[1]))
            reward.append(np.array(transition[2]))
            next_state.append(np.array(transition[3]))
            mask.append(np.array(transition[4]))
            gamma.append(np.array(transition[5]))

        return Batch(state=self._to_torch(state),
                     action=self._to_torch(action),
                     reward=self._to_torch(reward),
                     next_state=self._to_torch(next_state),
                     mask=self._to_torch(mask),
                     discount_factor=self._to_torch(gamma))

    def _to_torch(self,
                  batch: List[np.ndarray]) -> Union[np.ndarray, torch.Tensor]:
        batch = np.array(batch)
        if len(batch.shape) == 1:
            batch = batch.reshape((-1, 1))
        if self._torch_backend:
            return torch.from_numpy(batch).to(self._device).float()
        return batch

    @property
    def size(self) -> int:
        return len(self._buffer)

    def __getitem__(self, item: int) -> Tuple[np.ndarray, ...]:
        state, action, reward, next_state, mask, gamma = self._buffer[item]
        return state, action, reward, next_state, mask, gamma


class Rollout:
    def __init__(self, length: int, discount_factor: float):
        self._discount_factor = discount_factor
        self._buffer = deque(maxlen=length)

    @property
    def ready(self) -> int:
        return len(self._buffer) == self._buffer.maxlen

    def __call__(self,
                 state: np.ndarray,
                 action: np.ndarray,
                 reward: Union[np.ndarray, float],
                 next_state: np.ndarray,
                 done: Union[np.ndarray, bool]) -> _TRANSITION:
        self._buffer.append((state, action, reward))
        if self.ready:
            if done:
                transitions = list()
                for _ in range(self._buffer.maxlen):
                    transitions.append(self._get_transition(next_state, done))
                    self._buffer.popleft()
                return transitions

            return (self._get_transition(next_state, done),)
        return None

    def reset(self):
        self._buffer.clear()

    def _get_transition(self,
                        next_state: np.ndarray,
                        done: Union[np.ndarray, bool]) -> _TRANSITION:
        cum_reward, discount_factor = self._compute_cumulative_reward()
        return (self._buffer[0][0], self._buffer[0][1],
                cum_reward, next_state, done, discount_factor)

    def _compute_cumulative_reward(self) -> float:
        cum_reward = 0
        discount = 0.0
        for t in range(len(self._buffer)):
            discount = self._discount_factor ** t
            cum_reward += discount * self._buffer[t][2]
        return cum_reward, discount


if __name__ == '__main__':
    rollout = Rollout(1, 0.99)
    for i in range(6):
        mask = i == 5
        tran = rollout(i, i, 1, i+1, mask)
        print(tran)
