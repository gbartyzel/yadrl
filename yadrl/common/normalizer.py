from typing import Dict, Tuple, Union

import numpy as np
import torch as th

import yadrl.common.types as t
from yadrl.common.ops import to_numpy, to_tensor
from yadrl.common.running_mean_std import RunningMeanStd


class DummyNormalizer:
    def __init__(self, **kwargs):
        pass

    def __call__(self,
                 batch_input: t.TData,
                 device: th.device = th.device('cpu')) -> t.TData:
        return batch_input

    def update(self, batch_input):
        pass

    def load(self, state_dict: Dict[str, Union[np.ndarray, int]]):
        pass

    def state_dict(self) -> Dict[str, Union[np.ndarray, int]]:
        state_dict = {}
        return state_dict


class RMSNormalizer(DummyNormalizer):
    def __init__(self,
                 dim: Tuple[int, ...],
                 clip_min: float = -5.0,
                 clip_max: float = 5.0,
                 **kwargs):
        super().__init__(**kwargs)
        self._rms = RunningMeanStd(dim)
        self._clip = (clip_min, clip_max)

    def __call__(self,
                 batch_input: t.TData,
                 device: th.device = th.device('cpu')) -> t.TData:
        mean, std = self._rms()
        if isinstance(batch_input, th.Tensor):
            th_mean = to_tensor(mean, device)
            th_std = to_tensor(std, device)
            return th.clamp((batch_input - th_mean) / th_std, *self._clip)
        return np.clip(batch_input - mean / std, *self._clip)

    def update(self, batch_input: t.TData):
        if isinstance(batch_input, th.Tensor):
            self._rms.update(to_numpy(batch_input))
        else:
            self._rms.update(batch_input)

    def load(self, state_dict: Dict[str, Union[np.ndarray, int]]):
        mean = state_dict['mean']
        variance = state_dict['variance']
        count = state_dict['count']
        self._rms.set_parameters(mean, variance, count)

    def state_dict(self) -> Dict[str, Union[np.ndarray, int]]:
        state_dict = {
            'mean': self._rms.mean,
            'variance': self._rms.variance,
            'count': self._rms.count
        }
        return state_dict


class ScaleNormalizer(DummyNormalizer):
    def __init__(self,
                 target_min: np.ndarray,
                 target_max: np.ndarray,
                 source_min: np.ndarray,
                 source_max: np.ndarray,
                 **kwargs):
        super().__init__(**kwargs)
        self._t_min = target_min
        self._t_max = target_max
        self._s_min = source_min
        self._s_max = source_max

    def __call__(self,
                 batch_input: Union[np.ndarray, th.Tensor],
                 device: th.device = th.device('cpu')) -> t.TData:
        t_min = self._t_min
        t_max = self._t_max
        s_min = self._s_min
        s_max = self._s_max
        if isinstance(batch_input, th.Tensor):
            t_min = to_tensor(t_min, device)
            t_max = to_tensor(t_max, device)
            s_min = to_tensor(s_min, device)
            s_max = to_tensor(s_max, device)
        return (batch_input - s_min) / (s_max - s_min) * (t_max - t_min) + t_min


class ImageNormalizer(DummyNormalizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._scale_factor = 1.0 / 256.0

    def __call__(self,
                 batch_input: t.TData,
                 device: th.device = th.device('cpu')) -> t.TData:
        return batch_input * self._scale_factor
