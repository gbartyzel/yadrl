import copy
from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np
import torch

from yadrl.common.running_mean_std import RunningMeanStd


class DummyNormalizer:
    def __init__(self):
        pass

    def __call__(self,
                 batch_input,
                 device: torch.device = torch.device('cpu')):
        return batch_input

    def update(self, batch_input):
        return NotImplemented

    def load(self, state_dict: Dict[str, Union[np.ndarray, int]]):
        return NotImplemented

    def state_dict(self) -> Dict[str, Union[np.ndarray, int]]:
        state_dict = {}
        return state_dict


class RMSNormalizer(DummyNormalizer):
    def __init__(self,
                 dim: Tuple[int, ...],
                 clip_min: float = -5.0,
                 clip_max: float = 5.0):
        super(RMSNormalizer, self).__init__()
        self._rms = RunningMeanStd(dim)
        self._clip = (clip_min, clip_max)

    def __call__(self,
                 batch_input: Union[np.ndarray, torch.Tensor],
                 device: torch.device = torch.device('cpu')):
        mean, std = self._rms()
        if isinstance(batch_input, torch.Tensor):
            mean = torch.from_numpy(mean.copy()).float().to(device)
            std = torch.from_numpy(std.copy()).float().to(device)
            return torch.clamp((batch_input - mean) / std, *self._clip)
        return np.clip(batch_input - mean / std, *self._clip)

    def update(self, batch_input: Union[np.ndarray, torch.Tensor]):
        new_batch_input = copy.deepcopy(batch_input)
        if isinstance(batch_input, torch.Tensor):
            new_batch_input = batch_input.cpu().numpy()
        self._rms.update(new_batch_input)

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
                 source_max: np.ndarray):
        super(ScaleNormalizer, self).__init__()
        self._t_min = target_min
        self._t_max = target_max
        self._s_min = source_min
        self._s_max = source_max

    def __call__(self,
                 batch_input: Union[np.ndarray, torch.Tensor],
                 device: torch.device = torch.device('cpu')) -> Union[
        np.ndarray, torch.Tensor]:
        t_min = self._t_min
        t_max = self._t_max
        s_min = self._s_min
        s_max = self._s_max
        if isinstance(batch_input, torch.Tensor):
            t_min = torch.from_numpy(t_min).float().to(device)
            t_max = torch.from_numpy(t_max).float().to(device)
            s_min = torch.from_numpy(s_min).float().to(device)
            s_max = torch.from_numpy(s_max).float().to(device)
        return (batch_input - s_min) / (s_max - s_min) * (t_max - t_min) + t_min


class ImageNormalizer(DummyNormalizer):
    def __init__(self):
        super(ImageNormalizer, self).__init__()
        self._scale_factor = 1.0 / 256.0

    def __call__(self,
                 batch_input: Union[np.ndarray, torch.Tensor],
                 device: torch.device = torch.device('cpu')) -> Union[
        np.ndarray, torch.Tensor]:
        return batch_input * self._scale_factor
