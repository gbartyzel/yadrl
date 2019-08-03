from typing import Optional
from typing import Tuple

import numpy as np


class RunningMeanStd:
    def __init__(self,
                 dim: Tuple[int, ...],
                 eps: Optional[float] = 1e-8):
        self._count = eps
        self._mean = np.ones(dim) * eps
        self._var = np.zeros(dim)

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._count < 2:
            return np.nan, np.nan
        else:
            return self._mean, np.sqrt(self._var)

    def update(self, batch_value: np.ndarray):
        if not isinstance(batch_value, np.ndarray):
            batch_value = np.array([batch_value])
        if len(batch_value.shape) == 1:
            batch_value = np.reshape(batch_value, (-1, batch_value.shape[0]))

        batch_count = batch_value.shape[0]
        batch_mean = np.mean(batch_value, axis=0)
        batch_var = np.var(batch_value, axis=0)

        self._compute_new_rms(batch_count, batch_mean, batch_var)

    def set_parameters(self,
                       mean: np.ndarray,
                       variance: np.ndarray,
                       count: int):
        self._mean = mean
        self._var = variance
        self._count = count

    def _compute_new_rms(self,
                         batch_count: int,
                         batch_mean: np.ndarray,
                         batch_var: np.ndarray):
        self._count += batch_count
        delta = batch_mean - self._mean
        self._mean = ((self._count * self._mean + batch_count * batch_mean)
                      / (self._count + batch_count))

        m_a = self._var * (self._count - 1)
        m_b = batch_var * (batch_count - 1)

        m2 = (m_a + m_b + delta ** 2 * self._count * batch_count
              / (self._count + batch_count))
        self._var = m2 / (self._count + batch_count - 1)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._var

    @property
    def count(self):
        return self._count
