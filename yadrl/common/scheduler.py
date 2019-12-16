class BaseScheduler:
    def __init__(self,
                 start_value: float = 1.0,
                 end_value: float = 0.1,
                 annealing_steps: float = 1e4):
        self._start_value = start_value
        self._end_value = end_value
        self._annealing_steps = annealing_steps
        self._value = start_value

    def reset(self):
        self._value = self._start_value

    def step(self) -> float:
        return self._start_value


class LinearScheduler(BaseScheduler):
    def __init__(self,
                 start_value: float,
                 end_value: float,
                 annealing_steps: float):
        super(LinearScheduler, self).__init__(
            start_value, end_value, annealing_steps)
        self._decay_value = ((self._start_value - self._end_value)
                             / self._annealing_steps)

    def step(self) -> float:
        self._value -= self._decay_value
        self._value = max(self._value, self._end_value)
        return self._value


class ExponentialScheduler(BaseScheduler):
    def __init__(self,
                 start_value: float,
                 end_value: float,
                 annealing_steps: float):
        super(ExponentialScheduler, self).__init__(
            start_value, end_value, annealing_steps)
        self._decay_value = ((self._start_value - self._end_value)
                             * (self._start_value / self._end_value)
                             / self._annealing_steps)

    def step(self) -> float:
        self._value *= self._decay_value
        self._value = max(self._value, self._end_value)
        return self._value
