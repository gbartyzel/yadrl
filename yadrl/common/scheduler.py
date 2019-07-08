class LinearScheduler(object):
    def __init__(self,
                 start_value: float,
                 end_value: float,
                 annealing_steps: float):
        self._start_value = start_value
        self._end_value = end_value
        self._decay_value = (start_value - end_value) / annealing_steps

        self._value = start_value

    def reset(self):
        self._value = self._start_value

    def step(self) -> float:
        self._value -= self._decay_value
        self._value = max(self._value, self._end_value)
        return self._value
