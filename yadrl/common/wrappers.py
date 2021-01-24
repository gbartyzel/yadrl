from collections import deque
from typing import Union

import gym
import numpy as np

import yadrl.common.types as t


class ToNCHW(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        assert len(env.observation_space.shape) == 3
        shp = self.observation_space.shape

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shp[-1],) + shp[:-1],
            dtype=np.uint8,
        )

    def observation(self, observation) -> np.ndarray:
        return observation.transpose(2, 0, 1)


class RescaleAction(gym.wrappers.RescaleAction):
    def __init__(
        self, env: gym.Env, a: Union[np.ndarray, float], b: Union[np.ndarray, float]
    ):
        if isinstance(a, list):
            a = np.asarray(a)
        if isinstance(b, list):
            b = np.asarray(b)
        super().__init__(env, a, b)


class ActionRepeat(gym.Wrapper):
    def __init__(self, env: gym.Env, repeats: int):
        super().__init__(env)
        self._repeat = repeats

    def step(self, action: t.TActionOption) -> t.TEnvReturn:
        cum_reward = 0.0
        for _ in range(self._repeat):
            state, reward, done, info = self.env.step(action)
            if done:
                break
            cum_reward += reward
        return state, cum_reward, done, info


class FrameStack(gym.Wrapper):
    def __init__(self, env: gym.Env, num_stack: int, merge_stack: bool = False):
        super().__init__(env)
        self.k = num_stack
        self.merge_stack = merge_stack
        self.frames = deque([], maxlen=num_stack)
        shp = env.observation_space.shape

        if merge_stack:
            new_shp = shp[:-1] + (shp[-1] * num_stack,)
        else:
            new_shp = (num_stack,) + shp
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=new_shp,
            dtype=env.observation_space.dtype,
        )

    def reset(self) -> np.ndarray:
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action: t.TActionOption) -> t.TEnvReturn:
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self) -> np.ndarray:
        assert len(self.frames) == self.k
        if self.merge_stack:
            return np.concatenate(self.frames, axis=-1)
        return np.concatenate(
            [np.expand_dims(arr, axis=0) for arr in self.frames], axis=0
        )


_WRAPPERS = {
    "resize": gym.wrappers.ResizeObservation,
    "frame_stack": FrameStack,
    "to_grayscale": gym.wrappers.GrayScaleObservation,
    "action_repeat": ActionRepeat,
    "rescale_action": RescaleAction,
    "to_nchw": ToNCHW,
}


def apply_wrappers(env: gym.Env, selected_wrappers: t.TWrappersOption) -> gym.Env:
    if selected_wrappers is None:
        return env
    for w in selected_wrappers:
        wrap_fn = _WRAPPERS[w["type"]]
        if "parameters" in w:
            env = wrap_fn(env, **w["parameters"])
        else:
            env = wrap_fn(env)
    return env
