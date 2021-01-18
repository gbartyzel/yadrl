import gym


class ToNCHW(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        assert len(env.observation_space.shape) == 3
        shp = self.observation_space.shape

        self.observation_space = Box(
            low=0,
            high=255,
            shape=tuple(shp.insert(0, shp.pop())),
            dtype=np.uint8,
        )

    def observation_space(self, observation):
        return observation.transpose(2, 0, 1)


_WRAPPERS = {
    "resize": gym.wrappers.ResizeObservation,
    "frame_stack": gym.wrappers.FrameStack,
    "to_grayscale": gym.wrappers.GrayScaleObservation,
    "to_nchw": ToNCHW,
}


def apply_wrappers(env, selected_wrappers):
    for w in selected_wrappers:
        wrap_fn = _WRAPPERS[w["type"]]
        print(w)
        if "parameters" in w:
            env = wrap_fn(env, **w["parameters"])
        else:
            env = wrap_fn(env)
    return env
