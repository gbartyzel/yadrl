from dataclasses import InitVar, dataclass, field
from typing import Any, Tuple, Union

import gym
import yaml
import time

import yadrl.common.exploration_noise as noise
import yadrl.common.normalizer as norm
import yadrl.common.scheduler as sch
import yadrl.common.types as t
from yadrl.common.memory import ReplayMemory
from yadrl.networks.body import Body
from yadrl.networks.body_parameter import BodyParameters
from yadrl.common.wrappers import apply_wrappers


@dataclass
class Configuration:
    env_id: str = field(init=False)
    env_wrappers: t.TWrappersOption = field(init=False, default=None)
    experiment_name: str = field(init=False)
    agent_type: str = field(init=False)
    common: t.TConfig = field(init=False)
    specific: t.TConfig = field(init=False)
    state_normalizer: Any = field(init=False, default=None)
    exploration_strategy: Any = field(init=False, default=None)
    memory: Any = field(init=False, default=None)
    body: Union[Body, t.TModuleDict] = field(init=False)
    config_path: InitVar[str]

    def __post_init__(self, config_path: str):
        data, self.experiment_name = self.__load_config(config_path)
        self.agent_type = data["agent_type"]
        self.common = data["common"]
        self.specific = data["specific"]
        if "state_normalizer" in data:
            self.state_normalizer = self.__parse_state_normalizer(
                data["state_normalizer"]
            )
        self.env_id = data["env_id"]
        self.env_wrappers = data["env_wrappers"]

        if "env_wrappers" not in data:
            data["env_wrappers"] = None

        env = apply_wrappers(gym.make(data["env_id"]), data["env_wrappers"])
        if "memory" in data:
            self.memory = ReplayMemory(
                **data["memory"],
                observation_space=env.observation_space,
                action_space=env.action_space
            )
        env.close()
        time.sleep(1)

        if "exploration_strategy" in data:
            self.exploration_strategy = self.__parse_exploration_strategy(
                data["exploration_strategy"]
            )
        if any([key == "layers" for key in data["body"].keys()]):
            self.body = Body(BodyParameters(data["body"]))
        else:
            self.body = {
                k: Body(BodyParameters(data["body"][k])) for k in data["body"].keys()
            }

    @staticmethod
    def __load_config(config_path: str) -> Tuple[t.TConfig, str]:
        with open(config_path, "r") as config_file:
            data = yaml.safe_load(config_file)
        keys = list(data.keys())
        assert len(keys) == 1
        return data[keys[0]], keys[0]

    def __parse_exploration_strategy(
        self, data: t.TConfig
    ) -> Union[sch.BaseScheduler, noise.GaussianNoise]:
        action_type = data["action_type"]
        if action_type == "discrete":
            return self.__create_scheduler_policy(data)
        elif action_type == "continuous":
            return self.__create_noise_policy(data)
        else:
            raise ValueError("Invalid action type! Should be discrete or continuous")

    @staticmethod
    def __create_scheduler_policy(data: t.TConfig) -> sch.BaseScheduler:
        if data["type"] == "linear":
            return sch.LinearScheduler(**data["parameters"])
        elif data["type"] == "exponential":
            return sch.ExponentialScheduler(**data["parameters"])
        else:
            ValueError("Invalid epsilon schedule type!")

    def __create_noise_policy(self, data: t.TConfig) -> noise.GaussianNoise:
        if data["type"] == "gaussian":
            return noise.GaussianNoise(
                dim=self.env.action_space.shape[0], **data["parameters"]
            )
        elif data["type"] == "ou":
            return noise.OUNoise(
                dim=self.env.action_space.shape[0], **data["parameters"]
            )
        else:
            ValueError()

    @staticmethod
    def __parse_state_normalizer(data: t.TConfig) -> norm.DummyNormalizer:
        norm_type = data["type"]
        if norm_type == "rms":
            return norm.RMSNormalizer(**data["parameters"])
        elif norm_type == "scale":
            return norm.ScaleNormalizer(**data["parameters"])
        elif norm_type == "image":
            return norm.ImageNormalizer()
        else:
            return norm.DummyNormalizer()
