from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, Tuple, Union

import gym
import yaml

import yadrl.common.exploration_noise as noise
import yadrl.common.normalizer as norm
import yadrl.common.scheduler as sch
from yadrl.common.memory import ReplayMemory
from yadrl.networks.body import Body
from yadrl.networks.body_parameter import BodyParameters

T_CONFIG = Dict[str, Any]


@dataclass
class Configuration:
    env: gym.Env = field(init=False)
    experiment_name: str = field(init=False)
    agent_type: str = field(init=False)
    common: T_CONFIG = field(init=False)
    specific: T_CONFIG = field(init=False)
    state_normalizer: Any = field(init=False)
    exploration_strategy: Any = field(init=False, default=None)
    memory: Any = field(init=False, default=None)
    body: Body = field(init=False)
    config_path: InitVar[str]

    def __post_init__(self, config_path: str):
        data, self.experiment_name = self.__load_config(config_path)
        self.agent_type = data['agent_type']
        self.common = data['common']
        self.specific = data['specific']
        if 'memory' in data:
            self.memory = ReplayMemory(**data['memory'])
        self.state_normalizer = self.__parse_state_normalizer(
            data['state_normalizer'])
        if 'exploration_strategy' in data:
            self.exploration_strategy = self.__parse_exploration_strategy(
                data['exploration_strategy'])
        self.body = Body(BodyParameters(data['body']))
        self.env = gym.make(data['env_id'])

    @staticmethod
    def __load_config(config_path: str) -> Tuple[T_CONFIG, str]:
        with open(config_path, 'r') as config_file:
            data = yaml.safe_load(config_file)
        keys = list(data.keys())
        assert len(keys) == 1
        return data[keys[0]], keys[0]

    def __parse_exploration_strategy(self, data: T_CONFIG) -> Union[
        sch.BaseScheduler, noise.GaussianNoise]:
        action_type = data['action_type']
        if action_type == 'discrete':
            return self.__create_scheduler_policy(data)
        elif action_type == 'continuous':
            return self.__create_noise_policy(data)
        else:
            raise ValueError(
                'Invalid action type! Should be discrete or continuous')

    @staticmethod
    def __create_scheduler_policy(data: T_CONFIG) -> sch.BaseScheduler:
        if data['type'] == 'linear':
            return sch.LinearScheduler(**data['parameters'])
        elif data['type'] == 'exponential':
            return sch.ExponentialScheduler(**data['parameters'])
        else:
            ValueError('Invalid epsilon schedule type!')

    @staticmethod
    def __create_noise_policy(data: T_CONFIG) -> noise.GaussianNoise:
        if data['type'] == 'gaussian':
            return noise.GaussianNoise(**data['parameters'])
        elif data['type'] == 'ou':
            return noise.OUNoise(**data['parameters'])
        else:
            ValueError()

    @staticmethod
    def __parse_state_normalizer(data: T_CONFIG) -> norm.DummyNormalizer:
        norm_type = data['type']
        if norm_type == 'rms':
            return norm.RMSNormalizer(**data['parameters'])
        elif norm_type == 'scale':
            return norm.ScaleNormalizer(**data['parameters'])
        elif norm_type == 'image':
            return norm.ImageNormalizer()
        else:
            return norm.DummyNormalizer()
