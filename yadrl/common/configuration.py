from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, Tuple

import yaml

import yadrl.common.normalizer as norm
from yadrl.common.memory import ReplayMemory
from yadrl.networks.body_parameter import BodyParameters


@dataclass
class Configuration:
    experiment_name: str = field(init=False)
    type: str = field(init=False)
    common: Dict[str, Any] = field(init=False)
    specific: Dict[str, Any] = field(init=False)
    state_normalizer: Any = field(init=False)
    exploration_strategy: Any = field(init=False)
    memory: Any = field(init=False)
    body: BodyParameters = field(init=False)
    config_path: InitVar[str]

    def __post_init__(self, config_path: str):
        data, self.experiment_name = self._load_config(config_path)
        self.type = data['type']
        self.common = data['common']
        self.specific = data['specific']
        self.memory = ReplayMemory(**data['memory'])
        self.state_normalizer = self._parse_state_normalizer(data)
        self.body = BodyParameters(data['body'])

    def _load_config(self, config_path: str) -> Tuple[Dict[str, Any], str]:
        with open(config_path, 'r') as config_file:
            data = yaml.safe_load(config_file)
        keys = list(data.keys())
        assert len(keys) == 1
        return data[keys[0]], keys[0]

    def _parse_exploration_strategy(self, data):
        pass

    def _parse_state_normalizer(self, data) -> norm.DummyNormalizer:
        norm_type = data['type']
        if norm_type == 'rms':
            return norm.RMSNormalizer(**data['parameters'])
        elif norm_type == 'scale':
            return norm.ScaleNormalizer(**data['parameters'])
        elif norm_type == 'image':
            return norm.ImageNormalizer()
        else:
            return norm.DummyNormalizer()


if __name__ == '__main__':
    params = Configuration(config_path='../../experiments/cartpole_dqn.yaml')
    print(params.body)
