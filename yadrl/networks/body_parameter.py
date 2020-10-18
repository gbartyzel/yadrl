from typing import Any, Dict, List, Optional, Union

import torch.nn as nn
import yaml
from dataclasses import InitVar, dataclass, field

_act_fn = {
    'relu': nn.ReLU(),
    'elu': nn.ELU(),
    'tanh': nn.Tanh(),
    'gelu': nn.GELU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'none': nn.Identity(),
}


@dataclass
class LayerParameters:
    output: int
    activation: nn.Module = field(init=False)
    activation_str: InitVar[str]
    dropout: float = 0.0
    normalization: str = 'none'

    def __post_init__(self, activation_str: str):
        self.activation = _act_fn[activation_str]


@dataclass
class LinearParameters(LayerParameters):
    noise: str = 'none'
    noise_init: float = 0.5


@dataclass
class _Kernel:
    kernel: int


@dataclass
class Conv2dParameters(LayerParameters, _Kernel):
    stride: int = 1
    padding: int = 0
    num_group: int = 6


@dataclass
class InputParameters:
    primary: int
    secondary: Optional[int] = None


@dataclass
class VisionOptions:
    output_dim: int
    flatten: bool = True


_layer_parameters = {
    'linear': LinearParameters,
    'vision': Conv2dParameters
}


@dataclass
class BodyParameters:
    """
    Args:
        parameters
    Attributes:
        type ():
        input ():
        action_layer ():
        vision_option ():
        layers ():
    """
    type: str = field(init=False)
    input: InputParameters = field(init=False)
    action_layer: int = field(init=False, default=None)
    vision_option: VisionOptions = field(init=False, default=None)
    layers: List[Union[LinearParameters, Conv2dParameters]] = field(init=False)
    parameters: InitVar[Dict[str, Any]]

    def __post_init__(self, parameters):
        if isinstance(parameters, str):
            with open(parameters, 'r') as yaml_file:
                parameters = yaml.safe_load(yaml_file)
        self.type = parameters['type']
        self.input = InputParameters(**parameters['input'])
        if 'vision_options' in parameters:
            self.action_layer = parameters['action_layer']
        if 'vision_options' in parameters:
            self.vision_option = parameters['vision_options']
        self.layers = [_layer_parameters[self.type](**layer)
                       for layer in parameters['layers']]
