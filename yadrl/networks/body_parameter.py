from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn
import yaml


@dataclass
class LinearParameters:
    output: int
    activation: nn.Module
    noise: str = 'none'
    noise_init: float = 0.5
    activation: nn.Module = nn.ReLU()
    dropout: float = 0.0
    normalization: str = 'none'


@dataclass
class Conv2dParameters:
    output: int
    activation: nn.Module
    kernel: int
    stride: int = 1
    padding: int = 0
    num_group: int = 6
    dropout: float = 0.0
    normalization: str = 'none'


@dataclass
class InputParameters:
    state: int
    action: Optional[int] = None


@dataclass
class VisionOptions:
    output_dim: int
    flatten: bool = True


class BodyParameters:
    _layer_parameters = {
        'linear': LinearParameters,
        'vision': Conv2dParameters
    }

    _act_fn = {
        'relu': nn.ReLU(),
        'elu': nn.ELU(),
        'tanh': nn.Tanh(),
        'gelu': nn.GELU(),
        'sigmoid': nn.Sigmoid(),
        'selu': nn.SELU(),
        'none': nn.Identity(),
    }

    def __init__(self, parameters):
        if isinstance(parameters, str):
            with open(parameters, 'r') as yaml_file:
                self._parameters = yaml.safe_load(yaml_file)
        elif isinstance(parameters, dict):
            self._parameters = parameters
        else:
            ValueError('Wrong parameters!')

    @property
    def type(self):
        return self._parameters['type']

    @property
    def input(self):
        return InputParameters(**self._parameters['input'])

    @property
    def action_layer(self):
        if 'action_layer' in self._parameters:
            return self._parameters['action_layer']
        return None

    @property
    def vision_option(self):
        if 'vision_options' in self._parameters:
            return VisionOptions(**self._parameters['vision_options'])
        return None

    @property
    def layers(self):
        layers = []
        for layer in self._parameters['layers']:
            temp = deepcopy(layer)
            temp['activation'] = self._act_fn[temp['activation']]
            layers.append(self._layer_parameters[self.type](**temp))
        return layers
