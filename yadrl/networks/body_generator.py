import abc
from typing import Any, Dict, Union

import torch
import torch.nn as nn

from yadrl.networks.body_parameter import BodyParameters
from yadrl.networks.commons import (get_layer, get_normalization,
                                    is_noisy_layer,
                                    orthogonal_init)


class Body(nn.Module, abc.ABC):
    implemented_observations = {}

    def __init_subclass__(cls, body_type: str, **kwargs):
        super().__init_subclass__(**kwargs)
        Body.implemented_observations[body_type] = cls

    def __init__(self, parameters: BodyParameters):
        super().__init__()
        self._body_parameters = parameters
        self._body = self._build_network()

    def forward(self, input: torch.Tensor, *args) -> torch.Tensor:
        for layer in self._body:
            input = layer(input)
        return input

    @classmethod
    def build(cls, parameters: Union[str, Dict[str, Any]]) -> nn.Module:
        params = BodyParameters(parameters)
        return cls.implemented_observations[params.type](params)

    def sample_noise(self):
        for layer in self._body:
            if is_noisy_layer(layer[0]):
                layer[0].sample_noise()

    def reset_noise(self):
        for layer in self._body:
            if is_noisy_layer(layer[0]):
                layer[0].reset_noise()

    def _reset_parameters(self):
        pass

    @abc.abstractmethod
    def _build_network(self) -> nn.Module:
        pass

    @property
    def output_dim(self) -> int:
        return list(self._body.parameters())[-1].shape[0]


class LinearBody(Body, body_type='linear'):
    def _build_network(self) -> nn.Module:
        body = nn.ModuleList()
        input_size = self._body_parameters.input.state
        for i, params in enumerate(self._body_parameters.layers):
            inner = nn.Sequential()
            if self._body_parameters.action_layer == i:
                input_size += self._body_parameters.input.action
            layer = get_layer(params.noise, input_size,
                              params.output, params.noise_init)
            inner.add_module('Linear', layer)

            if params.dropout > 0.0:
                inner.add_module('Dropout', nn.Dropout(p=params.dropout))

            if params.normalization != 'none':
                inner.add_module('Normalization',
                                 get_normalization(params.normalization,
                                                   params.output))

            inner.add_module('Activation', params.activation)
            input_size = params.output
            body.add_module('Layer_{}'.format(i), inner)
        return body

    def forward(self,
                x_state: torch.Tensor,
                x_action: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self._body):
            if i == self._body_parameters.action_layer:
                x_state = torch.cat((x_state, x_action), dim=1)
            x_state = layer(x_state)
        return x_state

    def _reset_parameters(self):
        for layer in self._body:
            if not self._is_noisy_layer(layer[0]):
                orthogonal_init(layer[0])


class VisionBody(Body, body_type='vision'):
    def _build_network(self) -> nn.Module:
        body = nn.ModuleList()
        input_size = self._body_parameters.input.state
        for i, params in enumerate(self._body_parameters.layers):
            inner = nn.Sequential()
            inner.add_module('Conv2d', nn.Conv2d(
                in_channels=input_size, out_channels=params.output,
                kernel_size=params.kernel, stride=params.stride,
                padding=params.padding))

            if params.dropout > 0.0:
                inner.add_module('Dropout', nn.Dropout(p=params.dropout))

            if params.normalization != 'none':
                inner.add_module('Normalization',
                                 get_normalization(params.normalization,
                                                   params.output,
                                                   params.num_group))

            inner.add_module('Activation', params.activation)
            input_size = params.output
            body.add_module('Layer_{}'.format(i), inner)

        if self._body_parameters.vision_option.flatten:
            body.add_module('Flatten', nn.Flatten())
        return body

    @property
    def output_dim(self) -> int:
        return self._body_parameters.vision_option.output_dim
