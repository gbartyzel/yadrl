import abc

import torch.nn as nn
import yaml

from yadrl.networks.noisy_linear import FactorizedNoisyLinear, \
    IndependentNoisyLinear


class Body(nn.Module, abc.ABC):
    implemented_observations = {}

    _ACT_FN = {
        'relu': nn.ReLU(),
        'elu': nn.ELU(),
        'tanh': nn.Tanh(),
        'gelu': nn.GELU(),
        'sigmoid': nn.Sigmoid(),
        'selu': nn.SELU(),
        'none': nn.Identity(),
    }

    def __init_subclass__(cls, body_type, **kwargs):
        super().__init_subclass__(**kwargs)
        Body.implemented_observations[body_type] = cls

    def __init__(self, parameters):
        super().__init__()
        self._parameters = parameters
        self._body = self._build_network()

    def forward(self, input):
        for layer in self._body:
            input = layer(input)
        return input

    @classmethod
    def build(cls, parameters):
        return cls.implemented_observations[parameters['type']](parameters)

    @abc.abstractmethod
    def _build_network(self):
        pass


class LinearBody(Body, body_type='linear'):
    def _build_network(self):
        body = nn.ModuleList()
        input_size = self._parameters['input_size']
        for i, params in enumerate(self._parameters['layers']):
            inner = nn.Sequential()
            layer = self.__get_layer(params['noise'], input_size,
                                     params['output'], params['noise_init'])
            inner.add_module('Linear', layer)

            if 'dropout' in params and params['dropout'] > 0.0:
                inner.add_module('Dropout', nn.Dropout(p=params['dropout']))

            if 'normalization' in params or 'none' in params['normalization']:
                inner.add_module('Normalization', self.__get_normalization(
                    params['normalization'], params['output']))

            inner.add_module('Activation', Body._ACT_FN[params['activation']])
            input_size = params['output']
            body.add_module('Layer_{}'.format(i), inner)

        return body

    @staticmethod
    def __get_normalization(normalization_type, feature_size):
        if normalization_type == 'layer_norm':
            return nn.LayerNorm(feature_size)
        elif normalization_type == 'batch_norm':
            return nn.BatchNorm1d(feature_size)
        else:
            raise ValueError

    @staticmethod
    def __get_layer(layer_type, input_size, feature_size, noise_init):
        if layer_type == 'none':
            return nn.Linear(input_size, feature_size)
        elif layer_type == 'factorized':
            return FactorizedNoisyLinear(input_size, feature_size, noise_init)
        elif layer_type == 'independent':
            return IndependentNoisyLinear(input_size, feature_size, noise_init)
        else:
            raise ValueError('Wrong noise type!')


class VisionBody(Body, body_type='vision'):
    def _build_network(self):
        body = nn.ModuleList()
        input_size = self._parameters['input_size']
        for i, layer in enumerate(self._parameters['layers']):
            inner = nn.Sequential()
            inner.add_module('Conv2d', nn.Conv2d(
                in_channels=input_size, out_channels=layer['output'],
                kernel_size=layer['kernel'], stride=layer['stride'],
                padding=layer['padding']))

            if 'dropout' in layer and layer['dropout'] > 0.0:
                inner.add_module('Dropout', nn.Dropout(p=layer['dropout']))

            if 'normalization' in layer or 'none' in layer['normalization']:
                inner.add_module('Normalization', self.__get_normalization(
                    layer['normalization'], layer['output']))

            inner.add_module('Activation', Body._ACT_FN[layer['activation']])
            input_size = layer['output']
            body.add_module('Layer_{}'.format(i), inner)

    @staticmethod
    def __get_normalization(normalization_type, feature_size, num_group):
        if normalization_type == 'instance_norm':
            return nn.InstanceNorm2d(feature_size, affine=True)
        elif normalization_type == 'batch_norm':
            return nn.BatchNorm2d(feature_size, affine=True)
        elif normalization_type == 'group_norm':
            return nn.GroupNorm(num_group, feature_size, affine=True)
        else:
            raise ValueError


if __name__ == '__main__':
    with open('./body_generator.yaml') as yaml_file:
        parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    body = Body.build(parameters['body'])
    print(body)
