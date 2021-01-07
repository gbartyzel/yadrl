import abc
from typing import Callable, Dict

import torch
import torch.nn as nn

import yadrl.common.ops as ops
import yadrl.networks.noisy_linear as nl

normalizations: Dict[str, Callable] = {
    "layer_norm": nn.LayerNorm,
    "batch_norm_1d": nn.BatchNorm1d,
    "instance_norm_1d": (lambda in_dim: nn.InstanceNorm1d(in_dim, affine=True)),
    "batch_norm_2d": nn.BatchNorm2d,
    "instance_norm_2d": (lambda in_dim: nn.InstanceNorm2d(in_dim, affine=True)),
    "group_norm": nn.GroupNorm,
}

activation_fn: Dict[str, nn.Module] = {
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "tanh": nn.Tanh(),
    "gelu": nn.GELU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "identity": nn.Identity(),
    "softmax": nn.Softmax(dim=-1),
    "log_softmax": nn.LogSoftmax(dim=-1),
    "none": None,
}


class Layer(nn.Module, abc.ABC):
    registered_layer: Dict[str, "Layer"] = {}

    def __init_subclass__(cls, layer_type: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registered_layer[layer_type] = cls

    @classmethod
    def build(cls, in_dim: int, **kwargs) -> "Layer":
        layer_type = kwargs["layer_type"]
        return cls.registered_layer[layer_type](in_dim=in_dim, **kwargs)

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layer_type: str,
        activation: str = "none",
        normalization: str = "none",
        dropout_prob: float = 0.0,
        num_group: int = 6,
        layer_init: Callable = None,
        **kwargs
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_type = layer_type

        self._module = self._make_module(
            activation, normalization, dropout_prob, num_group, **kwargs
        )
        if layer_init is not None:
            layer_init(self._module[0])

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self._module(input_data)

    def sample_noise(self):
        if "noisy" in self.layer_type:
            self._module[0].sample_noise()

    def reset_noise(self):
        if "noisy" in self.layer_type:
            self._module[0].reset_noise()

    def _make_module(
        self,
        activation: str,
        normalization: str,
        dropout_prob: float,
        num_group: int,
        **kwargs
    ) -> nn.Sequential:
        block = [
            self._get_layer(**kwargs),
            self._get_normalization(normalization, num_group),
            activation_fn[activation],
            self._get_dropout(dropout_prob),
        ]
        return nn.Sequential(*[module for module in block if module])

    def _get_normalization(self, norm_type: str, num_group: int) -> nn.Module:
        if norm_type != "none":
            norm_params = [self.out_dim]
            if norm_type == "group":
                norm_params = [num_group] + norm_params
            return normalizations[norm_type](*norm_params)
        return None

    def _get_dropout(self, dropout_prob: float) -> nn.Module:
        if dropout_prob > 0:
            return nn.Dropout(dropout_prob)
        return None

    @abc.abstractmethod
    def _get_layer(self, **kwargs) -> nn.Module:
        pass


class Linear(Layer, layer_type="linear"):
    def __init__(self, **kwargs):
        if "layer_init" not in kwargs:
            kwargs["layer_init"] = ops.orthogonal_init
        super().__init__(**kwargs)

    def _get_layer(self, **kwargs) -> nn.Module:
        return nn.Linear(self.in_dim, self.out_dim, **kwargs)


class FactorizedLinear(Layer, layer_type="factorized_noisy_linear"):
    def __init__(self, **kwargs):
        kwargs["layer_init"] = None
        super().__init__(**kwargs)

    def _get_layer(self, **kwargs) -> nn.Module:
        return nl.FactorizedNoisyLinear(self.in_dim, self.out_dim, **kwargs)


class IndependentLinear(Layer, layer_type="independent_noisy_linear"):
    def __init__(self, **kwargs):
        super().__init__(layer_init=None, **kwargs)

    def _get_layer(self, **kwargs) -> nn.Module:
        return nl.IndependentNoisyLinear(self.in_dim, self.out_dim, **kwargs)


class Conv1d(Layer, layer_type="conv2d"):
    def _get_layer(self, **kwargs) -> nn.Module:
        return nn.Conv1d(in_channels=self.in_dim, out_channels=self.out_dim, **kwargs)


class Conv2d(Layer, layer_type="conv2d"):
    def _get_layer(self, **kwargs) -> nn.Module:
        return nn.Conv2d(in_channels=self.in_dim, out_channels=self.out_dim, **kwargs)

    def _get_dropout(self, dropout_prob: float) -> nn.Module:
        if dropout_prob > 0:
            return nn.Dropout2d(dropout_prob)
        return None
