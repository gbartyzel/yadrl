from typing import Any, Dict, Optional

import torch as th
import torch.nn as nn

from yadrl.networks.body_parameter import BodyParameters
from yadrl.networks.layer import Layer


class Body(nn.Module):
    def __init__(self, parameters: BodyParameters):
        super().__init__()
        self._body_parameters = parameters
        self._body = self._build_network()

        if self._body_parameters.output_dim:
            self.output_dim = self._body_parameters.output_dim
        else:
            self.output_dim = self._body[-1].out_dim

    def forward(self, x_1: th.Tensor, x_2: Optional[th.Tensor] = None) -> th.Tensor:
        for i, layer in enumerate(self._body):
            if i == self._body_parameters.action_layer:
                x_1 = th.cat((x_1, x_2), dim=1)
            x_1 = layer(x_1)
        return x_1

    def sample_noise(self):
        for layer in self._body:
            layer.sample_noise()

    def reset_noise(self):
        for layer in self._body:
            layer.reset_noise()

    @staticmethod
    def from_dict(parameters: Dict[str, Any]) -> "Body":
        return Body(BodyParameters(parameters))

    def _build_network(self) -> nn.ModuleList:
        body = nn.ModuleList()
        input_size = self._body_parameters.input.primary
        for i, params in enumerate(self._body_parameters.layers):
            if self._body_parameters.action_layer == i:
                input_size += self._body_parameters.input.secondary
            if params["layer_type"] == "flatten":
                layer = nn.Flatten()
            else:
                layer = Layer.build(input_size, **params)
            input_size = params["out_dim"]
            body.append(layer)
        return body
