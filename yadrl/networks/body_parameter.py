from dataclasses import InitVar, dataclass, field
from typing import List, Optional, Union

import yaml


@dataclass
class InputParameters:
    primary: int
    secondary: Optional[int] = None


@dataclass
class BodyParameters:
    parameters: InitVar[Union[str, dict]]
    input: InputParameters = field(init=False)
    action_layer: int = field(init=False, default=None)
    layers: List[dict] = field(init=False)
    output_dim: int = field(init=False, default=None)

    def __post_init__(self, parameters: Union[str, dict]):
        if not isinstance(parameters, (str, dict)):
            raise TypeError()
        if isinstance(parameters, str):
            with open(parameters, "r") as yaml_file:
                parameters: dict = yaml.safe_load(yaml_file)

        self.input = InputParameters(**parameters["input"])
        if "output_dim" in parameters:
            self.output_dim = parameters["output_dim"]

        if "action_layer" in parameters:
            self.action_layer = parameters["action_layer"]

        self.layers = [layer for layer in parameters["layers"]]
