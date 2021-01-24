from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th

TData = Union[np.ndarray, th.Tensor]
TTransition = Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, Optional[float]]
TNamedParameters = Iterator[Tuple[str, th.Tensor]]
TModuleDict = Dict[str, th.nn.Module]
TActionOption = Union[np.ndarray, int]
TConfig = Dict[str, Any]
TEnvReturn = Tuple[np.ndarray, float, bool, dict]
TWrappersOption = Sequence[Dict[str, Any]]
