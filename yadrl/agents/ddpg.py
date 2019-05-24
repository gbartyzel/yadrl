from copy import deepcopy
from typing import Any, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from yadrl.agents.base import BaseOffPolicy
from yadrl.common.exploration_noise import GaussianNoise
from yadrl.common.heads import DeterministicPolicyHead, ValueHead


class DDPG(BaseOffPolicy):
    def __init__(self,
                 phi: nn.Module,
                 noise: GaussianNoise,
                 actor_lrate: float,
                 critic_lrate: float,
                 l2_reg_value: float,
                 **kwargs):
        super(DDPG, self).__init__(**kwargs)

        self._actor = DeterministicPolicyHead(phi, self._action_dim, True).to(self._device)
        self._actor_optim = optim.Adam(self._actor.parameters(), lr=actor_lrate)
        self._target_actor = deepcopy(self._actor).to(self._device)
        self._target_actor.eval()

        self._critic = ValueHead(phi, True, True)
        self._critic_optim = optim.Adam(self._critic.parameters(), lr=critic_lrate,
                                        weight_decay=l2_reg_value)
        self._target_critic = deepcopy(self._critic).to(self._device)
        self._target_critic.eval()

        self._noise = noise

    def act(self, state: np.ndarray, train: bool = False) -> np.ndarray:
        state = torch.from_numpy(state).float().to(self._device)
        self._actor.eval()
        with torch.no_grad():
            action = self._actor(state)

        if train:
            return torch.clamp(action + self._noise(), -1.0, 1.0)
        return action




