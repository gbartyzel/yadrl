from copy import deepcopy

import torch

from yadrl.common.replay_memory import ReplayMemory
from yadrl.common.heads import DeterministicPolicyHead, ValueHead


class DDPG(object):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 phi: torch.nn.Module,
                 actor_lrate: float,
                 critic_lrate: float,
                 l2_reg_value: float,
                 discount_factor: float,
                 polyak_factor: float,
                 memory_capacity: int,
                 batch_size: int,
                 warm_up_steps: int,
                 noise_annealing_steps: int):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._discount = discount_factor
        self._polyak = polyak_factor

        self._output_dim = action_dim

        self._step = 0.0
        self._batch_size = batch_size
        self._warm_up_steps = warm_up_steps

        self._actor = DeterministicPolicyHead(phi, action_dim, True).to(self._device)
        self._actor_optim = torch.optim.Adam(self._actor.parameters(), lr=actor_lrate)
        self._target_actor = deepcopy(self._actor).to(self._device)
        self._target_actor.eval()

        self._critic = ValueHead(phi, action_dim, True, True)
        self._critic_optim = torch.optim.Adam(self._critic.parameters(), lr=critic_lrate,
                                              weight_decay=l2_reg_value)
        self._target_critic = deepcopy(self._critic).to(self._device)
        self._target_critic.eval()

        self._memory = ReplayMemory(memory_capacity, state_dim, action_dim, True)

    def act(self):
        pass
