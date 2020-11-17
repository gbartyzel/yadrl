from copy import deepcopy

import torch

import yadrl.common.utils as utils
from yadrl.agents.dpg.ddpg import DDPG
from yadrl.common.exploration_noise import GaussianNoise
from yadrl.common.memory import Batch
from yadrl.networks.head import Head


class TD3(DDPG, agent_type='td3'):
    def __init__(self,
                 target_noise_limit: float = 0.5,
                 target_noise_std: float = 0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self._target_noise_limit = (-target_noise_limit, target_noise_limit)
        self._target_noise = GaussianNoise(
            self._action_dim, sigma=target_noise_std)

    def _initialize_networks(self, phi):
        networks = super()._initialize_networks(phi)
        critic_net = Head.build(head_type='multi', body=phi['critic'],
                                output_dim=1, num_heads=2)
        target_critic_net = deepcopy(critic_net)
        critic_net.to(self._device)
        target_critic_net.to(self._device)
        target_critic_net.eval()
        networks['critic'] = critic_net
        networks['target_critic'] = target_critic_net

        return networks

    def _sample_q(self, state: torch.Tensor,
                  action: torch.Tensor) -> torch.Tensor:
        return self.qv(state, action)[0]

    def _compute_loss(self, batch: Batch) -> torch.Tensor:
        state = self._state_normalizer(batch.state)
        next_state = self._state_normalizer(batch.next_state)

        with torch.no_grad():
            noise = self._target_noise().clamp(
                *self._target_noise_limit).to(self._device)
            next_action = self.target_pi(next_state) + noise
            next_action = next_action.clamp(*self._action_limit)

            target_next_qs = self.target_qv(next_state, next_action)
            target_next_q = torch.min(*target_next_qs).view(-1, 1)
        target_q = utils.td_target(
            reward=batch.reward,
            mask=batch.mask,
            target=target_next_q,
            discount=batch.discount_factor * self._discount)
        expected_q1, expected_q2 = self.qv(state, batch.action)

        loss = utils.mse_loss(expected_q1, target_q) + \
               utils.mse_loss(expected_q2, target_q)
        return loss
