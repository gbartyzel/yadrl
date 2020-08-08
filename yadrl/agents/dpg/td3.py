import torch

import yadrl.common.utils as utils
from yadrl.agents.dpg.ddpg import DDPG
from yadrl.common.exploration_noise import GaussianNoise
from yadrl.common.memory import Batch
from yadrl.networks.value_heads import DoubleValueHead


class TD3(DDPG):
    def __init__(self,
                 target_noise_limit: float = 0.5,
                 target_noise_std: float = 0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self._target_noise_limit = (-target_noise_limit, target_noise_limit)
        self._target_noise = GaussianNoise(
            self._action_dim, sigma=target_noise_std)

    def _initialize_critic_networks(self, phi):
        self._qv = DoubleValueHead(phi=phi).to(self._device)
        self._target_qv = DoubleValueHead(phi=phi).to(self._device)
        self._target_qv.load_state_dict(self._qv.state_dict())
        self._target_qv.eval()

    def _q_value(self,
                 state: torch.Tensor,
                 action: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        return self._qv.eval_head_1((state, action))

    def _compute_loss(self, batch: Batch) -> torch.Tensor:
        state = self._state_normalizer(batch.state)
        next_state = self._state_normalizer(batch.next_state)

        with torch.no_grad():
            noise = self._target_noise().clamp(
                *self._target_noise_limit).to(self._device)
            next_action = self._target_pi(next_state) + noise
            next_action = next_action.clamp(*self._action_limit)

            target_next_qs = self._target_qv((next_state, next_action))
            target_next_q = torch.min(*target_next_qs).view(-1, 1)
        target_q = utils.td_target(
            reward=batch.reward,
            mask=batch.mask,
            target=target_next_q,
            discount=batch.discount_factor * self._discount)
        expected_q1, expected_q2 = self._qv((state, batch.action))

        loss = utils.mse_loss(expected_q1, target_q) + \
               utils.mse_loss(expected_q2, target_q)
        return loss
