from typing import NoReturn
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import yadrl.common.utils as utils
from yadrl.agents.base import BaseOffPolicy
from yadrl.common.exploration_noise import GaussianNoise
from yadrl.common.memory import Batch
from yadrl.networks.heads import DeterministicPolicyHead
from yadrl.networks.heads import ValueHead


class DDPG(BaseOffPolicy):
    def __init__(self,
                 pi_phi: nn.Module,
                 qv_phi: nn.Module,
                 pi_lrate: float,
                 qv_lrate: float,
                 action_limit: Union[Sequence[float], np.ndarray],
                 noise: GaussianNoise,
                 noise_scale_factor: float = 1.0,
                 l2_reg_value: float = 0.0,
                 pi_grad_norm_value: float = 0.0,
                 qv_grad_norm_value: float = 0.0,
                 distribution_type: str = 'quantile',
                 support_dim: int = 100,
                 v_limit: Tuple[float, float] = (-200.0, 200.0),
                 **kwargs):
        super(DDPG, self).__init__(**kwargs)
        assert np.shape(action_limit) == (2,), "Wrong action limit!"

        self._action_limit = action_limit
        self._distribution_type = distribution_type
        self._pi_grad_norm_value = pi_grad_norm_value
        self._qv_grad_norm_value = qv_grad_norm_value

        self._initialize_networks(pi_phi, qv_phi, support_dim, v_limit)
        self._pi_optim = optim.Adam(self._pi.parameters(), lr=pi_lrate)
        self._qv_optim = optim.Adam(
            self._qv.parameters(), qv_lrate, weight_decay=l2_reg_value)

        self._noise = noise
        self._noise_scale_factor = noise_scale_factor
        self._critic_loss_fn = {'none': self._compute_td_loss,
                                'categorical': self._compute_categorical_loss,
                                'quantile': self._compute_quantile_loss}

    def _initialize_networks(self,
                             pi_phi: nn.Module,
                             qv_phi: nn.Module,
                             support_dim: int,
                             v_limit: Tuple[float, float]):
        self._pi = DeterministicPolicyHead(phi=pi_phi,
                                           output_dim=self._action_dim,
                                           fan_init=True).to(self._device)
        self._target_pi = DeterministicPolicyHead(phi=pi_phi,
                                                  output_dim=self._action_dim,
                                                  fan_init=True).to(
            self._device)

        self._qv = ValueHead(phi=qv_phi,
                             distribution_type=self._distribution_type,
                             support_dim=support_dim).to(self._device)
        self._target_qv = ValueHead(phi=qv_phi,
                                    distribution_type=self._distribution_type,
                                    support_dim=support_dim).to(self._device)
        self._target_pi.load_state_dict(self._pi.state_dict())
        self._target_qv.load_state_dict(self._qv.state_dict())

        if self._distribution_type == 'categorical':
            self._v_limit = v_limit
            self._atoms = torch.linspace(v_limit[0], v_limit[1], support_dim,
                                         device=self._device).unsqueeze(0)
        if self._distribution_type == 'quantile':
            self._cumulative_density = torch.from_numpy(
                (np.arange(support_dim) + 0.5) / support_dim
            ).float().unsqueeze(0).to(self._device)

    def _act(self, state: np.ndarray, train: bool = False) -> np.ndarray:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        state = self._state_normalizer(state, self._device)
        self._pi.eval()
        with torch.no_grad():
            action = self._pi(state)
        self._pi.train()

        if train:
            noise = self._noise_scale_factor * self._noise().to(self._device)
            action = torch.clamp(action + noise, *self._action_limit)
        return action[0].cpu().numpy()

    def reset(self):
        self._noise.reset()

    def _update(self):
        batch = self._memory.sample(self._batch_size)
        self._update_critic(batch)
        self._update_actor(batch)
        self._update_target(self._pi, self._target_pi)
        self._update_target(self._qv, self._target_qv)

    def _update_critic(self, batch: Batch):
        next_state = self._state_normalizer(batch.next_state, self._device)
        next_action = self._target_pi(next_state)
        next_critic_output = self._target_qv(next_state, next_action).detach()

        loss = self._critic_loss_fn[self._distribution_type](batch,
                                                             next_critic_output)
        self._qv_optim.zero_grad()
        loss.backward()
        if self._qv_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._qv.parameters(),
                                     self._qv_grad_norm_value)
        self._qv_optim.step()
        self._writer.add_scalar('loss/qv', loss.item(), self._env_step)

    def _compute_td_loss(self,
                         batch: Batch,
                         target_next_q: torch.Tensor) -> torch.Tensor:
        state = self._state_normalizer(batch.state, self._device)
        target_q = utils.td_target(reward=batch.reward,
                                   mask=batch.mask,
                                   next_value=target_next_q.view(-1, 1),
                                   discount=self._discount).detach()
        expected_q = self._qv(state, batch.action)
        return utils.mse_loss(expected_q, target_q)

    def _compute_categorical_loss(self,
                                  batch: Batch,
                                  next_probs: torch.Tensor) -> torch.Tensor:
        state = self._state_normalizer(batch.state, self._device)
        target_probs = utils.l2_projection(next_probs=next_probs,
                                           reward=batch.reward,
                                           mask=batch.mask,
                                           atoms=self._atoms,
                                           v_limit=self._v_limit,
                                           discount=self._discount).detach()

        probs = torch.clamp(self._qv(state, batch.action), 1e-7, 1.0)
        loss = -(target_probs * probs.log()).sum(-1)
        return loss.mean()

    def _compute_quantile_loss(self,
                               batch: Batch,
                               next_quantiles: torch.Tensor) -> torch.Tensor:
        state = self._state_normalizer(batch.state, self._device)
        target_quantiles = utils.td_target(reward=batch.reward,
                                           mask=batch.mask,
                                           next_value=next_quantiles,
                                           discount=self._discount).detach()

        expected_quantiles = self._qv(state, batch.action)

        loss = utils.quantile_hubber_loss(
            prediction=expected_quantiles,
            target=target_quantiles,
            cumulative_density=self._cumulative_density)
        return loss

    def _update_actor(self, batch: Batch):
        state = self._state_normalizer(batch.state, self._device)
        q_value = self._qv(state, self._pi(state))

        if self._distribution_type == 'categorical':
            q_value = q_value.mul(self._atoms.expand_as(q_value)).sum(-1)
        if self._distribution_type == 'quantile':
            q_value = q_value.mean(-1)
        loss = torch.mean(-q_value)
        self._pi_optim.zero_grad()

        loss.backward()
        if self._pi_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._pi.parameters(),
                                     self._pi_grad_norm_value)
        self._pi_optim.step()
        self._writer.add_scalar('loss/pi', loss.item(), self._env_step)

    def load(self, path: str) -> NoReturn:
        model = torch.load(path)
        if model:
            self._pi.load_state_dict(model['actor'])
            self._qv.load_state_dict(model['critic'])
            self._target_qv.load_state_dict(model['target_critic'])
            self._env_step = model['step']
            if 'state_norm' in model:
                self._state_normalizer.load(model['state_norm'])

    def save(self):
        state_dict = dict()
        state_dict['actor'] = self._pi.state_dict(),
        state_dict['critic'] = self._qv.state_dict()
        state_dict['target_critic'] = self._target_qv.state_dict()
        state_dict['step'] = self._env_step
        if self._use_state_normalization:
            state_dict['state_norm'] = self._state_normalizer.state_dict()
        torch.save(state_dict, 'model_{}.pth'.format(self._env_step))

    @property
    def parameters(self):
        return list(self._qv.named_parameters()) + \
               list(self._pi.named_parameters())

    @property
    def target_parameters(self):
        return list(self._target_qv.named_parameters()) + \
               list(self._target_pi.named_parameters())
