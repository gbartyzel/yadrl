import copy
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Union
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import yadrl.common.exploration_noise as noise
from yadrl.agents.base import BaseOffPolicy
from yadrl.common.memory import Batch
from yadrl.networks.models import Critic, DoubleCritic
from yadrl.networks.models import DeterministicActor
import yadrl.common.utils as utils


class DDPG(BaseOffPolicy):
    def __init__(self,
                 pi_phi: nn.Module,
                 qv_phi: nn.Module,
                 pi_lrate: float,
                 qv_lrate: float,
                 action_limit: Union[Sequence[float], np.ndarray],
                 l2_reg_value: float = 0.0,
                 pi_grad_norm_value: float = 0.0,
                 qv_grad_norm_value: float = 0.0,
                 twin_critic: bool = False,
                 distribution_type: str = 'quantile',
                 support_dim: int = 100,
                 v_limit: Tuple[float, float] = (-200.0, 200.0),
                 noise_type: Optional[str] = "ou",
                 mean: Optional[float] = 0.0,
                 sigma: Optional[float] = 0.2,
                 sigma_min: Optional[float] = 0.01,
                 n_step_annealing: Optional[float] = 1e5,
                 theta: Optional[float] = 0.15,
                 dt: Optional[float] = 0.01,
                 **kwargs):
        super(DDPG, self).__init__(**kwargs)
        if np.shape(action_limit) != (2,):
            raise ValueError
        self._action_limit = action_limit
        self._distribution_type = distribution_type
        self._pi_grad_norm_value = pi_grad_norm_value
        self._qv_grad_norm_value = qv_grad_norm_value

        self._pi = DeterministicActor(
            phi=pi_phi,
            output_dim=self._action_dim,
            fan_init=True).to(self._device)
        self._pi_optim = optim.Adam(self._pi.parameters(), lr=pi_lrate)
        self._target_pi = copy.deepcopy(self._pi).to(self._device)

        critic = DoubleCritic if twin_critic else Critic
        self._qv = critic(
            phi=qv_phi,
            distribution_type=distribution_type,
            support_dim=support_dim).to(self._device)
        if distribution_type == 'categorical':
            self._v_limit = v_limit
            self._atoms = torch.linspace(v_limit[0], v_limit[1], support_dim,
                                         device=self._device).unsqueeze(0)
        if distribution_type == 'quantile':
            self._cumulative_density = torch.from_numpy(
                (np.arange(support_dim) + 0.5) / support_dim
            ).float().unsqueeze(0).to(self._device)

        self._qv_optim = optim.Adam(
            self._qv.parameters(), qv_lrate, weight_decay=l2_reg_value,
            eps=1.0 / 32.0)
        self._target_qv = copy.deepcopy(self._qv).to(self._device)

        self._target_pi.load_state_dict(self._pi.state_dict())
        self._target_qv.load_state_dict(self._qv.state_dict())

        self._noise = self._get_noise(noise_type, mean, sigma, sigma_min,
                                      theta, n_step_annealing, dt)

    def _act(self, state: np.ndarray, train: bool = False) -> np.ndarray:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        state = self._state_normalizer(state, self._device)
        self._pi.eval()
        with torch.no_grad():
            action = self._pi(state)
        self._pi.train()

        if train:
            noise = self._noise().to(self._device)
            action = torch.clamp(action + noise, *self._action_limit)
        return action[0].cpu().numpy()

    def reset(self):
        self._noise.reset()

    def _update(self):
        batch = self._memory.sample(self._batch_size, self._device)
        self._update_critic(batch)
        self._update_actor(batch)

        self._update_target(self._pi, self._target_pi)
        self._update_target(self._qv, self._target_qv)

    def _update_critic(self, batch: Batch):
        next_state = self._state_normalizer(batch.next_state, self._device)
        with torch.no_grad():
            next_action = self._target_pi(next_state)
            next_critic_output = self._target_qv(next_state, next_action)

        if self._distribution_type == 'categorical':
            loss = self._compute_categorical_loss(batch, next_critic_output)
        elif self._distribution_type == 'quantile':
            loss = self._compute_quantile_loss(batch, next_critic_output)
        else:
            loss = self._compute_td_loss(batch, next_critic_output)
        self._qv_optim.zero_grad()
        loss.backward()
        if self._qv_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._qv.parameters(),
                                     self._qv_grad_norm_value)
        self._qv_optim.step()
        self._writer.add_scalar('loss/qv', loss.item(), self._step)

    def _compute_td_loss(self,
                         batch: Batch,
                         target_next_q: torch.Tensor) -> torch.Tensor:
        state = self._state_normalizer(batch.state, self._device)
        target_q = utils.td_target(
            reward=batch.reward,
            mask=batch.mask,
            next_value=target_next_q.view(-1, 1),
            discount=self._discount).detach()
        expected_q = self._qv(state, batch.action)

        loss = utils.mse_loss(expected_q, target_q)
        return loss

    def _compute_categorical_loss(self,
                                  batch: Batch,
                                  next_probs: torch.Tensor) -> torch.Tensor:
        state = self._state_normalizer(batch.state, self._device)
        target_probs = utils.l2_projection(
            next_probs=next_probs,
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
        target_quantiles = utils.td_target(
            reward=batch.reward,
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
        self._writer.add_scalar('loss/pi', loss.item(), self._step)

    def load(self, path: str) -> NoReturn:
        model = torch.load(path)
        if model:
            self._pi.load_state_dict(model['actor'])
            self._qv.load_state_dict(model['critic'])
            self._target_qv.load_state_dict(model['target_critic'])
            self._step = model['step']
            if 'state_norm' in model:
                self._state_normalizer.load(model['state_norm'])

    def save(self):
        state_dict = dict()
        state_dict['actor'] = self._pi.state_dict(),
        state_dict['critic'] = self._qv.state_dict()
        state_dict['target_critic'] = self._target_qv.state_dict()
        state_dict['step'] = self._step
        if self._use_state_normalization:
            state_dict['state_norm'] = self._state_normalizer.state_dict()
        torch.save(state_dict, 'model_{}.pth'.format(self._step))

    def _get_noise(self, noise_type: str,
                   mean: float,
                   sigma: float,
                   sigma_min: float,
                   theta: float,
                   n_step_annealing: float,
                   dt: float) -> noise.GaussianNoise:

        if noise_type == "gaussian":
            return noise.GaussianNoise(
                self._action_dim, mean=mean, sigma=sigma, sigma_min=sigma_min,
                n_step_annealing=n_step_annealing)
        elif noise_type == "ou":
            return noise.OUNoise(
                self._action_dim, mean=mean, theta=theta, sigma=sigma,
                sigma_min=sigma_min, n_step_annealing=n_step_annealing, dt=dt)
        else:
            raise ValueError

    @property
    def parameters(self):
        return list(self._qv.named_parameters()) + \
               list(self._pi.named_parameters())

    @property
    def target_parameters(self):
        return list(self._target_qv.named_parameters()) + \
               list(self._target_pi.named_parameters())
