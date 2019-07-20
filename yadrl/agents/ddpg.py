from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
import torch

import yadrl.common.exploration_noise as noise
from yadrl.agents.base import BaseOffPolicy
from yadrl.common.replay_memory import Batch
from yadrl.common.utils import mse_loss
from yadrl.networks import Critic
from yadrl.networks import DeterministicActor


class DDPG(BaseOffPolicy):
    def __init__(self,
                 pi_phi: torch.nn.Module,
                 qv_phi: torch.nn.Module,
                 pi_lrate: float,
                 qv_lrate: float,
                 l2_reg_value: float,
                 action_limit: Union[Sequence[float], np.ndarray],
                 noise_type: Optional[str] = "ou",
                 mean: Optional[float] = 0.0,
                 sigma: Optional[float] = 0.2,
                 sigma_min: Optional[float] = 0.0,
                 n_step_annealing: Optional[float] = 1e6,
                 theta: Optional[float] = 0.15,
                 dt: Optional[float] = 0.01,
                 **kwargs):
        super(DDPG, self).__init__(agent_type='ddpg', **kwargs)
        if np.shape(action_limit) != (2,):
            raise ValueError
        self._action_limit = action_limit

        self._pi = DeterministicActor(pi_phi, self._action_dim, True).to(
            self._device)
        self._pi_optim = torch.optim.Adam(self._pi.parameters(), lr=pi_lrate)
        self._target_pi = DeterministicActor(
            pi_phi, self._action_dim, True).to(self._device)

        self._qv = Critic(qv_phi, True)
        self._qv_optim = torch.optim.Adam(
            self._qv.parameters(), qv_lrate, weight_decay=l2_reg_value)
        self._target_qv = Critic(qv_phi, True).to(self._device)

        self.load()
        self._target_pi.load_state_dict(self._pi.state_dict())
        self._target_qv.load_state_dict(self._qv.state_dict())

        self._noise = self._get_noise(noise_type, mean, sigma, sigma_min,
                                      theta, n_step_annealing, dt)

    def act(self, state: np.ndarray, train: bool = False) -> np.ndarray:
        state = torch.from_numpy(state).float().to(self._device)
        self._pi.eval()
        with torch.no_grad():
            action = self._pi(state)
        self._pi.eval()

        if train:
            action = torch.clamp(action + self._noise(), *self._action_limit)
        return action.cpu().numpy()

    def reset(self):
        self._noise.reset()

    def update(self):
        batch = self._memory(self._batch_size, self._device)
        self._update_critic(batch)
        self._update_actor(batch)

        self._soft_update(self._pi.parameters(), self._target_pi.parameters())
        self._soft_update(self._qv.parameters(), self._target_qv.parameters())

    def _update_critic(self, batch: Batch):
        mask = 1.0 - batch.done
        next_action = self._target_pi(batch.next_state)
        target_next_q = self._target_qv(
            batch.next_state, next_action).view(-1, 1).detach()

        target_q = self._td_target(batch.reward, mask, target_next_q)
        expected_q = self._qv(batch.state, batch.action)

        loss = mse_loss(expected_q, target_q)
        self._qv_optim.zero_grad()
        loss.backward()
        self._qv_optim.step()

    def _update_actor(self, batch: Batch):
        loss = -self._qv(batch.state, self._pi(batch.state))
        self._pi_optim.zero_grad()
        loss.backward()
        self._pi_optim.step()

    def load(self) -> NoReturn:
        model = self._checkpoint_manager.load()
        if model:
            self._pi.load_state_dict(model['actor'])
            self._qv.load_state_dict(model['critic'])

    def save(self):
        state_dicts = {
            'actor': self._pi.state_dict(),
            'critic': self._qv.state_dict()
        }
        self._checkpoint_manager.save(state_dicts, self.step)

    def _get_noise(self, noise_type: str,
                   mean: float,
                   sigma: float,
                   sigma_min: float,
                   theta: float,
                   n_step_annealing: float,
                   dt: float) -> noise.GaussianNoise:

        if noise_type == "normal":
            return noise.GaussianNoise(
                self._action_dim, mean=mean, sigma=sigma)
        elif noise_type == "adaptive":
            return noise.AdaptiveGaussianNoise(
                self._action_dim, mean=mean, sigma=sigma, sigma_min=sigma_min,
                n_step_annealing=n_step_annealing)
        elif noise_type == "ou":
            return noise.OUNoise(
                self._action_dim, mean=mean, theta=theta, sigma=sigma,
                sigma_min=sigma_min, n_step_annealing=n_step_annealing, dt=dt)
        else:
            raise ValueError
