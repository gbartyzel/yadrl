import random
from typing import NoReturn
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import yadrl.common.utils as utils
from yadrl.agents.base import BaseOffPolicy
from yadrl.common.scheduler import BaseScheduler
from yadrl.networks.heads import MultiDQNHead


class MaxminDQN(BaseOffPolicy):
    def __init__(self,
                 phi: nn.Module,
                 heads_num: int,
                 learning_rate: float,
                 grad_norm_value: float,
                 epsilon_scheduler: BaseScheduler = BaseScheduler(),
                 noise_type: str = 'none',
                 use_dueling: bool = False,
                 distribution_type: str = 'none',
                 v_limit: Tuple[float, float] = (-1.0, 1.0),
                 support_dim: int = 51, **kwargs):
        super(MaxminDQN, self).__init__(**kwargs)

        self._grad_norm_value = grad_norm_value
        self._use_noise = noise_type != 'none'
        self._heads_num = heads_num
        self._distribution_type = distribution_type

        self._epsilon_scheduler = epsilon_scheduler

        self._qv = MultiDQNHead(
            phi=phi,
            output_dim=self._action_dim,
            heads_num=heads_num,
            dueling=use_dueling,
            noise_type=noise_type,
            distribution_type=distribution_type,
            support_dim=support_dim).to(self._device)
        self._target_qv = MultiDQNHead(
            phi=phi,
            output_dim=self._action_dim,
            heads_num=heads_num,
            dueling=use_dueling,
            noise_type=noise_type,
            distribution_type=distribution_type,
            support_dim=support_dim).to(self._device)
        self._target_qv.load_state_dict(self._qv.state_dict())
        self._target_qv.eval()

        if distribution_type == 'categorical':
            self._v_limit = v_limit
            self._atoms = torch.linspace(v_limit[0], v_limit[1], support_dim,
                                         device=self._device).unsqueeze(0)
        self._optims = [
            optim.RMSprop(self._qv.parameters(item=i), lr=learning_rate)
            for i in range(heads_num)
        ]

    def _act(self, state: int, train: bool = False) -> np.ndarray:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        state = self._state_normalizer(state, self._device)

        self._qv.eval()
        with torch.no_grad():
            q_value = torch.cat(self._qv(state, train, True), 1)
            if self._distribution_type == 'categorical':
                q_value = q_value.mul(self._atoms.expand_as(q_value)).sum(-1)
            q_value = q_value.min(1)[0]
        self._qv.train()

        eps_flag = random.random() > self._epsilon_scheduler.step()
        if eps_flag or self._use_noise or not train:
            return q_value.argmax(-1)[0].cpu().numpy()
        return random.randint(0, self._action_dim - 1)

    def _update(self):
        batch = self._memory.sample(self._batch_size)

        idx = np.random.randint(0, self._heads_num)
        loss = self._compute_categorical_loss(batch, idx)

        self._writer.add_scalar('loss', loss, self._env_step)

        self._optims[idx].zero_grad()
        loss.backward()
        if self._grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self._qv.parameters(),
                                     self._grad_norm_value)
        self._optims[idx].step()

        self._update_target(self._qv, self._target_qv)

    def _compute_td_loss(self, batch, idx):
        state = self._state_normalizer(batch.state, self._device)
        next_state = self._state_normalizer(batch.next_state, self._device)

        target_next_qs = self._target_qv(next_state, True, True)
        target_next_q = torch.min(torch.cat(target_next_qs, 1), 1)[0]
        target_next_q = target_next_q.max(1)[0].view(-1, 1)

        target_q = utils.td_target(reward=batch.reward,
                                   mask=batch.mask,
                                   next_value=target_next_q,
                                   discount=self._discount).detach()

        expected_q = self._qv(state, True)[idx].gather(1, batch.action.long())
        loss = utils.mse_loss(expected_q, target_q)
        return loss

    def _compute_categorical_loss(self, batch, idx):
        state = self._state_normalizer(batch.state, self._device)
        next_state = self._state_normalizer(batch.next_state, self._device)

        batch_vec = torch.arange(self._batch_size).long()
        next_probs = torch.cat(self._target_qv(next_state, True, True), 1)
        exp_atoms = self._atoms.expand_as(next_probs)
        next_q, head_idx = next_probs.mul(exp_atoms).sum(-1).min(1)
        next_action = next_q.argmax(-1).long()
        head_idx = head_idx.gather(1, next_action.view(-1, 1)).squeeze()

        next_probs = next_probs[batch_vec, head_idx, next_action, :]
        target_probs = utils.l2_projection(next_probs=next_probs,
                                           reward=batch.reward,
                                           mask=batch.mask,
                                           atoms=self._atoms,
                                           v_limit=self._v_limit,
                                           discount=self._discount).detach()
        action = batch.action.squeeze().long()
        probs = self._qv(state, True)[idx][batch_vec, action, :]
        probs = torch.clamp(probs, 1e-7, 1.0)
        loss = -(target_probs * probs.log()).sum(-1)

        return loss.mean()

    def load(self, path: str) -> NoReturn:
        model = torch.load(path)
        if model:
            self._qv.load_state_dict(model['model'])
            self._target_qv.load_state_dict(model['target_model'])
            self._step = model['step']
            if 'state_norm' in model:
                self._state_normalizer.load(model['state_norm'])

    def save(self):
        state_dict = dict()
        state_dict['model'] = self._qv.state_dict()
        state_dict['target_model'] = self._target_qv.state_dict()
        state_dict['step'] = self._step
        if self._use_state_normalization:
            state_dict['state_norm'] = self._state_normalizer.state_dict()
        torch.save(state_dict, 'model_{}.pth'.format(self._step))

    @property
    def parameters(self):
        return self._qv.named_parameters()

    @property
    def target_parameters(self):
        return self._target_qv.named_parameters()
