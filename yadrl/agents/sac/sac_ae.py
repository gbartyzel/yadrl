from copy import deepcopy
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid

import yadrl.common.ops as ops
import yadrl.common.types as t
from yadrl.agents.sac.sac import SAC
from yadrl.common.normalizer import ScaleNormalizer
from yadrl.networks.head import Head
from yadrl.common.memory import Batch


class SAC_AE(SAC, agent_type="sac_ae"):
    def __init__(
        self,
        qv_learning_rate: float,
        encoder_learning_rate: float,
        decoder_learning_rate: float,
        encoder_polyak: float,
        latent_factor: float,
        decoder_weight_decay: float,
        **kwargs
    ):
        super().__init__(qv_learning_rate=qv_learning_rate, **kwargs)
        self._latent_factor = latent_factor
        self._encoder_polyak = encoder_polyak

        self._decoder_optim = optim.Adam(
            self.decoder.parameters(),
            decoder_learning_rate,
            weight_decay=decoder_weight_decay,
        )
        self._encoder_optim = optim.Adam(
            self.encoder.parameters(), encoder_learning_rate
        )
        self._qv_optim = optim.Adam(self.qv.parameters(), qv_learning_rate)

        self._decoder_normalizer = ScaleNormalizer(-1.0, 1.0, 0.0, 1.0)

    @property
    def encoder(self) -> nn.Module:
        return self._networks["encoder"]

    @property
    def decoder(self) -> nn.Module:
        return self._networks["decoder"]

    @property
    def target_encoder(self) -> nn.Module:
        return self._networks["target_encoder"]

    def _initialize_networks(self, phi: t.TModuleDict) -> t.TModuleDict:
        networks = super()._initialize_networks(phi)
        encoder_net = phi["encoder"]
        decoder_net = phi["decoder"]
        target_encoder_net = deepcopy(encoder_net)

        encoder_net.to(self._device)
        decoder_net.to(self._device)
        target_encoder_net.to(self._device)
        target_encoder_net.eval()

        networks["encoder"] = encoder_net
        networks["decoder"] = phi["decoder"]
        networks["target_encoder"] = target_encoder_net

        return networks

    def _preprocess_observation(self, state: np.ndarray):
        new_state = super()._preprocess_observation(state)
        return self.encoder(new_state)

    def _update(self):
        batch = self._memory.sample(self._batch_size, self._state_normalizer)
        self._update_decoder(batch.state)
        self._update_critic(batch)

        update_freq = self._policy_update_frequency * self._update_frequency
        if self._env_step % update_freq == 0:
            self._update_actor_and_entropy(self.encoder(batch.state).detach())
            self._update_target(self.qv, self.target_qv)
            self._update_target(self.encoder, self.target_encoder, self._encoder_polyak)

    def _update_critic(self, batch: Batch):
        with torch.no_grad():
            target_z = self.target_encoder(batch.next_state)
            next_action = self.pi.sample(target_z)
            log_prob = self.pi.log_prob(next_action)
            self.target_qv.sample_noise()
            target_next_q = torch.min(*self.target_qv(target_z, next_action))
            target_next_v = target_next_q - self._temperature * log_prob
            target_q = ops.td_target(
                batch.reward,
                batch.mask,
                target_next_v,
                batch.discount_factor * self._discount,
            )
        self.qv.sample_noise()
        expected_qs = self.qv(self.encoder(batch.state), batch.action)

        loss = sum(ops.mse_loss(q, target_q) for q in expected_qs)
        self._writer.add_scalar("loss/qv", loss, self._env_step)

        self._qv_optim.zero_grad()
        self._encoder_optim.zero_grad()
        loss.backward()
        if self._qv_grad_norm_value > 0.0:
            nn.utils.clip_grad_norm_(self.qv.parameters(), self._qv_grad_norm_value)
        self._qv_optim.step()
        self._encoder_optim.step()

    def _update_decoder(self, state: torch.Tensor):
        z = self.encoder(state)
        recon = self.decoder(z)
        # norm_image = self._decoder_normalizer(state, self._device)

        latent_loss = self._latent_factor * (0.5 * z.pow(2).sum(1)).mean()
        recon_loss = ops.mse_loss(recon, state)
        loss = recon_loss + latent_loss

        self._encoder_optim.zero_grad()
        self._decoder_optim.zero_grad()
        loss.backward()
        self._encoder_optim.step()
        self._decoder_optim.step()

        self._writer.add_scalar("loss/encoder", latent_loss, self._env_step)
        self._writer.add_scalar("loss/decoder", recon_loss, self._env_step)
        

        r_i = make_grid(torch.cat(recon.split(3, dim=1))[:8])
        s_i = make_grid(torch.cat(state.split(3, dim=1))[:8])

        self._writer.add_image('recon', r_i, self._env_step)
        self._writer.add_image('target', s_i, self._env_step) 
