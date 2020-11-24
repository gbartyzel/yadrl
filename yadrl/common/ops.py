import datetime
import os

import numpy as np
import torch


def uniform_init(x: torch.nn.Module, low: float = -3e-3, high: float = 3e-3):
    torch.nn.init.uniform_(x.weight.data, low, high)
    torch.nn.init.uniform_(x.bias, low, high)


def orthogonal_init(x: torch.nn.Module):
    torch.nn.init.orthogonal_(x.weight.data, gain=np.sqrt(2))
    torch.nn.init.constant_(x.bias.data, 0.0)


def to_tensor(data, device):
    return torch.from_numpy(data).float().to(device)


def l2_loss(model: torch.nn.Module, l2_lambda: float) -> torch.Tensor:
    l2_term = torch.zeros(1)
    for name, parameters in model.named_parameters():
        if 'weight' in name:
            l2_term += parameters.norm(2)
    return l2_term * l2_lambda


def mse_loss(prediction: torch.Tensor,
             target: torch.Tensor,
             reduction: str = 'mean') -> torch.Tensor:
    loss = 0.5 * (prediction - target).pow(2)

    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    return loss


def huber_loss(prediction: torch.Tensor,
               target: torch.Tensor,
               delta: float = 1.0,
               reduction: str = 'mean') -> torch.Tensor:
    error = target - prediction
    loss = torch.where(torch.abs(error) <= delta,
                       0.5 * error.pow(2),
                       delta * (error.abs() - 0.5 * delta))
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    return loss


def quantile_hubber_loss(prediction: torch.Tensor,
                         target: torch.Tensor,
                         cumulative_density: torch.Tensor,
                         delta: float = 1.0,
                         reduction: str = 'mean') -> torch.Tensor:
    transpose_target = target.t().unsqueeze(-1)
    diff = transpose_target - prediction
    loss = huber_loss(prediction, transpose_target, delta, 'none')
    loss *= torch.abs(cumulative_density - (diff < 0.0).float())
    loss = loss.mean(0).sum(1)

    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    return loss


def td_target(reward: torch.Tensor,
              mask: torch.Tensor,
              target: torch.Tensor,
              discount: float) -> torch.Tensor:
    return reward + (1.0 - mask) * discount * target


def l2_projection(next_probs: torch.Tensor,
                  atoms: torch.Tensor,
                  target_atoms: torch.Tensor) -> torch.Tensor:
    v_min = float(atoms.squeeze()[0].clone().cpu().numpy())
    v_max = float(atoms.squeeze()[-1].clone().cpu().numpy())
    z_delta = (v_max - v_min) / (atoms.shape[-1] - 1)
    atoms = atoms.expand_as(next_probs)
    target_atoms = target_atoms.clamp(v_min, v_max).t().unsqueeze(-1)
    next_probs = next_probs.t().unsqueeze(-1)

    target_probs = (1.0 - (target_atoms - atoms).abs() / z_delta).clamp(0, 1)
    target_probs *= next_probs
    return target_probs.sum(0)


def set_seeds(seed: int = 1337):
    torch.random.manual_seed(seed)
    np.random.seed(seed)


def create_log_dir(log_dir: str, experiment_name: str) -> str:
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    now = datetime.datetime.now().strftime('%d_%m_%y_%H_%M_%S')
    log_name = '{}_{}'.format(experiment_name, now)
    return os.path.join(log_dir, log_name)
