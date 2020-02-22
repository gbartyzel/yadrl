from enum import Enum
from typing import Tuple

import torch


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


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
    loss = torch.where(torch.abs(error) < delta,
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
              next_value: torch.Tensor,
              discount: float = 0.99) -> torch.Tensor:
    return reward + mask * discount * next_value


def l2_projection(next_probs: torch.Tensor,
                  reward: torch.Tensor,
                  mask: torch.Tensor,
                  atoms: torch.Tensor,
                  v_limit: Tuple[float, float],
                  discount: float) -> torch.Tensor:
    target_probs = torch.zeros(next_probs.shape, device=next_probs.device)
    next_atoms = td_target(reward, mask, atoms, discount)
    next_atoms = torch.clamp(next_atoms, *v_limit)

    z_delta = (v_limit[1] - v_limit[0]) / (atoms.shape[-1] - 1)
    bj = (next_atoms - v_limit[0]) / z_delta
    l = bj.floor()
    u = bj.ceil()

    delta_l_prob = next_probs * (u + (u == l).float() - bj)
    delta_u_prob = next_probs * (bj - l)

    for i in range(next_probs.shape[0]):
        target_probs[i].index_add_(0, l[i].long(), delta_l_prob[i])
        target_probs[i].index_add_(0, u[i].long(), delta_u_prob[i])

    return target_probs
