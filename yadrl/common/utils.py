from enum import Enum

import torch


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


def mse_loss(prediction: torch.Tensor,
             target: torch.Tensor,
             reduction: Reduction = Reduction.MEAN) -> torch.Tensor:
    loss = 0.5 * (prediction - target).pow(2)

    if reduction == Reduction.MEAN:
        return torch.mean(loss)
    elif reduction == Reduction.SUM:
        return torch.sum(loss)
    return loss


def huber_loss(prediction: torch.Tensor,
               target: torch.Tensor,
               delta: float = 1.0,
               reduction: Reduction = Reduction.MEAN) -> torch.Tensor:
    error = target - prediction
    loss = torch.where(torch.abs(error) <= delta,
                       0.5 * error.pow(2),
                       delta * (error.abs() - 0.5 * delta))
    if reduction == Reduction.MEAN:
        return torch.mean(loss)
    elif reduction == Reduction.SUM:
        return torch.sum(loss)
    return loss


def quantile_hubber_loss(prediction: torch.Tensor,
                         target: torch.Tensor,
                         cumulative_probs: torch.Tensor,
                         delta: float = 1.0,
                         reduction: Reduction = Reduction.MEAN) -> torch.Tensor:
    diff = target - prediction
    loss = huber_loss(prediction, target, delta, Reduction.NONE)
    loss = (loss * torch.abs(cumulative_probs - (diff < 0.0).float())).sum(-1)

    if reduction == Reduction.MEAN:
        return torch.mean(loss)
    elif reduction == Reduction.SUM:
        return torch.sum(loss)
    return loss
