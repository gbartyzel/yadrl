import torch


def mse_loss(prediction: torch.Tensor,
             target: torch.Tensor) -> torch.Tensor:
    return torch.mean(0.5 * (prediction - target).pow(2))


def huber_loss(prediction: torch.Tensor,
               target: torch.Tensor,
               delta: float = 1.0) -> torch.Tensor:
    error = target - prediction
    loss = torch.where(torch.abs(error) <= delta,
                       0.5 * error.pow(2),
                       delta * (error.abs() - 0.5 * delta))
    return loss.mean()
