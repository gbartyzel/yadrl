import datetime
import os

import numpy as np
import torch as th


def uniform_init(x: th.nn.Module, low: float = -3e-3, high: float = 3e-3):
    th.nn.init.uniform_(x.weight.data, low, high)
    th.nn.init.uniform_(x.bias, low, high)


def orthogonal_init(x: th.nn.Module):
    th.nn.init.orthogonal_(x.weight.data, gain=np.sqrt(2))
    th.nn.init.constant_(x.bias.data, 0.0)


def scaled_logsoftmax(input_value: th.Tensor,
                      term: float,
                      dim: int = -1,
                      keepdim: bool = True) -> th.Tensor:
    diff = input_value - input_value.max(dim, keepdim)[0]
    return diff - term * th.log(th.exp(diff / term).sum(dim, keepdim))


def to_tensor(data, device):
    return th.from_numpy(data).float().to(device)


def l2_loss(model: th.nn.Module, l2_lambda: float,
            device: th.device) -> th.Tensor:
    l2_term = th.tensor(0., device=device)
    for name, parameters in model.named_parameters():
        if 'weight' in name:
            l2_term += parameters.norm(2)
    return l2_term * l2_lambda


def mse_loss(prediction: th.Tensor,
             target: th.Tensor,
             reduction: str = 'mean') -> th.Tensor:
    loss = 0.5 * (target - prediction).pow(2)

    if reduction == 'mean':
        return th.mean(loss)
    elif reduction == 'sum':
        return th.sum(loss)
    return loss


def huber_loss(prediction: th.Tensor,
               target: th.Tensor,
               delta: float = 1.0,
               reduction: str = 'mean') -> th.Tensor:
    error = prediction - target
    loss = th.where(error.abs() < delta,
                    0.5 * error.pow(2),
                    delta * (error.abs() - 0.5 * delta))
    if reduction == 'mean':
        return th.mean(loss)
    elif reduction == 'sum':
        return th.sum(loss)
    return loss


def quantile_hubber_loss(prediction: th.Tensor,
                         target: th.Tensor,
                         cumulative_density: th.Tensor,
                         delta: float = 1.0,
                         reduction: str = 'mean') -> th.Tensor:
    transpose_target = target.t().unsqueeze(-1)
    prediction = prediction.unsqueeze(0)
    diff = transpose_target - prediction
    loss = huber_loss(prediction, transpose_target, delta, 'none')
    loss = loss * th.abs(
        cumulative_density - (diff.detach() < 0.0).float()) / delta
    loss = loss.mean(0).sum(-1)

    if reduction == 'mean':
        return th.mean(loss)
    elif reduction == 'sum':
        return th.sum(loss)
    return loss


def td_target(reward: th.Tensor,
              mask: th.Tensor,
              target: th.Tensor,
              discount: float) -> th.Tensor:
    return reward + (1.0 - mask) * discount * target


"""
def l2_projection(p: th.Tensor,
                  z: th.Tensor,
                  tz: th.Tensor) -> th.Tensor:
    v_min = float(z.squeeze()[0].clone().cpu().numpy())
    v_max = float(z.squeeze()[-1].clone().cpu().numpy())
    z_delta = (v_max - v_min) / (z.shape[-1] - 1)
    z = z.expand_as(p)
    tz = tz.clamp(v_min, v_max).t().unsqueeze(-1)

    p = p.t().unsqueeze(-1)
    proj = (1.0 - (tz - z).abs() / z_delta).clamp(0, 1) * p
    return proj.sum(0)


"""


def l2_projection(p: th.Tensor,
                  z: th.Tensor,
                  tz: th.Tensor) -> th.Tensor:
    v_min: float = float(z.squeeze()[0].clone().cpu().numpy())
    v_max: float = float(z.squeeze()[-1].clone().cpu().numpy())
    z_delta: float = (v_max - v_min) / (z.shape[-1] - 1)
    proj: th.Tensor = th.zeros_like(p)

    tz: th.Tensor = th.clamp(tz, v_min, v_max)
    bj: th.Tensor = (tz - v_min) / z_delta
    l: th.Tensor = bj.floor()
    u: th.Tensor = bj.ceil()

    delta_l_p: th.Tensor = p * (u + (u == l).float() - bj)
    delta_u_p: th.Tensor = p * (bj - l)

    for i in range(p.shape[0]):
        proj[i].index_add_(0, l[i].long(), delta_l_p[i])
        proj[i].index_add_(0, u[i].long(), delta_u_p[i])

    return proj


def set_seeds(seed: int = 1337):
    th.random.manual_seed(seed)
    np.random.seed(seed)


def create_log_dir(log_dir: str, experiment_name: str) -> str:
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    now = datetime.datetime.now().strftime('%d_%m_%y_%H_%M_%S')
    log_name = '{}_{}'.format(experiment_name, now)
    return os.path.join(log_dir, log_name)
