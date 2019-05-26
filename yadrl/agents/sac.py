import torch
import torch.nn as nn
import torch.optim as optim
from yadrl.agents.base import BaseOffPolicy


class SAC(BaseOffPolicy):
    def __init__(self,
                 actor_phi: nn.Module,
                 critic_phi: nn.Module,
                 ):

        super(SAC, self).__init__()
