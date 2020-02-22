__all__ = ['DDPG', 'DQN', 'TD3', 'SAC', 'SACDiscrete', 'MaxminDQN']

from yadrl.agents.ddpg import DDPG
from yadrl.agents.dqn import DQN
from yadrl.agents.maxmin_dqn import MaxminDQN
from yadrl.agents.sac import SAC
from yadrl.agents.sac_discrete import SACDiscrete
from yadrl.agents.td3 import TD3
