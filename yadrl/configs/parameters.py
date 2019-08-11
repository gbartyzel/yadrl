import os
from typing import Dict, Any, Optional

import yaml


class Parameters(object):

    def __init__(self, config_path: Optional[str] = None):
        self.__parameters = self.__load_config(config_path)

    @property
    def dqn(self) -> Dict[str, Any]:
        if 'dqn' in self.__parameters:
            return self.__parameters['dqn']
        raise ValueError

    @property
    def qrdqn(self) -> Dict[str, Any]:
        if 'dqn' in self.__parameters:
            return self.__parameters['qrdqn']
        raise ValueError

    @property
    def ddpg(self) -> Dict[str, Any]:
        if 'ddpg' in self.__parameters:
            return self.__parameters['ddpg']
        raise ValueError

    @property
    def td3(self) -> Dict[str, Any]:
        if 'td3' in self.__parameters:
            return self.__parameters['td3']
        raise ValueError

    @property
    def sac(self) -> Dict[str, Any]:
        if 'sac' in self.__parameters:
            return self.__parameters['sac']
        raise ValueError

    @staticmethod
    def __load_config(path) -> Dict[str, Dict[str, Any]]:
        if not path:
            dir_path, script = os.path.split(os.path.realpath(__file__))
            path = os.path.join(dir_path, 'parameters.yaml')
        config_file = open(path, 'r')
        parameters = yaml.load(config_file, Loader=yaml.SafeLoader)
        config_file.close()
        return parameters
