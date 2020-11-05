import argparse

import yaml

from yadrl.agents.agent import BaseAgent
from yadrl.common.configuration import Configuration


class Runner:
    def __init__(self):
        args = self._parse_arguments()
        self._test_flag = args.test

    def _load_configuration(self, path):
        config_file = open(path, 'r')
        config = yaml.safe_load(config_file)
        config_file.close()
        return config

    @staticmethod
    def _parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--config_path', type=str, required=True)
        return parser.parse_args()


def main():
    args = _parse_arguments()
    configs = Configuration(config_path=args.config_path)
    agent = BaseAgent.build(agent_type=configs.agent_type,
                            memory=configs.memory,
                            exploration_strategy=configs.exploration_strategy,
                            **configs.common,
                            **configs.specific)
    agent.train(100000)


if __name__ == '__main__':
    main()
