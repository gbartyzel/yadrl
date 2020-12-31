import argparse
import collections
from dataclasses import asdict

from yadrl.agents.agent import Agent
from yadrl.common.configuration import Configuration


def flatten_dictionary(in_dict):
    items: dict = {}
    for k, v in in_dict.items():
        if isinstance(v, collections.abc.MutableMapping) and k != 'body':
            items.update(flatten_dictionary(v).items())
        else:
            items.update({k: v})
    return items


class Runner:
    def __init__(self):
        args: argparse.Namespace = self._parse_arguments()
        self._test_flag: bool = args.test
        self._num_train_steps: int = args.num_train_steps
        self._config: Configuration = Configuration(args.config_path)
        self._agent: Agent = self._create_agent()
        self.start()

    def start(self):
        if self._test_flag:
            self._agent.eval()
        else:
            self._agent.train(self._num_train_steps)

    def _create_agent(self) -> Agent:
        dict_config: dict = asdict(self._config)
        for i in dict_config.copy():
            if dict_config[i] is None: dict_config.pop(i)
        return Agent.build(**flatten_dictionary(dict_config))

    @staticmethod
    def _parse_arguments() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--num_train_steps', type=int, default=1000000)
        parser.add_argument('--config_path', type=str, required=True)
        return parser.parse_args()


def main():
    Runner()


if __name__ == '__main__':
    main()
