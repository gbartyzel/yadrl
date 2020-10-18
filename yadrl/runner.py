import argparse

from yadrl.common.configuration import Configuration


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--config_path', type=str, required=True)
    return parser.parse_args()


def main():
    args = _parse_arguments()
    configs = Configuration(config_path=args.config_path)
    print(configs.state_normalizer)


if __name__ == '__main__':
    main()
