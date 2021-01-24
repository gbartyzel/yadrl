# Yet Another Deep Reinforcement Learning (YADRL)

YADRL is set of implementations of reinforcement learning algorithms. It is
written fully in PyTorch.

## Implemented algorithms

![Algos](misc/implemented_algos.png)

## Installation

Install YADRL package:

```bash
pip install .
```

## Execution example

To execute experiments just type run following command:

```bash
yadrl_runner --config_path experiments/carracing_sac.yaml
```

## TODO

Algorithm implementation:

- [ ] PPO
- [ ] TRPO
- [ ] VPG
- [ ] PER
- [ ] APE-X
- [ ] MPO
- [ ] HER
- [ ] IQN and M-IQN
- [ ] FQF

Software improvements:

- [ ] Documentation
- [ ] Distributed training
- [ ] On-policy base agent
