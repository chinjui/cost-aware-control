# Computational Cost-Aware Control Using Hierarchical Reinforcement Learning

This project uses and is modified from [stable-baselines](https://github.com/hill-a/stable-baselines/blob/master/docs/guide/rl_zoo.rst).

## Installation
```bash
git clone https://github.com/chinjui/macro-decision.git
cd macro-decision
pip install -r requirement.txt
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
cd stable-baselines
pip install -e .
```

## Training example
```bash
python rl-zoo/train_decision_net.py --env Swimmer-v3 --algo dqn --sub-hidden-sizes 8 256 --sub-policy-costs 1 245 --policy-cost-coef 1e-4
```
