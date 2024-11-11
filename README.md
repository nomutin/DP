# Diffusion Policy

![python](https://img.shields.io/badge/python-3.10-blue)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/nomutin/DP/actions/workflows/ci.yaml/badge.svg)](https://github.com/nomutin/DP/actions/workflows/ci.yaml)

Minimal implementation of [Diffusion Policy](https://arxiv.org/abs/2303.04137).
Modified from [lerobot's DP](https://github.com/huggingface/lerobot/tree/main/lerobot/common/policies/diffusion).

## Installation

```bash
pip install git+https://github.com/nomutin/DP.git
```

## API

```python
from dp import DiffusionPolicy

policy = DiffusionPolicy(
    states_seq_len=2,
    action_dim=8,
    action_generate_len=16,
    action_seq_len=8,
    ...
)

action_target = torch.randn(4, 16, 8)
states = torch.randn(4, 2)

loss = policy.forward(states, action_target)
action_prediction = policy.select_action(states) # [4, 8]
```

## Example

Training script is in [./example/](example/).

```bash
uv sync --extra train
uv run python example/cli.py fit --config example/config.yaml
```
