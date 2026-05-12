# StartTrek - Lunar Lander Autonomous Landing

## Installation

### 1. Create a virtual environment

Python 3.14 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Baseline evaluation

Run the reference policies, including the random and heuristic baselines:

```bash
python baseline.py
```

### Training

Train the DQN agent. You can specify a random seed or enable wind disturbances (these CLI arguments override `configs/settings.yml`):

```bash
# Default training (uses settings from config)
python train.py

# Train with a specific seed
python train.py --seed 42

# Train with wind enabled (uses wind_power from config)
python train.py --wind

# Train with wind enabled and a custom wind power
python train.py --wind 20.0
```

### Evaluation

Evaluate a trained model and generate a landing video. You can evaluate the default model or specify a path, and optionally override the seed and wind settings:

```bash
# Evaluate the default model (results/models/dqn-model.pth)
python eval.py

# Evaluate a specific model
python eval.py results/models/<MODEL_NAME>.pth

# Evaluate with a specific seed and wind enabled
python eval.py --seed 42 --wind
```

### Reproducibility

Run the full experiment suite on the five official seeds:

```bash
./scripts/reproduce.sh
```

## Project Structure

| Path | Description |
| --- | --- |
| `baseline.py` | Reference random and heuristic policies. |
| `train.py` | Main training entry point for the DQN agent. |
| `eval.py` | Evaluation script for trained agents. |
| `model/` | PyTorch implementation of the agent, memory, and neural network. |
| `configs/` | Hyperparameter configuration files in YAML format. |
| `results/` | Saved models, evaluation logs, and generated videos. |
| `scripts/reproduce.sh` | One-command script to reproduce the complete workflow. |

## Methodology

The agent is built around a Deep Q-Network (DQN) pipeline:

- A multilayer perceptron (MLP) approximates the action-value function.
- A replay buffer stores past transitions and improves training stability.
- A target network reduces oscillations by stabilizing value targets.
- An epsilon-greedy strategy balances exploration and exploitation during training.
