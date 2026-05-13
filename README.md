# StartTrek - Lunar Lander Autonomous Landing

## Installation

### 1. Create a virtual environment

Python 3.14 is recommended.

```bash
python3.14 -m venv .venv
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

Train the DQN agent. You can specify a random seed or enable domain randomization (these CLI arguments override `configs/settings.yml`). During training, the agent saves a checkpoint every 50 episodes and automatically keeps track of the best performing model (saved as `best_model_seed_<seed>.pth`):

```bash
# Default training (uses settings from config)
python train.py

# Train with a specific seed
python train.py --seed 42

# Train with domain randomization (randomly toggles wind and power each episode)
python train.py --random-wind
```

### Evaluation

Evaluate trained models and generate landing videos. You must specify a path to an artifact folder. You can optionally override the seed and wind settings. The wind behavior in evaluation is strict and can be explicitly controlled:

```bash
# Evaluate models in a specific artifact folder (uses evaluation wind settings from config)
python eval.py --artifact artifacts/YYYY-MM-DD_HH:MM:SS/

# Evaluate with a specific seed and default wind power (15.0)
python eval.py --artifact artifacts/YYYY-MM-DD_HH:MM:SS/ --seed 42 --wind

# Evaluate with a specific seed and a custom wind power (e.g., 20.0)
python eval.py --artifact artifacts/YYYY-MM-DD_HH:MM:SS/ --seed 42 --wind 20.0
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
