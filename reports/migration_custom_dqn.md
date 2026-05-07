# Daily Report - StartTrek

**Date:** 2026-07-05

## Work Completed

### Task: [1] Migration from Stable-Baselines3 version to PyTorch custom implementation of DQN

- **Details:** Delete non authorized use of SB3 DQN implementation, and replace it by a custom implementation of DQN using PyTorch.

### Task: [2] Add reproductibility script

- **Details:** Fix the seed of the random generator to ensure reproductibility of the results and add a script to run the training with a fixed seed (reproduce.sh).

### Task: [3] Baseline implementation

- **Details:** Create random and heuristic baselines to compare the performance of the DQN agent.
