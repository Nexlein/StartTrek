# Report — StartTrek

**Date:** 2026-5-15 | **Run:** 2026-5-15_17:40:53/

---

## Task

Final 5-Seed Evaluation & 95% Confidence Interval Analysis

### Context

This run constitutes the exhaustive evaluation of the DQN agent across 5 distinct canonical seeds (0 to 4). The agent was trained using a previous hyperparameter configuration. The goal is to mathematically determine the reliability of the agent using a 95% Confidence Interval across these multiple training instances.

### Expectations

We expect to see the variance between different seeds due to the stochastic nature of the environment initialization, wind variables, and the exploration strategy. The objective is to evaluate whether this specific configuration consistently hits the acceptance criteria of an average score $\ge$ 200.

---

## Results

Average Score (100 eps per seed) = **191.10 ± 40.90 (95% CI)**

**Individual Seed Means:**

- Seed 0: 203.12
- Seed 1: 242.68
- Seed 2: 179.20
- Seed 3: 170.57
- Seed 4: 159.92

*(Note: Success/Crash/Timeout rates require parsing the full training logs of the 5 seeds).*

### Observations

- **Behavior observed:** The overall mean score across the 5 seeds is **191.10**, which falls slightly short of the 200 threshold required to consider the environment solved. The 95% Confidence Interval of ± 40.90 indicates significant variance in agent performance depending on the initialization seed. While Seeds 0 and 1 successfully achieved excellent scores above 200, Seeds 2, 3, and 4 underperformed.
- **Difference vs. previous run:** This evaluation highlights the exact reason why we just updated the codebase in our recent intervention. The variance and sub-200 score observed in this specific artifact justify our decision to lower the `batch_size` to 64 and increase `max_episodes` to 1500. A new run with these updated parameters should successfully push the lower bound of the CI above 200.

---

## Files

- `models/checkpoint_model_*.pth`
- `configs/*.yaml`
- `videos/`
- `logs/logs.csv`
- `logs/eval_scores_seed_*.csv`
