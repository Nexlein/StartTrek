# Report — StartTrek

**Date:** 2026-5-15 | **Run:** 2026-5-15_15:16:53

---

## Task

Ablation Study - Hyperparameter Tuning (lr=0.0001, batch_size=64)

### Context

This run is part of an ablation study to find the best hyperparameters for the DQN agent. We are testing different combinations of `learning_rate` and `batch_size`.
This specific test uses a learning rate of 0.0001 and a batch size of 64.

### Expectations

We expect to see how the learning rate and batch size affect the convergence speed and the final evaluation performance. A smaller batch size might lead to more noisy gradients but faster updates, while the learning rate directly affects the step size during optimization.

---

## Results

Average Score (10 eps eval) = 197.48
Success Rate (Landed) = 93.0% (last 100 training eps)
Crash Rate = 3.0% (last 100 training eps)
Timeout Rate = 4.0% (last 100 training eps)

### Observations

- **Behavior observed:** The model achieved an average evaluation score of 197.48. During the last 100 episodes of training, the agent successfully landed (or survived) 93.0% of the time, crashed 3.0% of the time, and timed out 4.0% of the time.
- **Difference vs. previous run:** This was tested alongside other configurations in the ablation study. Compare with other artifacts to see the relative performance. The optimal configuration found in the study was lr=0.0001 and bs=64.

---

## Files

- `models/checkpoint_model_*.pth`
- `configs/*.yaml`
- `videos/`
- `logs/logs.csv`
