# Report - StartTrek

**Date:** 2026-5-15 | **Run:** 2026-5-15_16:24:7

---

## Task

Ablation Study (Wind Training) - Hyperparameters (lr=0.0001, batch_size=128)

### Context

This run trains the DQN agent with `random_wind` enabled to improve its robustness. We then evaluate it both with and without wind.

### Expectations

We expect the model to learn to stabilize despite the wind, achieving good evaluation scores in both calm and windy environments.

---

## Results

Average Score (No Wind) = 141.90
Average Score (With Wind) = 73.67
Combined Average Score = 107.78

Success Rate (Landed) = 66.0% (last 100 training eps)
Crash Rate = 18.0% (last 100 training eps)
Timeout Rate = 16.0% (last 100 training eps)

### Observations

- **Behavior observed:** The model achieved a combined average evaluation score of 107.78.
- **Difference vs. previous run:** Trained with random wind enabled to force robust behavior.

---

## Files

- `models/checkpoint_model_*.pth`
- `configs/*.yaml`
- `videos/`
- `logs/logs.csv`
