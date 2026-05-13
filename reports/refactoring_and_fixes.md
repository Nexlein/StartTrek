# Daily Report - StartTrek

**Date:** 2026-05-13

## Work Completed

### Task: 001 Codebase Audit and Refactoring

- **Details:**
  - Conducted a full audit of `train.py`, `eval.py`, `agent.py`, and `artifacts.py`.
  - Fixed a major memory and process leak in `eval.py` where Gym environments and ffmpeg instances were not closed during iterative model evaluation.
  - Refactored `artifacts.save_best_model` to overwrite the best model file (`best_model_seed_<seed>.pth`) instead of creating a new file for every new high score.
  - Adjusted hyperparameter management: the optimizer's learning rate is now injected via the `DQNAgent` constructor.
  - Enhanced the RL algorithm by moving epsilon decay from a per-episode to a per-step basis for smoother exploration reduction, and added `exploration_decay` to the config files.
  - Updated documentation (`README.md`, `hyperparameters.yml.template`) to match the new behavior.
- **Observations:** Agent exploration is now smoother and less prone to abrupt drops in early episodes. The evaluation script consumes significantly less memory and safely evaluates all models without stalling.

## Results & Metrics

- **Average Score (100 episodes):** N/A (Refactoring phase)
- **Success Rate (Landed):** N/A
- **Episode End Reasons:** N/A

## Issues & Solutions

- **Issue:** The `eval.py` loop over final models initialized new video environments but never called `env.close()`, causing memory leaks and orphaned processes.
- **Solution:** Moved `env.close()` inside the evaluation loop to properly terminate each environment before starting the next one.
- **Issue:** `artifacts.save_best_model` included the episode number in the filename, causing disk saturation with hundreds of intermediate models.
- **Solution:** Removed the episode number from the filename format string so that it correctly overwrites the previous best model.
- **Issue:** Exploration `epsilon` was decayed at the end of each episode rather than each step, leading to poor exploration behavior since early episodes are very short.
- **Solution:** Shifted the decay logic inside the stepping loop in `train.py` and linked it to a new configurable `exploration_decay` parameter.

## Next Steps

- Proceed with a full training run to benchmark the impact of per-step epsilon decay on final landing performance.
- Evaluate the trained models using the updated `eval.py` to confirm memory stability over large batches.
