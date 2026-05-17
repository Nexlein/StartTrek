# Daily Report - StartTrek

**Date:** 2026-05-17

## Work Completed

### Task: feat/figures-graphics — Loss logging & plot script

- **Details:** Modified `learn()` in `agent.py` to return `loss.item()`. Updated `train.py` to collect loss per episode (average) and log it to the CSV. Updated `artifacts.py` log header to include `Loss` column. Created `plot.py` to read `logs.csv` and generate PNG graphs (Reward, Length, Epsilon, Loss, Termination).
- **Observations:** Agent crashes frequently in early episodes, stabilizes around episode 50+, occasional successful landings from episode 56 onward.

## Results & Metrics

- **Average Score (100 episodes):** ~-52
- **Success Rate (Landed):** ~10% (10/100 episodes with `sleep` termination)
- **Episode End Reasons:** Crash (~55%), Timeout/out-of-view (~35%), Landed (~10%)

## Issues & Solutions

- **Issue:** `logs.csv` was empty when running `plot.py` on a stale artifact folder.
- **Solution:** Used the correct artifact folder from the current training run.
- **Issue:** `termination.png` crashed with `IndexError` on empty value counts.
- **Solution:** Added empty check before plotting.
