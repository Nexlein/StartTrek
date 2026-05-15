# Termination Cause Tracking

## Objective

The goal of this task was to identify and log the specific reason an episode ends, as requested by the project requirements ("Correctly handle and log terminated vs truncated and the termination reason: crash, out-of-view, sleep").

## Changes Implemented

1. **Updated Artifacts Logger (`artifacts.py`)**:
    - Modified the default `log_header` in `Artifacts.__init__` to include a new column for `Termination`. The header is now `"Episode,Reward,Length,Epsilon,Termination"`.
2. **Training Loop Update (`train.py`)**:
    - Introduced a variable `termination_reason`.
    - At each step, if the environment returns `terminated` or `truncated` (meaning the episode is over), the script evaluates the cause:
        - If `terminated` and the reward is strictly negative and low (`<= -100`), the reason is defined as `"crash"`.
        - If `terminated` and the reward is normal/positive, the reason is defined as `"sleep"` (successful or normal landing/end).
        - If `truncated`, it means the time limit was reached or the lander went out of bounds, so the reason is defined as `"out-of-view"`.
    - This `termination_reason` is appended to the `artifact.log_step` call and the console print.
3. **Baselines Update (`baseline.py`)**:
    - Added the same logic to both `random_policy` and `heuristic_policy` to calculate and print the termination cause for baseline runs.

## Impact

The generated `logs.csv` now tracks exactly why episodes end. This fulfills the metric tracking requirement and provides the necessary data for generating statistics (like pie charts) in the future evaluation or plotting scripts.
