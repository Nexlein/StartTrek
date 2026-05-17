# Daily Report - StartTrek

**Date:** 2026-05-17

## Work Completed

### Task: Automated Charts & 95% Confidence Interval Integration

- **Details:**
  - **`plot.py` Overhaul:** The plotting script was entirely rewritten to meet the project's reproducibility requirement (`≥5 seeds, report mean ± 95% CI`). It now handles two dynamic display modes based on the data present in the `logs.csv` file:
    - *Single-Seed Mode:* If there is only one training iteration (run) in the logs, the graph displays the "raw" data (with a transparent background) representing the exact score/statistic at each episode, along with a "smoothed" curve using a 20-episode moving average to indicate the global trend.
    - *Multi-Seed Mode:* If multiple runs (seeds) are detected, the data is automatically grouped by episode. The script then plots the Mean across all runs and surrounds it with a shaded area representing the 95% Confidence Interval (95% CI) calculated using Student's t-distribution (`scipy.stats`).
  - **Automation via `artifacts.py` & `train.py`:** The artifact architecture was extended to include a `charts/` subfolder. At the end of the `train()` cycle in `train.py`, the chart generation (`plot_all`) is now called automatically, and the resulting `.png` files are saved directly into the current artifact.
- **Observations:** The generated graphs are now fully exploitable for the final report. We have eliminated the illegible zigzags that occurred when logging multiple runs: we now obtain scientific-quality curves, ready in one click (validating the "one-click repro" requirement).

## Metrics

- **Episode End Reasons:** Handled properly; the bar chart automatically generates the distribution of episode termination causes (Crash / Out-of-view / Sleep).

## Issues & Solutions

- **Issue:** Previously, recording multiple seeds sequentially in `logs.csv` distorted the plot generation because Matplotlib connected the episodes sequentially (e.g., jumping from episode 400 of seed 0 back to episode 0 of seed 1). This made the graph unreadable and failed to demonstrate the model's robustness.
- **Solution:** Integrated `pandas.groupby("Episode")` to aggregate all seeds on the same x-axis. The graphs and loss are now handled according to a true statistical distribution.

## Next Steps

- Finalize the required ablations (buffer size, soft vs hard target updates, or reward clipping) and ensure that their reports utilize these new charts.
- Run a complete training session over 5 seeds to generate a clean artifact ready for the final defense.
