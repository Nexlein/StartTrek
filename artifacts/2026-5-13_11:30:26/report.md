# Daily Report - StartTrek

**Date:** 2026-05-12

## Work Completed

### Task: Legacy Test 1 - Training with wind and evaluation with/without wind

- **Details:**
  - Trained the model with wind: `python train.py --seed 1 --wind 20`
  - Evaluated the model with wind: `python eval.py results/models/model_seed_1_ep_1000.pth --seed 1 --wind 20`
  - Evaluated the model without wind: `python eval.py results/models/model_seed_1_ep_1000.pth --seed 1`

- **Observations:**
  - With wind: The model struggles to counter the wind (maybe the wind is too strong?).
  - Without wind: The model has difficulties landing, it acts as if there is wind.

## Results & Metrics

- **See videos**

## Issues & Solutions

- **Issue:** The model overcompensates for the wind even when there is none (overfitting to the windy condition).
- **Solution:** (Unresolved in this test) We should implement Domain Randomization for wind during training.

## Next Steps

- Implement a `--random-wind` option during training so the model can adapt to both conditions.
