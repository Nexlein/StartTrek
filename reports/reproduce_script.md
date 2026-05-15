# Reproduce Script Fix

## Objective

The reproduction script `reproduce.sh` had bugs related to seed selection and evaluation loop execution as per the requirement ("Reproducibility: >= 5 seeds (0..4)").

## Changes Implemented

1. **Fixed Seed List**: Changed the `SEEDS` variable from `"0 5"` to `"0 1 2 3 4"` to train on exactly five different seeds as specified.
2. **Corrected Evaluation Loop**: Initially, the evaluation was done outside the loop using an uninitialized or leftover `$seed` variable, resulting in evaluating only one model or crashing. A `for seed in $SEEDS` loop has been added around the `eval.py` call so that all 5 trained models are properly evaluated and saved in the artifact folder.

## Impact

The `reproduce.sh` script now reliably executes the full end-to-end experiment suite across 5 different seeds and evaluates them accordingly. This brings the codebase up to standard with the project requirements for reproducibility.
