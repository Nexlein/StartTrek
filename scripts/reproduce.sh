#!/bin/bash

set -e

SEEDS="0 1 2 3 4"

echo "=== Reproducing StartTrek Experimental Pipeline ==="

echo ""
echo ">>> [1/4] Running Baselines (Random & Heuristic)..."
python3 -W ignore baseline.py

latest_artifact=$(ls -td artifacts/*/ | head -n 1)

if [ -z "$latest_artifact" ]; then
    echo "[ERROR] No artifact folders found."
    exit 1
fi

echo "[INFO] Using Artifact Directory: $latest_artifact"

echo ""
echo ">>> [2/4] Training DQN across $SEEDS..."
for seed in $SEEDS
do
    echo "--- Training Seed: $seed ---"
    python3 -W ignore train.py --seed $seed --artifact "$latest_artifact"
    echo ""
done

echo ">>> [3/4] Evaluating models..."
for seed in $SEEDS
do
    echo "--- Evaluating Seed: $seed ---"
    python3 -W ignore eval.py --seed $seed --artifact "$latest_artifact"
    echo ""
done

echo ">>> [4/4] Calculating 95% Confidence Interval..."
python3 -W ignore scripts/calculate_ci.py --artifact "$latest_artifact"

echo ""
echo "=== Pipeline Execution Completed Successfully ==="
