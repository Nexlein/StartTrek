#!/bin/bash

set -e

SEEDS="0 5"

echo "Reproducing StartTrek tests..."

echo ""
echo "[1/3] Starting Baselines (Random & Heuristic)..."
python3 -W ignore baseline.py

latest_artifact=$(ls -td artifacts/*/ | head -n 1)

if [ -z "$latest_artifact" ]; then
    echo "Error: No artifacts folders find."
    exit 1
fi

echo "Using artifact folder: $latest_artifact"

echo ""
echo "[2/3] Starting training on 5 seeds (0 to 4)..."
for seed in $SEEDS
do
    echo "Training in progress with SEED : $seed"
    python3 -W ignore train.py --seed $seed --artifact "$latest_artifact"
done

echo "[3/3] Starting evaluating on 5 seeds (0 to 4)..."
for seed in $SEEDS
do
    echo "Evaluating in progress with SEED : $seed"
    python3 -W ignore eval.py --seed $seed --artifact "$latest_artifact"
    echo ""
done

echo "Reproduction completed."
