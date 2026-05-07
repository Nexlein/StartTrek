#!/bin/bash

set -e

echo "Reproducing StartTrek tests..."

echo ""
echo "[1/2] Starting Baselines (Random & Heuristic)..."
python3 baseline.py

echo ""
echo "[2/2] Starting training on 5 seeds (0 to 4)..."
for seed in {0..4}
do
    echo "Training in progress with SEED : $seed"
    python train.py --seed $seed
done

echo ""
echo "Reproduction completed."
