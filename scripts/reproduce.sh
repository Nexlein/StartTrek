#!/bin/bash

set -e

echo "Reproducing StartTrek tests..."

echo ""
echo "[1/2] Starting Baselines (Random & Heuristic)..."
python3 baseline.py

latest_artifact=$(ls -td artifacts/*/ | head -n 1)

if [ -z "$latest_artifact" ]; then
    echo "Erreur : Aucun dossier d'artifact trouvé."
    exit 1
fi

echo "Using artifact folder: $latest_artifact"

echo ""
echo "[2/2] Starting training on 5 seeds (0 to 4)..."
for seed in {0..4}
do
    echo "Training in progress with SEED : $seed"
    python train.py --seed $seed --artifact "$latest_artifact"
done

echo ""
echo "Reproduction completed."
