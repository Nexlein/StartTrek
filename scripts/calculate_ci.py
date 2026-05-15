import os
import argparse
import numpy as np
import scipy.stats as st


def calculate_ci(data, confidence=0.95):
    """Calculate the confidence interval for a given array of data."""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h


def analyze_artifact(artifact_path):
    logs_folder = os.path.join(artifact_path, "logs")
    if not os.path.exists(logs_folder):
        print(f"No logs folder found in {artifact_path}")
        return

    seed_scores = []
    # SEEDS 0 to 4
    for seed in range(5):
        csv_path = os.path.join(logs_folder, f"eval_scores_seed_{seed}.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found.")
            continue

        with open(csv_path, "r") as f:
            lines = f.readlines()[1:]  # skip header
            scores = [
                float(line.strip().split(",")[1]) for line in lines if line.strip()
            ]

        if scores:
            seed_scores.append(np.mean(scores))

    if not seed_scores:
        print("No evaluation scores found to analyze.")
        return

    mean, ci = calculate_ci(seed_scores)

    print("\n--- 95% Confidence Interval Analysis ---")
    print(f"Seeds analyzed: {len(seed_scores)}")
    for idx, score in enumerate(seed_scores):
        print(f"  Seed {idx} Mean Score: {score:.2f}")

    print(f"\nFinal Result: {mean:.2f} ± {ci:.2f} (95% CI)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate 95% CI for evaluation scores across seeds."
    )
    parser.add_argument(
        "--artifact",
        type=str,
        required=True,
        help="Path to the artifact folder containing the logs",
    )
    args = parser.parse_args()
    analyze_artifact(args.artifact)
