##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## plot
##

import os
import argparse
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt


def smooth(values, window=20):
    if len(values) < window:
        return values
    return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()


def plot_all(csv_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    plots = {
        "Returns": ("Reward", "Episode Reward", "reward.png"),
        "Episode Length": ("Length", "Steps per Episode", "length.png"),
        "Epsilon": ("Epsilon", "Epsilon", "epsilon.png"),
        "Loss": ("Loss", "Avg Training Loss", "loss.png"),
    }

    for title, (col, ylabel, fname) in plots.items():
        if col not in df.columns:
            print(f"[PLOT] Column '{col}' not found, skipping.")
            continue

        # Drop NaN values for calculation (e.g., if loss is not present in initial episodes)
        col_data = df.dropna(subset=[col])
        if col_data.empty:
            continue

        # Group by Episode to aggregate multiple seeds present in the same CSV
        col_grouped = col_data.groupby("Episode")[col]
        episodes = col_grouped.mean().index.to_numpy()
        mean_vals = col_grouped.mean().to_numpy()
        std_vals = col_grouped.std().to_numpy()
        counts = col_grouped.count().to_numpy()

        # Calculate 95% Confidence Interval
        ci = np.zeros_like(mean_vals)
        for i in range(len(counts)):
            if counts[i] > 1:
                ci[i] = (
                    std_vals[i]
                    * st.t.ppf((1 + 0.95) / 2.0, counts[i] - 1)
                    / np.sqrt(counts[i])
                )

        fig, ax = plt.subplots(figsize=(10, 4))

        # If there's only 1 seed, also show the smoothed raw data to maintain original behavior
        if counts.max() == 1:
            ax.plot(episodes, mean_vals, alpha=0.3, color="steelblue", label="raw")
            ax.plot(
                episodes,
                smooth(mean_vals),
                color="steelblue",
                linewidth=2,
                label="smoothed (20ep)",
            )
        else:
            # For multiple seeds, plot the mean and 95% CI
            ax.plot(episodes, mean_vals, color="steelblue", linewidth=2, label="Mean")
            ax.fill_between(
                episodes,
                mean_vals - ci,
                mean_vals + ci,
                color="steelblue",
                alpha=0.3,
                label="95% CI",
            )

        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(output_dir, fname)
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[PLOT] Saved {out}")

    # Termination causes bar chart
    if "Termination" in df.columns:
        counts = df["Termination"].value_counts()
        if counts.empty:
            print("[PLOT] No termination data, skipping.")
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
            ax.set_title("Termination Causes")
            ax.set_xlabel("Cause")
            ax.set_ylabel("Count")
            plt.tight_layout()
            out = os.path.join(output_dir, "termination.png")
            plt.savefig(out, dpi=150)
            plt.close()
            print(f"[PLOT] Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to logs.csv")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for PNGs (defaults to same dir as csv)",
    )
    args = parser.parse_args()

    out_dir = args.output or os.path.dirname(os.path.abspath(args.csv))
    plot_all(args.csv, out_dir)
