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
import matplotlib.pyplot as plt


def smooth(values, window=20):
    if len(values) < window:
        return values
    return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()


def plot_all(csv_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    episodes = df["Episode"].to_numpy()

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

        values = df[col].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(episodes, values, alpha=0.3, color="steelblue", label="raw")
        ax.plot(
            episodes,
            smooth(values),
            color="steelblue",
            linewidth=2,
            label="smoothed (20ep)",
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
