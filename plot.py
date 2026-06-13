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


def plot_eval(csv_path: str, output_dir: str, label: str = ""):
    """
    Generate a reward distribution histogram for a single eval run.

    Args:
        csv_path: Path to the eval CSV (columns: Episode, Reward).
        output_dir: Directory where the chart PNG is saved.
        label: Human-readable condition label (e.g. "No Wind", "Wind 15").
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    rewards = df["Reward"].to_numpy()
    mean = rewards.mean()
    std = rewards.std()
    n = len(rewards)
    ci = st.t.ppf(0.975, n - 1) * std / np.sqrt(n) if n > 1 else 0

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rewards, bins=20, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(mean, color="red", linewidth=2, label=f"Mean = {mean:.1f} ± {ci:.1f}")
    ax.axvline(mean - ci, color="red", linewidth=1, linestyle=":", alpha=0.6)
    ax.axvline(mean + ci, color="red", linewidth=1, linestyle=":", alpha=0.6)
    ax.axvline(200, color="orange", linewidth=1.5, linestyle="--", label="Target ≥ 200")
    ax.set_title(f"Eval Reward Distribution — {label} (n={n})")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_label = label.lower().replace(" ", "_")
    out = os.path.join(output_dir, f"chart_eval_{safe_label}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[PLOT] Saved {out}")


def plot_eval_comparison(csv_paths: dict, output_dir: str):
    """
    Generate a side-by-side bar chart comparing mean rewards across eval conditions.

    Args:
        csv_paths: Mapping of label → CSV path, e.g. {"No Wind": "...", "Wind 15": "..."}.
        output_dir: Directory where the chart PNG is saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    labels, means, cis, colors = [], [], [], ["steelblue", "tomato"]

    for label, path in csv_paths.items():
        if not os.path.exists(path):
            print(f"[PLOT] Missing {path}, skipping comparison.")
            return
        df = pd.read_csv(path)
        rewards = df["Reward"].to_numpy()
        n = len(rewards)
        mean = rewards.mean()
        ci = st.t.ppf(0.975, n - 1) * rewards.std() / np.sqrt(n) if n > 1 else 0
        labels.append(label)
        means.append(mean)
        cis.append(ci)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        x,
        means,
        yerr=cis,
        capsize=8,
        color=colors[: len(labels)],
        edgecolor="black",
        alpha=0.85,
        error_kw={"elinewidth": 2},
    )
    ax.axhline(200, color="orange", linewidth=1.5, linestyle="--", label="Target ≥ 200")

    for bar, mean, ci in zip(bars, means, cis):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ci + 3,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Reward (100 episodes)")
    ax.set_title("Eval Comparison — No Wind vs Wind (mean ± 95% CI)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    out = os.path.join(output_dir, "chart_eval_comparison.png")
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
