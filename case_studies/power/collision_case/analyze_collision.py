"""Analysis and plotting tools for collision detection experiments.

Reproduces the analysis from the original collision platform with:
- Annual collision frequency trends
- Reward vs safety trade-offs
- Agent-specific collision attribution
- Comparison between shared and independent reward schemes

Usage:
    python -m collision_case.analyze_collision \
        --shared results/shared_reward.csv \
        --independent results/independent_reward.csv \
        --output results/
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

# Constants
STEPS_PER_YEAR = 24 * 365
ROLLING_WINDOW = 10  # years for smoothing


def read_and_group(filepath: Path, steps_per_year: int = STEPS_PER_YEAR) -> Tuple:
    """Read CSV log and group by year.

    Args:
        filepath: Path to CSV log file
        steps_per_year: Timesteps per year (default: 24 * 365 = 8760)

    Returns:
        Tuple of (avg_reward, avg_safety, safety_freq) pandas Series indexed by year
    """
    df = pd.read_csv(filepath)
    df = df.reset_index(drop=True)

    # Newer runs (RLlib with multiple env runners) may include duplicated rows per
    # timestep or multiple env instances writing to the same file. Deduplicate to
    # one row per unique (env_instance, episode, timestep) before grouping.
    if {"timestep", "episode"}.issubset(df.columns):
        key_cols = ["episode", "timestep"]
        if "env_instance" in df.columns:
            key_cols = ["env_instance", "episode", "timestep"]
        df = (
            df.groupby(key_cols, as_index=False)
            .agg(reward_sum=("reward_sum", "mean"), safety_sum=("safety_sum", "mean"))
        )

    if "timestep" in df.columns:
        # Interpret timestep as absolute simulation time; convert to "year index".
        # (timestep starts at 0 or 1 depending on env; this is robust either way.)
        df["year"] = (df["timestep"].astype(int) // steps_per_year).astype(int)
    else:
        # Legacy: fall back to row index.
        df["year"] = df.index // steps_per_year

    grouped = df.groupby("year")
    avg_reward = grouped["reward_sum"].mean()
    avg_safety = grouped["safety_sum"].mean()
    safety_freq = grouped["safety_sum"].apply(lambda x: np.mean(x != 0))

    return avg_reward, avg_safety, safety_freq


def plot_annual_reward(
    shared_path: Path,
    independent_path: Path,
    output_dir: Path,
    last_n_years: int = 100,
):
    """Plot annual mean reward (last N years).

    Args:
        shared_path: Path to shared reward CSV
        independent_path: Path to independent reward CSV
        output_dir: Output directory for plots
        last_n_years: Number of recent years to plot
    """
    avg_reward_shared, _, _ = read_and_group(shared_path)
    avg_reward_indep, _, _ = read_and_group(independent_path)

    n_years = min(len(avg_reward_shared), len(avg_reward_indep))
    years = np.arange(1, n_years + 1)

    # Plot last N years
    plt.figure(figsize=(9, 5))
    years_plot = years[-last_n_years:]
    shared_plot = avg_reward_shared.iloc[-last_n_years:]
    indep_plot = avg_reward_indep.iloc[-last_n_years:]

    plt.plot(
        years_plot, shared_plot, label="Shared Reward", color="#0066CC", lw=2, marker="o", markersize=3
    )
    plt.plot(
        years_plot,
        indep_plot,
        label="Independent Reward",
        color="#FF6600",
        lw=2,
        marker="s",
        markersize=3,
    )
    plt.title(f"Annual Mean Reward (Last {last_n_years} Years)", fontsize=16, fontweight="bold")
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Mean Reward", fontsize=14)
    plt.legend(fontsize=13)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "annual_mean_reward_last100.png", dpi=300)
    plt.close()


def plot_annual_collision_log(
    shared_path: Path, independent_path: Path, output_dir: Path
):
    """Plot annual mean collision metric (log scale).

    Args:
        shared_path: Path to shared reward CSV
        independent_path: Path to independent reward CSV
        output_dir: Output directory for plots
    """
    _, avg_safety_shared, _ = read_and_group(shared_path)
    _, avg_safety_indep, _ = read_and_group(independent_path)

    n_years = min(len(avg_safety_shared), len(avg_safety_indep))
    years = np.arange(1, n_years + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(
        years,
        avg_safety_shared.iloc[:n_years],
        label="Shared Reward",
        color="#0066CC",
        lw=2,
        marker="o",
        markersize=3,
    )
    plt.plot(
        years,
        avg_safety_indep.iloc[:n_years],
        label="Independent Reward",
        color="#FF6600",
        lw=2,
        marker="s",
        markersize=3,
    )
    plt.yscale("log")
    plt.title("Annual Mean Collision Metric (Log Scale)", fontsize=16, fontweight="bold")
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Mean Safety/Collision (log)", fontsize=14)
    plt.legend(fontsize=13)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "annual_mean_collision_log.png", dpi=300)
    plt.close()


def plot_collision_frequency(
    shared_path: Path, independent_path: Path, output_dir: Path
):
    """Plot annual collision frequency.

    Args:
        shared_path: Path to shared reward CSV
        independent_path: Path to independent reward CSV
        output_dir: Output directory for plots
    """
    _, _, safety_freq_shared = read_and_group(shared_path)
    _, _, safety_freq_indep = read_and_group(independent_path)

    n_years = min(len(safety_freq_shared), len(safety_freq_indep))
    years = np.arange(1, n_years + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(
        years,
        safety_freq_shared.iloc[:n_years],
        label="Shared Reward",
        color="#0066CC",
        lw=2,
        marker="o",
        markersize=3,
    )
    plt.plot(
        years,
        safety_freq_indep.iloc[:n_years],
        label="Independent Reward",
        color="#FF6600",
        lw=2,
        marker="s",
        markersize=3,
    )
    plt.title("Annual Frequency of Non-zero Collision Metric", fontsize=16, fontweight="bold")
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Frequency of Collision (Proportion)", fontsize=14)
    plt.legend(fontsize=13)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "annual_collision_frequency.png", dpi=300)
    plt.close()


def plot_collision_frequency_smoothed(
    shared_path: Path, independent_path: Path, output_dir: Path, window: int = ROLLING_WINDOW
):
    """Plot smoothed collision frequency trends.

    Args:
        shared_path: Path to shared reward CSV
        independent_path: Path to independent reward CSV
        output_dir: Output directory for plots
        window: Rolling window size for smoothing
    """
    _, _, safety_freq_shared = read_and_group(shared_path)
    _, _, safety_freq_indep = read_and_group(independent_path)

    # Smoothing
    shared_smooth = safety_freq_shared.rolling(window=window, min_periods=1, center=True).mean()
    indep_smooth = safety_freq_indep.rolling(window=window, min_periods=1, center=True).mean()

    n_years = min(len(shared_smooth), len(indep_smooth))
    years = np.arange(1, n_years + 1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(years, shared_smooth.iloc[:n_years], color="#cce6ff", alpha=0.6)
    plt.plot(years, shared_smooth.iloc[:n_years], color="#0066CC", lw=2, label="Shared Reward (smoothed)")
    plt.fill_between(years, indep_smooth.iloc[:n_years], color="#ffd4b2", alpha=0.5)
    plt.plot(
        years, indep_smooth.iloc[:n_years], color="#FF6600", lw=2, label="Independent Reward (smoothed)"
    )
    plt.title(
        f"Annual Collision Frequency (Smoothed, {window}-year window)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Frequency of Collision (Proportion)", fontsize=14)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "collision_frequency_smoothed.png", dpi=300)
    plt.close()


def plot_agent_attribution(
    shared_path: Path, independent_path: Path, output_dir: Path, agent_ids: list = ["MG1", "MG2", "MG3"]
):
    """Plot per-agent collision attribution.

    Args:
        shared_path: Path to shared reward CSV
        independent_path: Path to independent reward CSV
        output_dir: Output directory for plots
        agent_ids: List of agent IDs
    """
    df_shared = pd.read_csv(shared_path)
    df_indep = pd.read_csv(independent_path)

    # Compute totals per agent
    shared_totals = [df_shared[f"{aid}_safety_sum"].sum() for aid in agent_ids]
    indep_totals = [df_indep[f"{aid}_safety_sum"].sum() for aid in agent_ids]

    # Compute percentages
    shared_pct = [100 * x / sum(shared_totals) for x in shared_totals] if sum(shared_totals) > 0 else [0] * len(agent_ids)
    indep_pct = [100 * x / sum(indep_totals) for x in indep_totals] if sum(indep_totals) > 0 else [0] * len(agent_ids)

    # Print table
    print("\nAgent Collision Attribution:")
    print(f"{'Agent':<8} | {'Independent':<15} | {'Independent %':<13} | {'Shared':<15} | {'Shared %':<10}")
    print("-" * 80)
    for i, aid in enumerate(agent_ids):
        print(
            f"{aid:<8} | {indep_totals[i]:>15.2f} | {indep_pct[i]:>12.1f}% | "
            f"{shared_totals[i]:>15.2f} | {shared_pct[i]:>9.1f}%"
        )

    # Plot stacked bars
    colors = ["#4E79A7", "#F28E2B", "#76B7B2"]
    fig, ax = plt.subplots(figsize=(9, 2.5))

    # Stacked bars
    lefts = [0, 0]
    for idx, color in enumerate(colors):
        ax.barh([0], indep_totals[idx], left=lefts[0], color=color, edgecolor="k", height=0.38)
        lefts[0] += indep_totals[idx]
        ax.barh([1], shared_totals[idx], left=lefts[1], color=color, edgecolor="k", height=0.38)
        lefts[1] += shared_totals[idx]

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Independent", "Shared"])
    ax.set_xlabel("Total Collision Metric")
    ax.set_title("Agent Contribution to Total Collision Metric by Scenario")

    # Annotate bars with percentages
    for i, vals in enumerate([indep_pct, indep_totals]):
        xpos = 0
        totals = indep_totals if i == 0 else shared_totals
        pcts = indep_pct if i == 0 else shared_pct
        for idx, (tot, pct) in enumerate(zip(totals, pcts)):
            if tot > 0:
                color = "white" if pct > 20 else "black"
                ax.text(
                    xpos + tot / 2,
                    i,
                    f"{agent_ids[idx]} ({pct:.0f}%)",
                    va="center",
                    ha="center",
                    fontsize=10,
                    color=color,
                )
            xpos += tot

    plt.tight_layout()
    plt.savefig(output_dir / "agent_attribution_comparison.png", dpi=300)
    plt.close()


def generate_summary_stats(
    shared_path: Path, independent_path: Path, output_dir: Path, last_n_years: int = 100
):
    """Generate summary statistics table.

    Args:
        shared_path: Path to shared reward CSV
        independent_path: Path to independent reward CSV
        output_dir: Output directory for outputs
        last_n_years: Number of recent years for statistics
    """
    _, _, safety_freq_shared = read_and_group(shared_path)
    _, _, safety_freq_indep = read_and_group(independent_path)

    # Extract last N years
    shared_last = safety_freq_shared.iloc[-last_n_years:]
    indep_last = safety_freq_indep.iloc[-last_n_years:]

    # Create table
    table = pd.DataFrame(
        {
            "Scheme": ["Shared Reward", "Independent Reward"],
            f"Mean Frequency (final {last_n_years}y)": [
                shared_last.mean(),
                indep_last.mean(),
            ],
            f"Std Dev (final {last_n_years}y)": [
                shared_last.std(),
                indep_last.std(),
            ],
            f"Min (final {last_n_years}y)": [
                shared_last.min(),
                indep_last.min(),
            ],
            f"Max (final {last_n_years}y)": [
                shared_last.max(),
                indep_last.max(),
            ],
        }
    )

    print("\n" + "=" * 80)
    print("COLLISION FREQUENCY SUMMARY (Final 100 Years)")
    print("=" * 80)
    print(table.to_string(index=False))
    print("=" * 80)

    # Save to CSV
    table.to_csv(output_dir / "collision_frequency_summary_100y.csv", index=False)


def main():
    """Main analysis entry point."""
    parser = argparse.ArgumentParser(description="Analyze collision detection results")

    parser.add_argument(
        "--shared", type=Path, required=True, help="Path to shared reward CSV log"
    )
    parser.add_argument(
        "--independent", type=Path, required=True, help="Path to independent reward CSV log"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results"), help="Output directory for plots"
    )
    parser.add_argument(
        "--last-n-years", type=int, default=100, help="Number of recent years to analyze"
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("Generating collision analysis plots...")

    print("  1. Annual mean reward (last 100 years)...")
    plot_annual_reward(args.shared, args.independent, args.output, args.last_n_years)

    print("  2. Annual collision metric (log scale)...")
    plot_annual_collision_log(args.shared, args.independent, args.output)

    print("  3. Annual collision frequency...")
    plot_collision_frequency(args.shared, args.independent, args.output)

    print("  4. Smoothed collision frequency...")
    plot_collision_frequency_smoothed(args.shared, args.independent, args.output)

    print("  5. Agent attribution comparison...")
    plot_agent_attribution(args.shared, args.independent, args.output)

    print("  6. Summary statistics...")
    generate_summary_stats(args.shared, args.independent, args.output, args.last_n_years)

    print(f"\nAll plots saved to: {args.output}")


if __name__ == "__main__":
    main()
