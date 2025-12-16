
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from analysis.metrics import (
    calculate_kappa_stats,
    calculate_avalanche_stats,
    calculate_microsleep_replay_rates,
    calculate_score_distributions,
)

def generate_report(processed_dir: Path, report_dir: Path):
    """
    Generates a markdown report with plots from the processed data.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    
    ticks_path = processed_dir / "ticks.parquet"
    episodes_path = processed_dir / "episodes.parquet"
    leaderboard_path = processed_dir / "leaderboard.parquet"
    
    ticks_df = pd.read_parquet(ticks_path) if ticks_path.exists() else pd.DataFrame()
    episodes_df = pd.read_parquet(episodes_path) if episodes_path.exists() else pd.DataFrame()
    leaderboard_df = pd.read_parquet(leaderboard_path) if leaderboard_path.exists() else pd.DataFrame()

    report_parts = ["# Analysis Report\n"]

    # --- Metadata ---
    report_parts.append("## Run Metadata\n")
    metadata = {
        "Processed Directory": f"`{processed_dir}`",
        "Agents": len(leaderboard_df),
        "Protocols": episodes_df["protocol"].nunique(),
        "Episodes": len(episodes_df),
        "Ticks": len(ticks_df),
    }
    report_parts.append(pd.DataFrame([metadata]).to_markdown(index=False))
    report_parts.append("\n")


    # --- Kappa ---
    report_parts.append("## Kappa Distribution\n")
    kappa_stats = calculate_kappa_stats(ticks_df)
    if kappa_stats:
        report_parts.append(pd.DataFrame([kappa_stats]).to_markdown(index=False))
        
        plt.figure()
        ticks_df["kappa"].hist(bins=50)
        plt.title("Kappa Distribution")
        plt.xlabel("Kappa")
        plt.ylabel("Frequency")
        kappa_plot_path = report_dir / "kappa_dist.png"
        plt.savefig(kappa_plot_path)
        plt.close()
        report_parts.append(f"\n![Kappa Distribution](kappa_dist.png)\n")
    else:
        report_parts.append("No kappa data found.\n")

    # --- Avalanches ---
    report_parts.append("## Avalanche Size Distribution\n")
    avalanche_stats = calculate_avalanche_stats(ticks_df)
    if avalanche_stats.get("count", 0) > 0:
        report_parts.append(pd.DataFrame([avalanche_stats]).to_markdown(index=False))

        avalanches = ticks_df[ticks_df["avalanche_size"] > 0]["avalanche_size"]
        plt.figure()
        avalanches.hist(bins=50, log=True)
        plt.title("Avalanche Size Distribution (Log Scale)")
        plt.xlabel("Avalanche Size")
        plt.ylabel("Frequency")
        avalanche_plot_path = report_dir / "avalanche_dist.png"
        plt.savefig(avalanche_plot_path)
        plt.close()
        report_parts.append(f"\n![Avalanche Distribution](avalanche_dist.png)\n")
    else:
        report_parts.append("No avalanche data found.\n")
        
    # --- Microsleep/Replay ---
    report_parts.append("## Microsleep and Replay Rates\n")
    rates_df = calculate_microsleep_replay_rates(ticks_df)
    if not rates_df.empty:
        report_parts.append(rates_df.to_markdown())
    else:
        report_parts.append("No microsleep/replay data found.")
    report_parts.append("\n")

    # --- Gating Sanity Checks ---
    report_parts.append("### Gating Sanity Checks\n")
    if not ticks_df.empty and "replay_active" in ticks_df.columns and "microsleep_active" in ticks_df.columns:
        replay_without_microsleep = ticks_df[ticks_df["replay_active"] & ~ticks_df["microsleep_active"]]
        microsleep_without_replay = ticks_df[ticks_df["microsleep_active"] & ~ticks_df["replay_active"]]
        
        checks = {
            "% replay_active & ~microsleep_active": (len(replay_without_microsleep) / len(ticks_df)) * 100 if len(ticks_df) > 0 else 0,
            "% microsleep_active & ~replay_active": (len(microsleep_without_replay) / len(ticks_df)) * 100 if len(ticks_df) > 0 else 0,
        }
        report_parts.append(pd.DataFrame([checks]).to_markdown(index=False))
    else:
        report_parts.append("Not enough data for gating sanity checks.")
    report_parts.append("\n")

    # --- Scores ---
    report_parts.append("## Protocol Score Distributions\n")
    score_dist = calculate_score_distributions(episodes_df)
    if not score_dist.empty:
        report_parts.append(score_dist.to_markdown())

        plt.figure()
        episodes_df.boxplot(column="score", by="protocol", grid=False)
        plt.title("Score Distribution by Protocol")
        plt.suptitle("") # remove default title
        plt.xlabel("Protocol")
        plt.ylabel("Score")
        score_plot_path = report_dir / "score_dist.png"
        plt.savefig(score_plot_path)
        plt.close()
        report_parts.append(f"\n![Score Distribution](score_dist.png)\n")
    else:
        report_parts.append("No score data found.\n")

    # --- Write Report ---
    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(report_parts))
    print(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate analysis report from processed data.")
    parser.add_argument("--processed-dir", type=Path, default=Path("analysis/processed"), help="Directory with Parquet tables.")
    parser.add_argument("--report-dir", type=Path, default=Path("analysis/report"), help="Directory to save the report and plots.")
    args = parser.parse_args()

    generate_report(args.processed_dir, args.report_dir)

if __name__ == "__main__":
    main()
