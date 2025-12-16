
import argparse
import json
from pathlib import Path
import pandas as pd

def extract_data(tournament_dir: Path, output_dir: Path):
    """
    Extracts data from a tournament run into Parquet tables.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up old files
    for f in output_dir.glob("*.parquet"):
        f.unlink()

    # Leaderboard
    leaderboard_path = tournament_dir / "leaderboard.json"
    if leaderboard_path.exists():
        leaderboard_df = pd.read_json(leaderboard_path)
        # The 'protocols' column is dict-like, which can be tricky.
        # We'll leave it as is for now, but for a real analysis db,
        # this would be normalized.
        leaderboard_df.to_parquet(output_dir / "leaderboard.parquet")
        print(f"Leaderboard table saved to {output_dir / 'leaderboard.parquet'}")
    else:
        pd.DataFrame().to_parquet(output_dir / "leaderboard.parquet")

    # Episodes
    episode_summaries = []
    summary_files = sorted(list(tournament_dir.glob("agents/**/summary.json")))
    for summary_file in summary_files:
        summary_data = json.loads(summary_file.read_text())
        episode_summaries.append(summary_data)
    
    if episode_summaries:
        episodes_df = pd.DataFrame(episode_summaries)
        episodes_df.to_parquet(output_dir / "episodes.parquet")
        print(f"Episodes table saved to {output_dir / 'episodes.parquet'}")
    else:
        pd.DataFrame().to_parquet(output_dir / "episodes.parquet")


    # Ticks
    tick_files = sorted(list(tournament_dir.glob("agents/**/ticks.jsonl")))
    if tick_files:
        all_ticks = []
        for tick_file in tick_files:
            # Add identifiers from the path
            agent_id = tick_file.parent.parent.name
            protocol = tick_file.parent.name
            
            with open(tick_file, "r") as f:
                for line in f:
                    tick_data = json.loads(line)
                    tick_data["agent_id"] = agent_id
                    tick_data["protocol"] = protocol
                    all_ticks.append(tick_data)
        
        if all_ticks:
            ticks_df = pd.DataFrame(all_ticks)
            ticks_df.to_parquet(output_dir / "ticks.parquet")
            print(f"Ticks table saved to {output_dir / 'ticks.parquet'}")
        else:
            pd.DataFrame().to_parquet(output_dir / "ticks.parquet")
    else:
        pd.DataFrame().to_parquet(output_dir / "ticks.parquet")

def main():
    parser = argparse.ArgumentParser(description="Extract tournament data into Parquet tables.")
    parser.add_argument("tournament_dir", type=Path, help="Directory of the tournament run.")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/processed"), help="Directory to save the Parquet tables.")
    args = parser.parse_args()

    extract_data(args.tournament_dir, args.output_dir)

if __name__ == "__main__":
    main()
