
import pandas as pd
from typing import Dict

def calculate_kappa_stats(ticks_df: pd.DataFrame) -> Dict[str, float]:
    """Calculates statistics for the kappa values."""
    if 'kappa' not in ticks_df.columns or ticks_df['kappa'].empty:
        return {}
    
    return {
        "mean": ticks_df["kappa"].mean(),
        "std": ticks_df["kappa"].std(),
        "min": ticks_df["kappa"].min(),
        "max": ticks_df["kappa"].max(),
    }

def calculate_avalanche_stats(ticks_df: pd.DataFrame) -> Dict[str, float]:
    """Calculates statistics for avalanche sizes."""
    if 'avalanche_size' not in ticks_df.columns:
        return {}
        
    avalanches = ticks_df[ticks_df["avalanche_size"] > 0]["avalanche_size"]
    if avalanches.empty:
        return {"count": 0}
        
    return {
        "count": len(avalanches),
        "mean": avalanches.mean(),
        "std": avalanches.std(),
        "min": avalanches.min(),
        "max": avalanches.max(),
    }

def calculate_microsleep_replay_rates(ticks_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the rates of microsleep and replay, grouped by protocol."""
    if ticks_df.empty or "protocol" not in ticks_df.columns:
        return pd.DataFrame()
        
    rates = ticks_df.groupby("protocol")[["microsleep_active", "replay_active"]].mean()
    rates = rates.rename(columns={"microsleep_active": "microsleep_rate", "replay_active": "replay_rate"})
    return rates

def calculate_score_distributions(episodes_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates score distributions per protocol."""
    if 'score' not in episodes_df.columns or 'protocol' not in episodes_df.columns or episodes_df.empty:
        return pd.DataFrame()

    return episodes_df.groupby("protocol")["score"].describe()

def run_all_metrics(processed_dir: str) -> Dict:
    """
    Runs all metric calculations and returns a dictionary of results.
    """
    ticks_path = f"{processed_dir}/ticks.parquet"
    episodes_path = f"{processed_dir}/episodes.parquet"

    try:
        ticks_df = pd.read_parquet(ticks_path)
    except FileNotFoundError:
        ticks_df = pd.DataFrame()

    try:
        episodes_df = pd.read_parquet(episodes_path)
    except FileNotFoundError:
        episodes_df = pd.DataFrame()
        
    metrics = {
        "kappa_stats": calculate_kappa_stats(ticks_df),
        "avalanche_stats": calculate_avalanche_stats(ticks_df),
        "microsleep_replay_rates": calculate_microsleep_replay_rates(ticks_df),
        "score_distributions": calculate_score_distributions(episodes_df).to_dict(),
    }
    
    return metrics
