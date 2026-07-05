#!/usr/bin/env python3
"""
Update NFL Power Ratings

Blends in-season ratings with a preseason baseline early in the year.
Each team's blend weight = min(games_played * 0.125, 1.0), so:
  - 1 game:  12.5% in-season, 87.5% preseason
  - 4 games: 50% in-season, 50% preseason
  - 8+ games: 100% in-season

If no preseason baseline exists (ratings_{season}_preseason.csv), uses
pure in-season ratings — so regenerating historical seasons is unaffected.

Usage:
    python update_ratings.py --season 2024
    python update_ratings.py --season 2024 --output ratings.csv
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

from power_rating import RatingConfig, run_power_ratings

RATINGS_DIR = Path(__file__).parent / "historical_ratings"


def blend_with_preseason(in_season: pd.DataFrame, season: int,
                         per_game_step: float = 0.125) -> pd.DataFrame:
    """Blend in-season power ratings with the preseason baseline."""
    baseline_path = RATINGS_DIR / f"ratings_{season}_preseason.csv"
    if not baseline_path.exists():
        return in_season

    preseason = pd.read_csv(baseline_path)[["team", "power_rating"]]
    preseason = preseason.rename(columns={"power_rating": "preseason_rating"})
    df = in_season.merge(preseason, on="team", how="left")

    # Teams without a preseason baseline use pure in-season ratings
    w = (df["games_played"] * per_game_step).clip(upper=1.0)
    df["blend_weight"] = w.where(df["preseason_rating"].notna(), 1.0).round(3)
    df["power_rating"] = (
        df["blend_weight"] * df["power_rating"]
        + (1 - df["blend_weight"]) * df["preseason_rating"].fillna(0.0)
    ).round(3)

    df = df.drop(columns=["preseason_rating"])
    df = df.sort_values("power_rating", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    print(f"  Blended with preseason baseline "
          f"(in-season weights {df['blend_weight'].min():.2f}-{df['blend_weight'].max():.2f})")
    return df


def main():
    parser = argparse.ArgumentParser(description="Update NFL Power Ratings")
    parser.add_argument('--season', type=int, required=True, help='Season year')
    parser.add_argument('--output', type=str, help='Output path (default: historical_ratings/ratings_{season}.csv)')

    args = parser.parse_args()

    output_dir = Path(__file__).parent / "historical_ratings"
    output_dir.mkdir(exist_ok=True)

    output_path = args.output or output_dir / f"ratings_{args.season}.csv"

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting NFL ratings update for {args.season}")

    config = RatingConfig()
    ratings = run_power_ratings(args.season, config)

    ratings = blend_with_preseason(ratings, args.season)

    ratings.to_csv(output_path, index=False)

    top_team = ratings.iloc[0]
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Update complete")
    print(f"  Teams rated: {len(ratings)}")
    print(f"  #1 team: {top_team['team']} ({top_team['power_rating']:.2f})")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
