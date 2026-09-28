#!/usr/bin/env python3
"""
Update NFL Power Ratings

Blends in-season ratings with a preseason baseline early in the year.
The nonlinear curves react somewhat faster to offensive evidence while
regressing volatile defensive samples more heavily. Both retain a small
preseason prior throughout the year. With the backtested offensive scale:
  - 1 game:  16.6% in-season
  - 2 games: 30.5% in-season
  - 4 games: 51.7% in-season
  - 8 games: 76.7% in-season
  - 17+ games: capped at 95% in-season

If no preseason baseline exists (ratings_{season}_preseason.csv), uses
pure in-season ratings — so regenerating historical seasons is unaffected.

Usage:
    python update_ratings.py --season 2024
    python update_ratings.py --season 2024 --output ratings.csv
"""

import argparse
import math
import pandas as pd
from pathlib import Path
from datetime import datetime

RATINGS_DIR = Path(__file__).parent / "historical_ratings"


def current_season_weight(games_played: float, curve_scale: float = 5.5,
                          max_weight: float = 0.95) -> float:
    """Return a fast early-season blend that retains a small prior all year."""
    if games_played <= 0:
        return 0.0
    return min(max_weight, 1 - math.exp(-games_played / curve_scale))


def load_current_qb_context(season: int) -> pd.DataFrame | None:
    """Load the current expected starter and QB value for weekly lineup changes."""
    path = Path(__file__).parent / f"qb_adjustments_{season}.csv"
    if not path.exists():
        return None
    qb = pd.read_csv(path).rename(columns={
        "Team": "team",
        "Quarterback": "current_qb",
        "Point Spread Rating QB": "current_qb_value",
    })
    needed = {"team", "current_qb", "current_qb_value"}
    if not needed.issubset(qb.columns):
        return None
    qb["current_qb_value"] = pd.to_numeric(qb["current_qb_value"], errors="coerce")
    return qb[["team", "current_qb", "current_qb_value"]]


def blend_with_preseason(in_season: pd.DataFrame, season: int,
                         offense_curve_scale: float = 5.5,
                         defense_curve_scale: float = 9.0,
                         max_weight: float = 0.95) -> pd.DataFrame:
    """Blend in-season power ratings with the preseason baseline."""
    baseline_path = RATINGS_DIR / f"ratings_{season}_preseason.csv"
    if not baseline_path.exists():
        return in_season

    preseason = pd.read_csv(baseline_path)
    blend_cols = [c for c in ("power_rating", "off_pts", "def_pts")
                  if c in preseason.columns and c in in_season.columns]
    renames = {c: f"pre_{c}" for c in blend_cols}
    context_cols = [c for c in ("preseason_qb", "preseason_qb_value")
                    if c in preseason.columns]
    df = in_season.merge(preseason[["team"] + blend_cols + context_cols].rename(columns=renames),
                         on="team", how="left")

    # Teams without a preseason baseline use pure in-season ratings
    off_w = df["games_played"].apply(
        lambda games: current_season_weight(games, offense_curve_scale, max_weight)
    )
    def_w = df["games_played"].apply(
        lambda games: current_season_weight(games, defense_curve_scale, max_weight)
    )
    has_prior = df["pre_power_rating"].notna()
    df["offense_blend_weight"] = off_w.where(has_prior, 1.0).round(3)
    df["defense_blend_weight"] = def_w.where(has_prior, 1.0).round(3)
    # Retain the legacy field name for downstream consumers; it represents offense.
    df["blend_weight"] = df["offense_blend_weight"]

    if "off_pts" in blend_cols:
        df["off_pts"] = (
            df["offense_blend_weight"] * df["off_pts"]
            + (1 - df["offense_blend_weight"]) * df["pre_off_pts"].fillna(0.0)
        ).round(3)
    if "def_pts" in blend_cols:
        df["def_pts"] = (
            df["defense_blend_weight"] * df["def_pts"]
            + (1 - df["defense_blend_weight"]) * df["pre_def_pts"].fillna(0.0)
        ).round(3)
    df["power_rating"] = (df["off_pts"] + df["def_pts"]).round(3)

    # Apply an immediate starter-change adjustment relative to the preseason
    # expectation. It naturally fades as the new QB's observed offense becomes
    # a larger part of the team rating, avoiding permanent double counting.
    current_qb = load_current_qb_context(season)
    df["qb_lineup_adjustment"] = 0.0
    if current_qb is not None and "preseason_qb_value" in df.columns:
        df = df.merge(current_qb, on="team", how="left")
        qb_delta = (df["current_qb_value"] - df["preseason_qb_value"]).fillna(0.0)
        df["qb_lineup_adjustment"] = (
            qb_delta * (1 - df["offense_blend_weight"])
        ).round(3)
        df["off_pts"] = (df["off_pts"] + df["qb_lineup_adjustment"]).round(3)
        df["power_rating"] = (df["off_pts"] + df["def_pts"]).round(3)

    df = df.drop(columns=[f"pre_{c}" for c in blend_cols])
    df = df.sort_values("power_rating", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    print(f"  Blended with preseason baseline "
          f"(offense {df['offense_blend_weight'].min():.2f}-"
          f"{df['offense_blend_weight'].max():.2f}; defense "
          f"{df['defense_blend_weight'].min():.2f}-"
          f"{df['defense_blend_weight'].max():.2f})")
    return df


def main():
    # Keep the data-loader dependency out of lightweight model/unit-test imports.
    from power_rating import RatingConfig, run_power_ratings

    parser = argparse.ArgumentParser(description="Update NFL Power Ratings")
    parser.add_argument('--season', type=int, required=True, help='Season year')
    parser.add_argument('--output', type=str, help='Output path (default: historical_ratings/ratings_{season}.csv)')
    parser.add_argument('--blend-scale', type=float, default=5.5,
                        help='Offensive in-season blend scale (default: 5.5)')
    parser.add_argument('--defense-blend-scale', type=float, default=9.0,
                        help='Defensive in-season blend scale (default: 9.0)')
    parser.add_argument('--max-in-season-weight', type=float, default=0.95,
                        help='Maximum weight on current-season data (default: 0.95)')

    args = parser.parse_args()

    output_dir = Path(__file__).parent / "historical_ratings"
    output_dir.mkdir(exist_ok=True)

    output_path = args.output or output_dir / f"ratings_{args.season}.csv"

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting NFL ratings update for {args.season}")

    config = RatingConfig()
    ratings = run_power_ratings(args.season, config)

    ratings = blend_with_preseason(
        ratings, args.season, args.blend_scale, args.defense_blend_scale,
        args.max_in_season_weight
    )

    ratings.to_csv(output_path, index=False)

    top_team = ratings.iloc[0]
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Update complete")
    print(f"  Teams rated: {len(ratings)}")
    print(f"  #1 team: {top_team['team']} ({top_team['power_rating']:.2f})")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
