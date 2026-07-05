#!/usr/bin/env python3
"""Generate NFL power ratings for all available seasons (2000-2025)."""

from pathlib import Path
from power_rating import RatingConfig, run_power_ratings

def main():
    output_dir = Path(__file__).parent / "historical_ratings"
    output_dir.mkdir(exist_ok=True)

    config = RatingConfig()

    # nflverse has data from 2000 onwards (1999 has issues)
    for season in range(2000, 2026):
        print(f"\n{'='*60}")
        print(f"Processing {season} season")
        print('='*60)

        try:
            ratings = run_power_ratings(season, config)
            output_path = output_dir / f"ratings_{season}.csv"
            ratings.to_csv(output_path, index=False)

            top = ratings.iloc[0]
            bottom = ratings.iloc[-1]
            print(f"  Saved: {output_path.name}")
            print(f"  Best: {top['team']} ({top['power_rating']:+.1f})")
            print(f"  Worst: {bottom['team']} ({bottom['power_rating']:+.1f})")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

if __name__ == "__main__":
    main()
