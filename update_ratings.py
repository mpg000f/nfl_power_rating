#!/usr/bin/env python3
"""
Update NFL Power Ratings

Usage:
    python update_ratings.py --season 2024
    python update_ratings.py --season 2024 --output ratings.csv
"""

import argparse
from pathlib import Path
from datetime import datetime

from power_rating import RatingConfig, run_power_ratings


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

    ratings.to_csv(output_path, index=False)

    top_team = ratings.iloc[0]
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Update complete")
    print(f"  Teams rated: {len(ratings)}")
    print(f"  #1 team: {top_team['team']} ({top_team['power_rating']:.2f})")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
