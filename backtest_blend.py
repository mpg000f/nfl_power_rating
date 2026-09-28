#!/usr/bin/env python3
"""Walk-forward backtest for early-season NFL prior blend curves.

Weekly ratings use only plays available before the predicted week. The script
compares the legacy linear blend with nonlinear curves and reports next-game
margin MAE/RMSE. It deliberately excludes manual QB adjustments so the result
isolates how quickly team performance should replace the preseason prior.
"""

import argparse
import contextlib
import io
import math

import numpy as np
import pandas as pd

from generate_preseason import build_preseason_ratings
from power_rating import (
    RatingConfig,
    add_success_column,
    calculate_game_stats,
    calculate_power_rating,
    calculate_raw_team_stats,
    filter_plays,
    load_pbp_data,
    opponent_adjust,
)


SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def weekly_in_season_ratings(pbp: pd.DataFrame, config: RatingConfig) -> pd.DataFrame:
    plays = add_success_column(filter_plays(pbp.copy()))
    raw = calculate_raw_team_stats(plays)
    games = calculate_game_stats(plays)
    adjusted = opponent_adjust(games, config)
    return calculate_power_rating(adjusted, raw, config)


def blend_weight(games: pd.Series, method: str, scale: float = 3.5,
                 cap: float = 0.95) -> pd.Series:
    if method == "legacy":
        return (games * 0.125).clip(upper=1.0)
    return games.apply(lambda g: min(cap, 1 - math.exp(-g / scale)))


def evaluate(seasons: list[int], defense_scales: list[float]) -> pd.DataFrame:
    schedule = pd.read_csv(SCHEDULE_URL, low_memory=False)
    config = RatingConfig()
    rows = []

    for season in seasons:
        pbp = load_pbp_data([season])
        season_games = schedule[
            (schedule["season"] == season)
            & schedule["home_score"].notna()
            & (schedule["game_type"] == "REG")
        ].copy()

        # No historical QB files are required; this isolates blend speed.
        with contextlib.redirect_stdout(io.StringIO()):
            preseason = build_preseason_ratings(season, qb_anchor_weight=0.0)
        preseason = preseason[["team", "power_rating", "off_pts", "def_pts"]].rename(
            columns={"power_rating": "pre_rating", "off_pts": "pre_off",
                     "def_pts": "pre_def"}
        )

        max_week = int(season_games["week"].max())
        for prediction_week in range(2, max_week + 1):
            prior_plays = pbp[pbp["week"] < prediction_week]
            if prior_plays.empty:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                current = weekly_in_season_ratings(prior_plays, config)
            current = current[["team", "power_rating", "off_pts", "def_pts",
                               "games_played"]]
            ratings = current.merge(preseason, on="team", how="left")

            methods = [("legacy", None)] + [("split", scale) for scale in defense_scales]
            games = season_games[season_games["week"] == prediction_week]
            for method, scale in methods:
                if method == "legacy":
                    weights = blend_weight(ratings["games_played"], "legacy")
                    values = (weights * ratings["power_rating"]
                              + (1 - weights) * ratings["pre_rating"].fillna(0.0))
                    label = "legacy"
                else:
                    off_w = blend_weight(ratings["games_played"], "exp", scale=5.5)
                    def_w = blend_weight(ratings["games_played"], "exp", scale=scale)
                    values = (
                        off_w * ratings["off_pts"]
                        + (1 - off_w) * ratings["pre_off"].fillna(0.0)
                        + def_w * ratings["def_pts"]
                        + (1 - def_w) * ratings["pre_def"].fillna(0.0)
                    )
                lookup = dict(zip(ratings["team"], values))
                if method != "legacy":
                    label = f"off5.5_def{scale:g}"
                for _, game in games.iterrows():
                    if game.home_team not in lookup or game.away_team not in lookup:
                        continue
                    prediction = (
                        lookup[game.home_team] - lookup[game.away_team]
                        + config.home_field_advantage
                    )
                    actual = game.home_score - game.away_score
                    rows.append({
                        "season": season,
                        "week": prediction_week,
                        "method": label,
                        "prediction": prediction,
                        "actual": actual,
                        "error": prediction - actual,
                    })

    results = pd.DataFrame(rows)
    results["window"] = np.where(results["week"] <= 5, "weeks_2_5", "full_season")
    full = results.assign(window="all_weeks")
    scored = pd.concat([results, full], ignore_index=True)
    summary = scored.groupby(["window", "method"]).agg(
        games=("error", "size"),
        mae=("error", lambda x: np.abs(x).mean()),
        rmse=("error", lambda x: np.sqrt(np.mean(x ** 2))),
        bias=("error", "mean"),
    ).reset_index()
    return summary.sort_values(["window", "mae"])


def main():
    parser = argparse.ArgumentParser(description="Backtest NFL blend curves")
    parser.add_argument("--start", type=int, default=2022)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--defense-scales", default="5.5,7,9,11,13")
    args = parser.parse_args()

    defense_scales = [float(value) for value in args.defense_scales.split(",")]
    summary = evaluate(list(range(args.start, args.end + 1)), defense_scales)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
