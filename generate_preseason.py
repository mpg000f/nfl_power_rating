#!/usr/bin/env python3
"""
Generate NFL Preseason Power Ratings

Methodology:
  1. Split each prior season's rating into offensive and defensive point components
  2. Blend the two prior seasons (75% last year, 25% two years ago), component-wise
  3. Reweight offense 60% / defense 40% -- offense is more repeatable year to year
  4. Anchor projected offense 25% to the centered absolute QB rating
  5. Rescale to the historical in-season spread (std ~4.5) so QB points mean the same
     thing as in-season points
  6. Apply manual QB adjustments (points) from qb_adjustments_{year}.csv
  7. Recenter so the average team = 0
  8. Shrink the final spread toward the mean (default x0.80) -- preseason blends
     only correlate ~0.5 with the following season, so the full in-season spread
     overstates preseason certainty (backtested optimum is ~0.5; 0.8 keeps the
     scale readable and credits the QB layer the backtest can't see)

The QB file also supports two optional per-team columns:
  - "Off t1 weight": override the offensive blend weight on t-1 (e.g. 0.95 to
    trust a second-year QB's breakout and mostly ignore his rookie-year offense)
  - "Situation adj": extra points for non-QB continuity shocks (coaching change,
    scheme reset), kept separate from the QB swap adjustment

QB adjustment file format (same as the old nfl-qb-power-ratings.csv):
  Team,Quarterback,Point Spread Rating QB,YoY adjustment

Usage:
    python generate_preseason.py --target 2026
    python generate_preseason.py --target 2026 --weights 0.75,0.25
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

RATINGS_DIR = Path(__file__).parent / "historical_ratings"

# Keep in sync with RatingConfig in power_rating.py
WEIGHT_EPA = 0.6
WEIGHT_SUCCESS = 0.4
EPA_TO_POINTS = 46.0
SUCCESS_TO_POINTS = 23.0
TARGET_STD = 4.5

STAT_COLS = [
    "adj_off_epa", "adj_def_epa", "adj_epa_margin",
    "adj_off_success", "adj_def_success", "adj_success_margin",
]

RAW_COLS = [
    "raw_off_epa", "raw_def_epa", "raw_epa_margin",
    "raw_off_success", "raw_def_success", "raw_success_margin",
]


def load_season_ratings(year: int) -> pd.DataFrame | None:
    path = RATINGS_DIR / f"ratings_{year}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return add_components(df)


def add_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose the season's power rating into offensive and defensive point
    components using the in-season formula, scaled so off_pts + def_pts
    reproduces the published (normalized) power_rating.
    """
    off = (WEIGHT_EPA * EPA_TO_POINTS * df["adj_off_epa"]
           + WEIGHT_SUCCESS * SUCCESS_TO_POINTS * df["adj_off_success"])
    dfn = -(WEIGHT_EPA * EPA_TO_POINTS * df["adj_def_epa"]
            + WEIGHT_SUCCESS * SUCCESS_TO_POINTS * df["adj_def_success"])
    off = off - off.mean()
    dfn = dfn - dfn.mean()

    total = off + dfn
    scale = df["power_rating"].std() / total.std() if total.std() > 0 else 1.0
    df["off_pts"] = off * scale
    df["def_pts"] = dfn * scale
    return df


def blend_ratings(target_year: int, weights: tuple = (0.75, 0.25)) -> pd.DataFrame:
    """Blend the two prior seasons into a preseason baseline."""
    years = [target_year - 1, target_year - 2]
    seasons = []
    actual_weights = []

    for yr, w in zip(years, weights):
        df = load_season_ratings(yr)
        if df is not None:
            seasons.append((yr, df))
            actual_weights.append(w)
        else:
            print(f"  Warning: no ratings found for {yr}, skipping")

    if not seasons:
        raise ValueError(f"No historical ratings found for {years}")

    total_w = sum(actual_weights)
    actual_weights = [w / total_w for w in actual_weights]

    all_teams = pd.concat([df[["team"]] for _, df in seasons]).drop_duplicates("team")
    blended = all_teams.copy()

    cols = ["power_rating", "off_pts", "def_pts"] + STAT_COLS
    for col in cols:
        blended[col] = 0.0
        blended[f"{col}_wsum"] = 0.0

    for (yr, df), w in zip(seasons, actual_weights):
        merged = blended.merge(df[["team"] + cols], on="team", how="left",
                               suffixes=("", f"_{yr}"))
        for col in cols:
            col_yr = f"{col}_{yr}"
            has_data = merged[col_yr].notna().values
            blended.loc[has_data, col] += merged.loc[has_data, col_yr].values * w
            blended.loc[has_data, f"{col}_wsum"] += w

    for col in cols:
        wsum = blended[f"{col}_wsum"]
        blended[col] = blended[col] / wsum.where(wsum > 0, 1.0)
        blended.drop(columns=[f"{col}_wsum"], inplace=True)

    print(f"  Blended {len(blended)} teams from {[yr for yr, _ in seasons]} "
          f"with weights {[f'{w:.2f}' for w in actual_weights]}")
    return blended


def load_qb_file(target_year: int) -> pd.DataFrame | None:
    path = Path(__file__).parent / f"qb_adjustments_{target_year}.csv"
    if not path.exists():
        print(f"  Warning: {path.name} not found, skipping QB adjustments")
        return None
    qb = pd.read_csv(path)
    qb = qb.rename(columns={
        "Team": "team",
        "Quarterback": "quarterback",
        "Point Spread Rating QB": "qb_value",
        "YoY adjustment": "qb_adj",
        "Situation adj": "situation_adj",
        "Off t1 weight": "off_t1_weight",
    })
    for col, default in [("qb_value", 0.0), ("qb_adj", 0.0),
                         ("situation_adj", 0.0), ("off_t1_weight", None)]:
        if col not in qb.columns:
            qb[col] = default
        else:
            qb[col] = pd.to_numeric(qb[col], errors="coerce")
    qb["qb_adj"] = qb["qb_adj"].fillna(0.0)
    qb["situation_adj"] = qb["situation_adj"].fillna(0.0)
    qb["qb_value"] = qb["qb_value"].fillna(qb["qb_value"].mean())
    return qb


def apply_off_weight_overrides(blended: pd.DataFrame, qb: pd.DataFrame,
                               target_year: int, default_w: float) -> pd.DataFrame:
    """Re-blend the offensive component for teams with a custom t-1 weight."""
    overrides = qb[qb["off_t1_weight"].notna()][["team", "off_t1_weight"]]
    if overrides.empty:
        return blended

    s1 = load_season_ratings(target_year - 1)
    s2 = load_season_ratings(target_year - 2)
    if s1 is None or s2 is None:
        print("  Warning: need both prior seasons for off-weight overrides, skipping")
        return blended
    s1 = s1.set_index("team")["off_pts"]
    s2 = s2.set_index("team")["off_pts"]

    for _, row in overrides.iterrows():
        team, w = row["team"], row["off_t1_weight"]
        if team in s1.index and team in s2.index:
            new_off = w * s1[team] + (1 - w) * s2[team]
            old_off = blended.loc[blended["team"] == team, "off_pts"].iloc[0]
            blended.loc[blended["team"] == team, "off_pts"] = new_off
            print(f"  Off blend override {team}: t-1 weight {w:.2f} "
                  f"(off {old_off:+.2f} -> {new_off:+.2f})")
    return blended


def apply_qb_adjustments(blended: pd.DataFrame, qb: pd.DataFrame | None) -> pd.DataFrame:
    """Add manual QB and situation adjustments (in points) to the blended ratings."""
    if qb is None:
        blended["qb_adj"] = 0.0
        blended["quarterback"] = None
        return blended

    df = blended.merge(qb[["team", "quarterback", "qb_adj", "situation_adj"]],
                       on="team", how="left")
    missing = df[df["qb_adj"].isna()]["team"].tolist()
    if missing:
        print(f"  Warning: no QB adjustment for {missing}, using 0")
    df["qb_adj"] = df["qb_adj"].fillna(0.0) + df["situation_adj"].fillna(0.0)

    print(f"  Applied QB/situation adjustments (sum {df['qb_adj'].sum():+.1f}, "
          f"range {df['qb_adj'].min():+.1f} to {df['qb_adj'].max():+.1f})")
    df["power_rating"] = df["power_rating"] + df["qb_adj"]
    # QB/situation shocks are offensive by nature — keep components summing
    if "off_final" in df.columns:
        df["off_final"] = df["off_final"] + df["qb_adj"]
    return df


def apply_qb_anchor(blended: pd.DataFrame, qb: pd.DataFrame | None,
                    anchor_weight: float) -> pd.DataFrame:
    """Shrink the team offensive projection toward its starting-QB value.

    QB values are centered before blending, so they change the distribution of
    offensive strength without moving the league average. This makes the
    absolute QB rating an actual model input rather than a descriptive column.
    """
    if qb is None or anchor_weight <= 0:
        blended["qb_value"] = np.nan
        blended["qb_value_centered"] = 0.0
        return blended

    context = qb[["team", "qb_value"]].copy()
    context["qb_value_centered"] = context["qb_value"] - context["qb_value"].mean()
    df = blended.merge(context, on="team", how="left")
    df["qb_value_centered"] = df["qb_value_centered"].fillna(0.0)
    df["off_final"] = (
        (1 - anchor_weight) * df["off_final"]
        + anchor_weight * df["qb_value_centered"]
    )
    df["power_rating"] = df["off_final"] + df["def_final"]
    print(f"  Anchored offense {anchor_weight:.0%} to absolute QB value")
    return df


def build_preseason_ratings(target_year: int,
                            weights: tuple = (0.75, 0.25),
                            off_weight: float = 0.60,
                            shrink: float = 0.80,
                            qb_anchor_weight: float = 0.25) -> pd.DataFrame:
    print(f"\nGenerating preseason {target_year} NFL ratings...")

    blended = blend_ratings(target_year, weights)

    qb = load_qb_file(target_year)
    if qb is not None:
        blended = apply_off_weight_overrides(blended, qb, target_year, weights[0])

    # Reweight offense vs defense: offense is more repeatable year to year.
    # (x2 keeps a 50/50 split identical to the plain off+def sum.)
    # off_final/def_final track the displayed components through every
    # subsequent transform so they always sum to power_rating.
    blended["off_final"] = 2 * off_weight * blended["off_pts"]
    blended["def_final"] = 2 * (1 - off_weight) * blended["def_pts"]
    blended["power_rating"] = blended["off_final"] + blended["def_final"]
    print(f"  Reweighted components: {off_weight:.0%} offense / {1 - off_weight:.0%} defense")

    # Absolute QB quality stabilizes the noisier team-offense projection.
    blended = apply_qb_anchor(blended, qb, qb_anchor_weight)

    # Rescale the blend back to the in-season spread before adding QB points,
    # so a +2 QB adjustment is worth the same as 2 points in-season.
    prior_stds = []
    for yr in [target_year - 1, target_year - 2]:
        s = load_season_ratings(yr)
        if s is not None:
            prior_stds.append(s["power_rating"].std())
    target_std = sum(prior_stds) / len(prior_stds) if prior_stds else 4.5

    raw_std = blended["power_rating"].std()
    scale = target_std / raw_std if raw_std > 0 else 1.0
    blended["off_final"] = (blended["off_final"] - blended["off_final"].mean()) * scale
    blended["def_final"] = (blended["def_final"] - blended["def_final"].mean()) * scale
    blended["power_rating"] = blended["off_final"] + blended["def_final"]
    print(f"  Rescaled blend: std {raw_std:.2f} -> {target_std:.2f}")

    df = apply_qb_adjustments(blended, qb)

    # Recenter so the average team is 0 (QB adjustments don't sum to zero)
    df["off_final"] = df["off_final"] - df["off_final"].mean()
    df["def_final"] = df["def_final"] - df["def_final"].mean()

    # Compress the preseason spread toward the mean
    df["off_final"] = df["off_final"] * shrink
    df["def_final"] = df["def_final"] * shrink
    df["power_rating"] = df["off_final"] + df["def_final"]
    print(f"  Shrunk final spread by x{shrink:.2f} (std {df['power_rating'].std():.2f})")

    df = df.sort_values("power_rating", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    output = pd.DataFrame({
        "rank": df["rank"],
        "team": df["team"],
        "power_rating": df["power_rating"].round(3),
        "record": "Preseason",
        "wins": 0,
        "losses": 0,
        "ties": 0,
        "off_pts": df["off_final"].round(2),
        "def_pts": df["def_final"].round(2),
        "preseason_qb": df.get("quarterback"),
        "preseason_qb_value": df.get("qb_value"),
    })
    for col in STAT_COLS:
        output[col] = df[col].round(3)
    for col in RAW_COLS:
        output[col] = None
    output["games_played"] = 0

    return output


def main():
    parser = argparse.ArgumentParser(description="Generate NFL preseason power ratings")
    parser.add_argument("--target", type=int, default=2026, help="Target season year")
    parser.add_argument("--weights", type=str, default="0.75,0.25",
                        help="Blend weights for years t-1,t-2 (default: 0.75,0.25)")
    parser.add_argument("--off-weight", type=float, default=0.60,
                        help="Offense share of the off/def reweight (default: 0.60)")
    parser.add_argument("--qb-anchor-weight", type=float, default=0.25,
                        help="Share of preseason offense anchored to absolute QB value (default: 0.25)")
    parser.add_argument("--shrink", type=float, default=0.80,
                        help="Final spread compression factor (default: 0.80)")
    args = parser.parse_args()

    weights = tuple(float(w) for w in args.weights.split(","))
    if len(weights) != 2:
        print("Error: --weights must have exactly 2 values")
        return

    ratings = build_preseason_ratings(args.target, weights, args.off_weight,
                                      args.shrink, args.qb_anchor_weight)

    # Current-display ratings (overwritten by in-season ratings once games start)
    output_path = RATINGS_DIR / f"ratings_{args.target}.csv"
    ratings.to_csv(output_path, index=False)

    # Permanent preseason baseline (never overwritten)
    baseline_path = RATINGS_DIR / f"ratings_{args.target}_preseason.csv"
    ratings.to_csv(baseline_path, index=False)

    print(f"\nSaved preseason {args.target} ratings to: {output_path}")
    print(f"Saved preseason baseline to: {baseline_path}")
    print("\nTop 10:")
    print(ratings[["rank", "team", "power_rating"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
