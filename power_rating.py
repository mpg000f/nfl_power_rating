#!/usr/bin/env python3
"""
NFL Power Rating System
=======================
Opponent-adjusted EPA and Success Rate ratings.

Power Rating = expected point spread vs average NFL team
Based on EPA margin × plays per game, with success rate adjustment.

Success Rate Definition (traditional):
- 1st down: 50% of yards to go
- 2nd down: 70% of yards to go
- 3rd/4th down: 100% of yards to go (or first down/TD)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
try:
    import nfl_data_py as nfl
except ImportError:
    nfl = None


@dataclass
class RatingConfig:
    """Configuration for NFL power ratings."""
    # Opponent adjustment iterations
    opp_adjust_iterations: int = 15
    convergence_threshold: float = 0.001

    # Rating weights (EPA vs Success Rate)
    weight_epa: float = 0.6
    weight_success_rate: float = 0.4

    # Scaling factors (calibrated to FPI-like range, +15% stretch)
    epa_to_points: float = 46.0
    success_to_points: float = 23.0

    # Target standard deviation for normalized ratings (ensures consistent scale across years)
    target_std: float = 4.5

    # Home field advantage (points)
    home_field_advantage: float = 2.5

    # Minimum plays to be rated
    min_plays: int = 100


def calculate_success(row: pd.Series) -> Optional[bool]:
    """
    Determine if a play was successful using traditional success rate.

    - 1st down: gain 50% of yards to go
    - 2nd down: gain 70% of yards to go
    - 3rd/4th down: gain 100% (convert or TD)
    """
    down = row.get('down')
    ydstogo = row.get('ydstogo')
    yards_gained = row.get('yards_gained')
    first_down = row.get('first_down_rush') or row.get('first_down_pass') or row.get('first_down_penalty')
    td = row.get('touchdown')

    if pd.isna(down) or pd.isna(ydstogo) or pd.isna(yards_gained):
        return None

    # TD is always success
    if td == 1:
        return True

    # First down gained is always success
    if first_down == 1:
        return True

    if down == 1:
        return yards_gained >= (ydstogo * 0.50)
    elif down == 2:
        return yards_gained >= (ydstogo * 0.70)
    elif down in [3, 4]:
        return yards_gained >= ydstogo

    return None


def load_pbp_data(seasons: list[int]) -> pd.DataFrame:
    """Load play-by-play data for specified seasons."""
    print(f"Loading play-by-play data for seasons: {seasons}")

    all_pbp = []
    for season in seasons:
        try:
            if nfl is None:
                raise ImportError("nfl_data_py is unavailable")
            pbp = nfl.import_pbp_data([season], downcast=False)
            all_pbp.append(pbp)
            print(f"  Loaded {len(pbp):,} plays for {season} via nfl_data_py")
        except Exception as e:
            # Fallback to direct nflverse download
            print(f"  nfl_data_py failed for {season}, trying direct download...")
            url = f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet'
            try:
                pbp = pd.read_parquet(url)
                all_pbp.append(pbp)
                print(f"  Loaded {len(pbp):,} plays for {season} via direct download")
            except Exception as e2:
                print(f"  Failed to load {season}: {e2}")
                continue

    if not all_pbp:
        raise ValueError("No play-by-play data could be loaded")

    combined = pd.concat(all_pbp, ignore_index=True)
    print(f"  Total: {len(combined):,} plays")
    return combined


def filter_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Filter to relevant plays (run/pass, no penalties, etc.)."""
    # Keep only run and pass plays
    mask = (
        (pbp['play_type'].isin(['run', 'pass'])) &
        (pbp['posteam'].notna()) &
        (pbp['defteam'].notna()) &
        (pbp['epa'].notna()) &
        (pbp['down'].notna())
    )

    filtered = pbp[mask].copy()
    print(f"  Filtered to {len(filtered):,} run/pass plays")
    return filtered


def add_success_column(pbp: pd.DataFrame) -> pd.DataFrame:
    """Add success rate column based on traditional definition."""
    print("Calculating success rate...")
    pbp['success'] = pbp.apply(calculate_success, axis=1)
    pbp['success'] = pbp['success'].astype(float)

    success_rate = pbp['success'].mean()
    print(f"  Overall success rate: {success_rate:.1%}")
    return pbp


def calculate_raw_team_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Calculate raw (non-adjusted) EPA and success rate per team."""
    print("Calculating raw team statistics...")

    # Offensive stats
    off_stats = pbp.groupby('posteam').agg(
        raw_off_epa=('epa', 'mean'),
        raw_off_success=('success', 'mean'),
        off_plays=('epa', 'count')
    ).reset_index().rename(columns={'posteam': 'team'})

    # Defensive stats (opponent's offense)
    def_stats = pbp.groupby('defteam').agg(
        raw_def_epa=('epa', 'mean'),
        raw_def_success=('success', 'mean'),
        def_plays=('epa', 'count')
    ).reset_index().rename(columns={'defteam': 'team'})

    # Merge
    stats = off_stats.merge(def_stats, on='team', how='outer')
    stats['total_plays'] = stats['off_plays'] + stats['def_plays']

    print(f"  Calculated raw stats for {len(stats)} teams")
    return stats


def calculate_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-game EPA and success rate for opponent adjustment."""
    print("Calculating game-level statistics...")

    # Group by game and team
    game_off = pbp.groupby(['game_id', 'posteam', 'defteam']).agg(
        off_epa=('epa', 'mean'),
        off_success=('success', 'mean'),
        off_plays=('epa', 'count'),
        home=('posteam', lambda x: (pbp.loc[x.index, 'home_team'] == x.iloc[0]).iloc[0] if len(x) > 0 else False)
    ).reset_index()

    game_off.columns = ['game_id', 'team', 'opponent', 'off_epa', 'off_success', 'off_plays', 'is_home']

    game_def = pbp.groupby(['game_id', 'defteam', 'posteam']).agg(
        def_epa=('epa', 'mean'),
        def_success=('success', 'mean'),
        def_plays=('epa', 'count')
    ).reset_index()

    game_def.columns = ['game_id', 'team', 'opponent', 'def_epa', 'def_success', 'def_plays']

    # Merge offensive and defensive game stats
    games = game_off.merge(game_def, on=['game_id', 'team', 'opponent'], how='outer')

    print(f"  Calculated {len(games)} team-game records")
    return games


def opponent_adjust(games: pd.DataFrame, config: RatingConfig) -> pd.DataFrame:
    """
    Iteratively adjust EPA and success rate for opponent strength.

    For each team:
    - Adjusted Off EPA = Raw Off EPA - Avg(Opponent Def EPA)
    - Adjusted Def EPA = Raw Def EPA - Avg(Opponent Off EPA)
    """
    print(f"Running opponent adjustment ({config.opp_adjust_iterations} iterations)...")

    teams = games['team'].unique()

    # Initialize with raw averages
    team_ratings = games.groupby('team').agg(
        off_epa=('off_epa', 'mean'),
        def_epa=('def_epa', 'mean'),
        off_success=('off_success', 'mean'),
        def_success=('def_success', 'mean'),
        games_played=('game_id', 'nunique')
    ).reset_index()

    # League averages
    league_off_epa = team_ratings['off_epa'].mean()
    league_def_epa = team_ratings['def_epa'].mean()
    league_off_success = team_ratings['off_success'].mean()
    league_def_success = team_ratings['def_success'].mean()

    for iteration in range(config.opp_adjust_iterations):
        prev_ratings = team_ratings.copy()

        # Create lookup for opponent ratings
        opp_lookup = team_ratings.set_index('team')[['off_epa', 'def_epa', 'off_success', 'def_success']].to_dict('index')

        adjusted_stats = []

        for team in teams:
            team_games = games[games['team'] == team]

            # Calculate opponent-adjusted stats
            adj_off_epa_list = []
            adj_def_epa_list = []
            adj_off_success_list = []
            adj_def_success_list = []

            for _, game in team_games.iterrows():
                opp = game['opponent']
                if opp not in opp_lookup:
                    continue

                opp_stats = opp_lookup[opp]

                # Adjust offensive EPA: raw - (opp_def - league_avg_def)
                opp_def_adj = opp_stats['def_epa'] - league_def_epa
                adj_off_epa_list.append(game['off_epa'] - opp_def_adj)

                # Adjust defensive EPA: raw - (opp_off - league_avg_off)
                opp_off_adj = opp_stats['off_epa'] - league_off_epa
                adj_def_epa_list.append(game['def_epa'] - opp_off_adj)

                # Same for success rate
                opp_def_success_adj = opp_stats['def_success'] - league_def_success
                adj_off_success_list.append(game['off_success'] - opp_def_success_adj)

                opp_off_success_adj = opp_stats['off_success'] - league_off_success
                adj_def_success_list.append(game['def_success'] - opp_off_success_adj)

            if adj_off_epa_list:
                adjusted_stats.append({
                    'team': team,
                    'adj_off_epa': np.mean(adj_off_epa_list),
                    'adj_def_epa': np.mean(adj_def_epa_list),
                    'adj_off_success': np.mean(adj_off_success_list),
                    'adj_def_success': np.mean(adj_def_success_list),
                    'games_played': len(adj_off_epa_list)
                })

        team_ratings = pd.DataFrame(adjusted_stats)
        # Rename for iteration tracking
        team_ratings = team_ratings.rename(columns={
            'adj_off_epa': 'off_epa',
            'adj_def_epa': 'def_epa',
            'adj_off_success': 'off_success',
            'adj_def_success': 'def_success'
        })

        # Check convergence
        if iteration > 0:
            merged = prev_ratings.merge(team_ratings, on='team', suffixes=('_prev', '_new'))
            avg_change = np.abs(merged['off_epa_new'] - merged['off_epa_prev']).mean()

            if iteration % 3 == 0 or avg_change < config.convergence_threshold:
                print(f"    Iteration {iteration + 1}: avg change = {avg_change:.6f}")

            if avg_change < config.convergence_threshold:
                print(f"  Converged at iteration {iteration + 1}")
                break

    # Rename back to adjusted columns
    team_ratings = team_ratings.rename(columns={
        'off_epa': 'adj_off_epa',
        'def_epa': 'adj_def_epa',
        'off_success': 'adj_off_success',
        'def_success': 'adj_def_success'
    })

    return team_ratings


def calculate_power_rating(adjusted: pd.DataFrame, raw_stats: pd.DataFrame, config: RatingConfig) -> pd.DataFrame:
    """
    Calculate final power rating as expected point spread vs average team.

    Power Rating = (EPA margin per play) × (plays per game)
    This gives us expected point differential, i.e., spread vs average.
    """
    print("Calculating power ratings...")

    # Merge adjusted and raw stats
    df = adjusted.merge(raw_stats[['team', 'raw_off_epa', 'raw_def_epa', 'raw_off_success', 'raw_def_success']], on='team')

    # EPA margins
    df['adj_epa_margin'] = df['adj_off_epa'] - df['adj_def_epa']
    df['raw_epa_margin'] = df['raw_off_epa'] - df['raw_def_epa']

    # Success rate margins
    df['adj_success_margin'] = df['adj_off_success'] - df['adj_def_success']
    df['raw_success_margin'] = df['raw_off_success'] - df['raw_def_success']

    # Convert EPA margin to points
    # Calibrated to match FPI-like scale (best teams ~+8-10, worst ~-8-10)
    df['epa_points'] = df['adj_epa_margin'] * config.epa_to_points

    # Success rate contribution
    df['success_points'] = df['adj_success_margin'] * config.success_to_points

    # Combined power rating (weighted average)
    df['power_rating'] = (
        config.weight_epa * df['epa_points'] +
        config.weight_success_rate * df['success_points']
    )

    # Offensive/defensive point components (off_pts + def_pts = power_rating)
    off_comp = (config.weight_epa * config.epa_to_points * df['adj_off_epa'] +
                config.weight_success_rate * config.success_to_points * df['adj_off_success'])
    def_comp = -(config.weight_epa * config.epa_to_points * df['adj_def_epa'] +
                 config.weight_success_rate * config.success_to_points * df['adj_def_success'])
    df['off_pts'] = off_comp - off_comp.mean()
    df['def_pts'] = def_comp - def_comp.mean()

    # Center so average team = 0
    df['power_rating'] = df['power_rating'] - df['power_rating'].mean()

    # Normalize to target standard deviation for consistent scale across years
    current_std = df['power_rating'].std()
    if current_std > 0:
        scale = config.target_std / current_std
        df['power_rating'] = df['power_rating'] * scale
        df['off_pts'] = df['off_pts'] * scale
        df['def_pts'] = df['def_pts'] * scale

    # Rank
    df = df.sort_values('power_rating', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    return df


def calculate_records(season: int) -> pd.DataFrame:
    """Calculate win-loss records from schedule data."""
    print("Calculating team records...")

    # Map historical team abbreviations to current ones (PBP data uses current)
    TEAM_MAP = {
        'OAK': 'LV',   # Oakland Raiders -> Las Vegas Raiders
        'STL': 'LA',   # St. Louis Rams -> Los Angeles Rams
        'SD': 'LAC',   # San Diego Chargers -> Los Angeles Chargers
    }

    try:
        if nfl is None:
            raise ImportError("nfl_data_py is unavailable")
        schedules = nfl.import_schedules([season])
        games = schedules[schedules['home_score'].notna()].copy()
    except Exception as e:
        # Fallback to direct download
        print(f"  nfl_data_py schedules failed, trying direct download...")
        url = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
        schedules = pd.read_csv(url, low_memory=False)
        schedules = schedules[schedules['season'] == season]
        games = schedules[schedules['home_score'].notna()].copy()

    records = {}
    for _, game in games.iterrows():
        home = TEAM_MAP.get(game['home_team'], game['home_team'])
        away = TEAM_MAP.get(game['away_team'], game['away_team'])
        home_score = game['home_score']
        away_score = game['away_score']

        if home not in records:
            records[home] = {'wins': 0, 'losses': 0, 'ties': 0}
        if away not in records:
            records[away] = {'wins': 0, 'losses': 0, 'ties': 0}

        if home_score > away_score:
            records[home]['wins'] += 1
            records[away]['losses'] += 1
        elif away_score > home_score:
            records[away]['wins'] += 1
            records[home]['losses'] += 1
        else:
            records[home]['ties'] += 1
            records[away]['ties'] += 1

    records_df = pd.DataFrame([
        {'team': team, 'wins': r['wins'], 'losses': r['losses'], 'ties': r['ties']}
        for team, r in records.items()
    ])

    # Create record string (e.g., "15-4" or "14-2-1")
    def format_record(row):
        if row['ties'] > 0:
            return f"{row['wins']}-{row['losses']}-{row['ties']}"
        return f"{row['wins']}-{row['losses']}"

    records_df['record'] = records_df.apply(format_record, axis=1)

    return records_df[['team', 'record', 'wins', 'losses', 'ties']]


def run_power_ratings(season: int, config: RatingConfig = None) -> pd.DataFrame:
    """Run the full NFL power rating pipeline."""
    if config is None:
        config = RatingConfig()

    # Load data
    pbp = load_pbp_data([season])

    # Filter to run/pass plays
    pbp = filter_plays(pbp)

    # Add success column
    pbp = add_success_column(pbp)

    # Calculate raw stats
    raw_stats = calculate_raw_team_stats(pbp)

    # Calculate game-level stats
    games = calculate_game_stats(pbp)

    # Opponent adjustment
    adjusted = opponent_adjust(games, config)

    # Calculate final ratings
    ratings = calculate_power_rating(adjusted, raw_stats, config)

    # Calculate records
    records = calculate_records(season)
    ratings = ratings.merge(records, on='team', how='left')

    # Select output columns
    output_cols = [
        'rank', 'team', 'power_rating', 'record', 'wins', 'losses', 'ties',
        # Point components (sum to power_rating)
        'off_pts', 'def_pts',
        # Adjusted stats
        'adj_off_epa', 'adj_def_epa', 'adj_epa_margin',
        'adj_off_success', 'adj_def_success', 'adj_success_margin',
        # Raw stats
        'raw_off_epa', 'raw_def_epa', 'raw_epa_margin',
        'raw_off_success', 'raw_def_success', 'raw_success_margin',
        'games_played'
    ]

    result = ratings[output_cols].copy()

    # Round for display
    for col in ['power_rating', 'off_pts', 'def_pts',
                'adj_off_epa', 'adj_def_epa', 'adj_epa_margin',
                'raw_off_epa', 'raw_def_epa', 'raw_epa_margin']:
        result[col] = result[col].round(3)
    for col in ['adj_off_success', 'adj_def_success', 'adj_success_margin',
                'raw_off_success', 'raw_def_success', 'raw_success_margin']:
        result[col] = result[col].round(3)

    return result


def display_ratings(df: pd.DataFrame, top_n: int = 32) -> None:
    """Display ratings in readable format."""
    print("\n" + "=" * 120)
    print(f"NFL POWER RATINGS (Expected Spread vs Average Team)")
    print("=" * 120)

    display_cols = ['rank', 'team', 'power_rating',
                    'adj_off_epa', 'adj_def_epa',
                    'adj_off_success', 'adj_def_success',
                    'games_played']

    display_df = df[display_cols].head(top_n).copy()
    display_df['adj_off_success'] = (display_df['adj_off_success'] * 100).round(1).astype(str) + '%'
    display_df['adj_def_success'] = (display_df['adj_def_success'] * 100).round(1).astype(str) + '%'

    print(display_df.to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NFL Power Ratings")
    parser.add_argument('--season', type=int, default=2024, help='Season year')
    parser.add_argument('--output', type=str, help='Output CSV path')

    args = parser.parse_args()

    config = RatingConfig()
    ratings = run_power_ratings(args.season, config)

    display_ratings(ratings)

    if args.output:
        ratings.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")
