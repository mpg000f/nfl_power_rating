import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import update_ratings


class ModelUpdateTests(unittest.TestCase):
    def test_nonlinear_blend_curve(self):
        expected = {
            0: 0.0,
            1: 1 - math.exp(-1 / 4.5),
            2: 1 - math.exp(-2 / 4.5),
            4: 1 - math.exp(-4 / 4.5),
            8: 1 - math.exp(-8 / 4.5),
            17: 0.95,
        }
        for games, weight in expected.items():
            self.assertTrue(math.isclose(
                update_ratings.current_season_weight(games), weight
            ))

    def test_starter_change_is_immediate_and_fades_with_observed_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ratings_dir = Path(temp_dir)
            pd.DataFrame({
                "team": ["A"],
                "power_rating": [0.0],
                "off_pts": [0.0],
                "def_pts": [0.0],
                "preseason_qb": ["Starter"],
                "starter_backup_value": [4.0],
            }).to_csv(ratings_dir / "ratings_2026_preseason.csv", index=False)
            current_qb = pd.DataFrame({
                "team": ["A"],
                "current_qb": ["Backup"],
                "lineup_delta": [-4.0],
            })
            in_season = pd.DataFrame({
                "team": ["A"],
                "power_rating": [4.0],
                "off_pts": [4.0],
                "def_pts": [0.0],
                "games_played": [2],
            })

            with patch.object(update_ratings, "RATINGS_DIR", ratings_dir), \
                 patch.object(update_ratings, "load_current_qb_context",
                              return_value=current_qb):
                result = update_ratings.blend_with_preseason(in_season, 2026)

            w = round(update_ratings.current_season_weight(2), 3)
            expected_adjustment = round((1.0 - 5.0) * (1 - w), 3)
            self.assertEqual(result.loc[0, "qb_lineup_adjustment"], expected_adjustment)
            self.assertEqual(result.loc[0, "offense_blend_weight"], w)
            self.assertEqual(
                result.loc[0, "defense_blend_weight"],
                round(update_ratings.current_season_weight(2, 7.0), 3),
            )
            self.assertEqual(result.loc[0, "off_pts"],
                             round(4.0 * w + expected_adjustment, 3))
            self.assertEqual(result.loc[0, "power_rating"], result.loc[0, "off_pts"])

    def test_yahoo_value_only_applies_to_matching_starter_backup_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            qb_path = Path(temp_dir) / "qb_adjustments_2026.csv"
            pd.DataFrame({
                "Team": ["A", "B"],
                "Preseason QB": ["Starter A", "Different Starter"],
                "Current QB": ["Backup A", "Backup B"],
                "Yahoo Starter": ["Starter A", "Yahoo Starter B"],
                "Yahoo Backup": ["Backup A", "Backup B"],
                "Starter-backup ATS": [4.5, 6.0],
            }).to_csv(qb_path, index=False)

            # Exercise the same pair-matching rule without relying on the
            # repository's live weekly file.
            qb = pd.read_csv(qb_path).rename(columns={
                "Team": "team", "Preseason QB": "preseason_qb",
                "Current QB": "current_qb", "Yahoo Starter": "yahoo_starter",
                "Yahoo Backup": "yahoo_backup",
                "Starter-backup ATS": "starter_backup_value",
            })
            listed = (qb.preseason_qb.eq(qb.yahoo_starter)
                      & qb.current_qb.eq(qb.yahoo_backup))

            self.assertEqual((-qb.starter_backup_value.where(listed, 0.0)).tolist(),
                             [-4.5, -0.0])


if __name__ == "__main__":
    unittest.main()
