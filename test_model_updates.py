import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import update_ratings
from generate_preseason import apply_qb_anchor


class ModelUpdateTests(unittest.TestCase):
    def test_nonlinear_blend_curve(self):
        expected = {
            0: 0.0,
            1: 1 - math.exp(-1 / 5.5),
            2: 1 - math.exp(-2 / 5.5),
            4: 1 - math.exp(-4 / 5.5),
            8: 1 - math.exp(-8 / 5.5),
            17: 0.95,
        }
        for games, weight in expected.items():
            self.assertTrue(math.isclose(
                update_ratings.current_season_weight(games), weight
            ))

    def test_absolute_qb_value_anchors_offense_without_moving_average(self):
        teams = pd.DataFrame({
            "team": ["A", "B"],
            "off_final": [2.0, -2.0],
            "def_final": [0.0, 0.0],
            "power_rating": [2.0, -2.0],
        })
        qbs = pd.DataFrame({
            "team": ["A", "B"],
            "quarterback": ["Elite", "Replacement"],
            "qb_value": [6.0, 0.0],
        })

        result = apply_qb_anchor(teams, qbs, anchor_weight=0.25)

        self.assertEqual(result.loc[result.team == "A", "off_final"].iloc[0], 2.25)
        self.assertEqual(result.loc[result.team == "B", "off_final"].iloc[0], -2.25)
        self.assertTrue(math.isclose(result["off_final"].mean(), 0.0))

    def test_starter_change_is_immediate_and_fades_with_observed_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ratings_dir = Path(temp_dir)
            pd.DataFrame({
                "team": ["A"],
                "power_rating": [0.0],
                "off_pts": [0.0],
                "def_pts": [0.0],
                "preseason_qb": ["Starter"],
                "preseason_qb_value": [5.0],
            }).to_csv(ratings_dir / "ratings_2026_preseason.csv", index=False)
            current_qb = pd.DataFrame({
                "team": ["A"],
                "current_qb": ["Backup"],
                "current_qb_value": [1.0],
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
                round(update_ratings.current_season_weight(2, 9.0), 3),
            )
            self.assertEqual(result.loc[0, "off_pts"],
                             round(4.0 * w + expected_adjustment, 3))
            self.assertEqual(result.loc[0, "power_rating"], result.loc[0, "off_pts"])


if __name__ == "__main__":
    unittest.main()
