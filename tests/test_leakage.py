"""The whole model is worthless if a week's own production reaches its features."""

import numpy as np
import pandas as pd

from build_features import ROLL_COLS, add_rolling_features, build_opponent_adjustment


def _player_weeks(player_id, points):
    n = len(points)
    df = pd.DataFrame(
        {
            "player_id": [player_id] * n,
            "season": [2023] * n,
            "week": list(range(1, n + 1)),
            "fantasy_points_ppr": points,
        }
    )
    for col in ROLL_COLS:
        if col not in df.columns:
            df[col] = np.arange(n, dtype=float)
    return df


def test_shifted_column_is_the_previous_week():
    out = add_rolling_features(_player_weeks("A", [10.0, 20.0, 30.0, 40.0]))
    assert np.isnan(out["fp_ppr_shifted"].iloc[0])
    assert out["fp_ppr_shifted"].tolist()[1:] == [10.0, 20.0, 30.0]


def test_rolling_window_excludes_the_current_week():
    out = add_rolling_features(_player_weeks("A", [10.0, 20.0, 30.0, 40.0]))
    roll3 = out["fp_ppr_roll3"].tolist()
    assert np.isnan(roll3[0])          # nothing prior
    assert roll3[1] == 10.0            # only week 1
    assert roll3[2] == 15.0            # mean(10, 20) — not mean(10, 20, 30)
    assert roll3[3] == 20.0            # mean(10, 20, 30) — week 4's own 40 absent


def test_rolling_never_equals_a_window_containing_the_current_week():
    points = [10.0, 20.0, 30.0, 40.0]
    out = add_rolling_features(_player_weeks("A", points))
    leaky = pd.Series(points).rolling(3, min_periods=1).mean()
    assert not np.allclose(
        out["fp_ppr_roll3"].fillna(-1), leaky.fillna(-1)
    ), "rolling window appears to include the current week"


def test_players_do_not_bleed_into_each_other():
    df = pd.concat(
        [_player_weeks("A", [10.0, 20.0]), _player_weeks("B", [100.0, 200.0])],
        ignore_index=True,
    )
    out = add_rolling_features(df).set_index(["player_id", "week"])
    assert np.isnan(out.loc[("B", 1), "fp_ppr_shifted"])
    assert out.loc[("B", 2), "fp_ppr_shifted"] == 100.0
    assert out.loc[("A", 2), "fp_ppr_shifted"] == 10.0


def test_games_played_counts_prior_games_only():
    out = add_rolling_features(_player_weeks("A", [10.0, 20.0, 30.0]))
    assert out["games_played"].tolist() == [0, 1, 2]


def test_opponent_adjustment_is_trailing_not_centered():
    # One WR per week against SF, scoring 10 / 20 / 30.
    df = pd.DataFrame(
        {
            "season": [2023] * 3,
            "week": [1, 2, 3],
            "opponent_team": ["SF"] * 3,
            "position": ["WR"] * 3,
            "fantasy_points_ppr": [10.0, 20.0, 30.0],
        }
    )
    out = build_opponent_adjustment(df).sort_values("week")
    allowed = out["def_fp_allowed_roll"].tolist()
    assert np.isnan(allowed[0])   # no prior history
    assert allowed[1] == 10.0     # week 1 only
    assert allowed[2] == 15.0     # mean(10, 20) — week 3's own 30 excluded


def test_opponent_adjustment_is_per_position():
    df = pd.DataFrame(
        {
            "season": [2023] * 4,
            "week": [1, 1, 2, 2],
            "opponent_team": ["SF"] * 4,
            "position": ["WR", "TE", "WR", "TE"],
            "fantasy_points_ppr": [30.0, 5.0, 30.0, 5.0],
        }
    )
    out = build_opponent_adjustment(df)
    week2 = out[out["week"] == 2].set_index("position")["def_fp_allowed_roll"]
    assert week2["WR"] == 30.0
    assert week2["TE"] == 5.0
