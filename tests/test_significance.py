"""The lift is only testable if the per-group rhos are genuinely paired."""

import numpy as np
import pandas as pd

from train_model import group_rho_table


def _group(week, preds, base, actual, position="WR"):
    return pd.DataFrame(
        {
            "season": 2024,
            "week": week,
            "position": position,
            "predicted_points": preds,
            "fp_ppr_shifted": base,
            "fantasy_points_ppr": actual,
        }
    )


COLS = ["predicted_points", "fp_ppr_shifted"]


def test_perfect_ordering_scores_one():
    df = _group(1, [30.0, 20.0, 10.0], [30.0, 20.0, 10.0], [28.0, 19.0, 9.0])
    table = group_rho_table(df, COLS)
    assert table["predicted_points"].iloc[0] == 1.0


def test_inverted_ordering_scores_minus_one():
    df = _group(1, [30.0, 20.0, 10.0], [10.0, 20.0, 30.0], [9.0, 19.0, 28.0])
    table = group_rho_table(df, COLS)
    assert table["predicted_points"].iloc[0] == -1.0
    assert table["fp_ppr_shifted"].iloc[0] == 1.0


def test_one_row_per_group():
    df = pd.concat(
        [
            _group(1, [30.0, 20.0, 10.0], [30.0, 20.0, 10.0], [28.0, 19.0, 9.0]),
            _group(2, [30.0, 20.0, 10.0], [30.0, 20.0, 10.0], [28.0, 19.0, 9.0]),
            _group(1, [30.0, 20.0, 10.0], [30.0, 20.0, 10.0], [28.0, 19.0, 9.0], "TE"),
        ]
    )
    table = group_rho_table(df, COLS)
    assert len(table) == 3
    assert set(zip(table.week, table.position)) == {(1, "WR"), (2, "WR"), (1, "TE")}


def test_groups_smaller_than_three_are_dropped():
    df = _group(1, [30.0, 20.0], [30.0, 20.0], [28.0, 19.0])
    assert group_rho_table(df, COLS).empty


def test_a_group_unusable_for_one_score_is_dropped_for_both():
    # Baseline is flat, so it has no rank order — the pair is meaningless and the
    # whole group must go, or the two means would be averaged over different
    # groups and the lift would compare unlike with unlike.
    df = _group(1, [30.0, 20.0, 10.0], [5.0, 5.0, 5.0], [28.0, 19.0, 9.0])
    assert group_rho_table(df, COLS).empty


def test_rows_missing_either_score_are_excluded():
    df = _group(
        1,
        [30.0, 20.0, 10.0, 40.0],
        [30.0, 20.0, 10.0, np.nan],
        [28.0, 19.0, 9.0, 35.0],
    )
    table = group_rho_table(df, COLS)
    assert table["n"].iloc[0] == 3


def test_group_size_is_reported():
    df = _group(1, [30.0, 20.0, 10.0], [30.0, 20.0, 10.0], [28.0, 19.0, 9.0])
    assert group_rho_table(df, COLS)["n"].iloc[0] == 3
