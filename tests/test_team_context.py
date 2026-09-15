"""The home/away spread sign flip is easy to get backwards and silently wrong."""

import pandas as pd

from build_features import build_team_game_context


def _schedule():
    # KC hosts BUF, KC favoured by 3, total 50.
    # OAK hosts SD, away favoured by 2.5 (negative home spread), total 44.
    return pd.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "home_team": ["KC", "OAK"],
            "away_team": ["BUF", "SD"],
            "spread_line": [3.0, -2.5],
            "total_line": [50.0, 44.0],
        }
    )


def test_one_row_per_team_per_game():
    ctx = build_team_game_context(_schedule())
    assert len(ctx) == 4
    assert set(ctx["team"]) == {"KC", "BUF", "LV", "LAC"}


def test_home_flag():
    ctx = build_team_game_context(_schedule()).set_index("team")
    assert ctx.loc["KC", "is_home"] == 1
    assert ctx.loc["BUF", "is_home"] == 0


def test_favoured_team_gets_the_higher_implied_total():
    ctx = build_team_game_context(_schedule()).set_index("team")
    # KC favoured by 3 on a 50 total -> 26.5 / 23.5
    assert ctx.loc["KC", "implied_team_total"] == 26.5
    assert ctx.loc["BUF", "implied_team_total"] == 23.5
    assert ctx.loc["KC", "implied_team_total"] > ctx.loc["BUF", "implied_team_total"]


def test_away_favourite_gets_the_higher_implied_total():
    ctx = build_team_game_context(_schedule()).set_index("team")
    # Away side favoured by 2.5 on a 44 total -> LAC 23.25, LV 20.75
    assert ctx.loc["LAC", "implied_team_total"] == 23.25
    assert ctx.loc["LV", "implied_team_total"] == 20.75


def test_implied_totals_sum_to_the_game_total():
    ctx = build_team_game_context(_schedule())
    per_game = ctx.groupby(["season", "week", "total_line"])["implied_team_total"].sum()
    for total, summed in zip(per_game.index.get_level_values("total_line"), per_game):
        assert summed == total


def test_spread_is_flipped_for_the_away_side():
    ctx = build_team_game_context(_schedule()).set_index("team")
    assert ctx.loc["KC", "spread_line"] == 3.0
    assert ctx.loc["BUF", "spread_line"] == -3.0


def test_relocated_teams_are_normalized():
    ctx = build_team_game_context(_schedule())
    assert "OAK" not in set(ctx["team"])
    assert "SD" not in set(ctx["team"])
