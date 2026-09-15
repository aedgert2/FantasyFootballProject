"""Routes are derived by joining two tables on gsis id, with two real traps.

1. participation's ``offense_positions`` is 100% blank before 2023, so filtering
   on it silently yields zero routes for 2019-2022.
2. Only pass plays count, and participation's own ``time_to_throw`` misses 12%
   of them (sacks and scrambles), so the pass flag has to come from pbp.
"""

import pandas as pd

from build_features import build_routes


def _participation(rows):
    return pd.DataFrame(
        [{"nflverse_game_id": g, "play_id": p, "offense_players": players,
          "offense_positions": positions}
         for g, p, players, positions in rows]
    )


def _pbp(rows):
    return pd.DataFrame(
        [{"game_id": g, "play_id": p, "season": s, "week": w, "pass": ps}
         for g, p, s, w, ps in rows]
    )


def test_counts_pass_plays_a_player_was_on_the_field_for():
    part = _participation([
        ("2024_01_A_B", 1, "p1;p2", "WR;TE"),
        ("2024_01_A_B", 2, "p1", "WR"),
    ])
    pbp = _pbp([("2024_01_A_B", 1, 2024, 1, 1), ("2024_01_A_B", 2, 2024, 1, 1)])
    r = build_routes(part, pbp).set_index("player_id")["routes"]
    assert r["p1"] == 2
    assert r["p2"] == 1


def test_run_plays_are_not_routes():
    part = _participation([
        ("2024_01_A_B", 1, "p1", "WR"),
        ("2024_01_A_B", 2, "p1", "WR"),
    ])
    pbp = _pbp([("2024_01_A_B", 1, 2024, 1, 1), ("2024_01_A_B", 2, 2024, 1, 0)])
    assert build_routes(part, pbp).set_index("player_id")["routes"]["p1"] == 1


def test_blank_offense_positions_still_yields_routes():
    # The 2019-2022 shape: players present, positions column empty. Filtering on
    # positions here produced zero routes for four full seasons.
    part = _participation([("2019_01_A_B", 1, "p1;p2", "")])
    pbp = _pbp([("2019_01_A_B", 1, 2019, 1, 1)])
    r = build_routes(part, pbp)
    assert len(r) == 2
    assert set(r.player_id) == {"p1", "p2"}


def test_missing_offense_players_is_skipped_not_fatal():
    part = _participation([
        ("2024_01_A_B", 1, None, "WR"),
        ("2024_01_A_B", 2, "p1", "WR"),
    ])
    pbp = _pbp([("2024_01_A_B", 1, 2024, 1, 1), ("2024_01_A_B", 2, 2024, 1, 1)])
    r = build_routes(part, pbp)
    assert r.set_index("player_id")["routes"]["p1"] == 1


def test_routes_are_split_by_week():
    part = _participation([
        ("2024_01_A_B", 1, "p1", "WR"),
        ("2024_02_A_B", 1, "p1", "WR"),
    ])
    pbp = _pbp([("2024_01_A_B", 1, 2024, 1, 1), ("2024_02_A_B", 1, 2024, 2, 1)])
    r = build_routes(part, pbp)
    assert len(r) == 2
    assert sorted(r.week) == [1, 2]
    assert set(r.routes) == {1}


def test_plays_absent_from_pbp_are_dropped():
    part = _participation([("2024_01_A_B", 99, "p1", "WR")])
    pbp = _pbp([("2024_01_A_B", 1, 2024, 1, 1)])
    assert build_routes(part, pbp).empty
