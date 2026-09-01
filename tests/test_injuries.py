"""The injury signal has to come from something the row grain can carry.

A player ruled Out has no player_stats row and therefore no feature row, so an
"is out" flag is structurally dead. Practice participation is recorded for
players who go on to play, which is the population the model actually scores.
"""

import pandas as pd
import pytest

from build_features import merge_injuries, practice_severity


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Full Participation in Practice", 1),
        ("Limited Participation in Practice", 2),
        ("Did Not Participate In Practice", 3),
        ("did not participate in practice", 3),  # casing varies in the source
        ("  Limited Participation in Practice  ", 2),
    ],
)
def test_practice_severity_is_ordinal(raw, expected):
    assert practice_severity(pd.Series([raw])).iloc[0] == expected


@pytest.mark.parametrize("junk", ["", "\\n", "   ", None])
def test_malformed_participation_is_listed_but_unlimited(junk):
    # On the report, no limitation recorded — same severity as full practice,
    # and never confused with "not on the report at all" (which is 0).
    assert practice_severity(pd.Series([junk])).iloc[0] == 1


def _features(player_ids):
    return pd.DataFrame(
        {"season": 2024, "week": 3, "player_id": player_ids, "team": "KC"}
    )


def _injuries(rows):
    return pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 3,
                "gsis_id": pid,
                "team": "KC",
                "full_name": "X",
                "report_status": rep,
                "practice_status": prac,
            }
            for pid, rep, prac in rows
        ]
    )


def test_player_absent_from_the_report_scores_zero():
    out = merge_injuries(_features(["p1"]), _injuries([("p2", "Questionable", "Limited Participation in Practice")]))
    assert out["practice_status_code"].iloc[0] == 0
    assert out["is_questionable"].iloc[0] == 0


def test_practice_status_reaches_players_who_played():
    out = merge_injuries(
        _features(["p1"]),
        _injuries([("p1", None, "Did Not Participate In Practice")]),
    )
    assert out["practice_status_code"].iloc[0] == 3
    # No game designation, but the practice signal still lands — this is the
    # coverage the old is_doubtful_or_out flag could never reach.
    assert out["is_questionable"].iloc[0] == 0


def test_questionable_still_flagged():
    out = merge_injuries(
        _features(["p1"]),
        _injuries([("p1", "Questionable", "Limited Participation in Practice")]),
    )
    assert out["is_questionable"].iloc[0] == 1
    assert out["practice_status_code"].iloc[0] == 2


def test_duplicate_report_rows_take_the_worst():
    out = merge_injuries(
        _features(["p1"]),
        _injuries(
            [
                ("p1", None, "Full Participation in Practice"),
                ("p1", "Questionable", "Did Not Participate In Practice"),
            ]
        ),
    )
    assert len(out) == 1
    assert out["practice_status_code"].iloc[0] == 3
    assert out["is_questionable"].iloc[0] == 1


def test_severity_is_never_null_after_merge():
    out = merge_injuries(_features(["p1", "p2"]), _injuries([("p1", None, "Limited Participation in Practice")]))
    assert out["practice_status_code"].notna().all()
    assert out["practice_status_code"].dtype.kind == "i"
