"""CLAUDE.md flags FEATURE_COLS / ROLL_COLS drift as the standing footgun."""

import build_features as bf
import train_model as tm


def test_every_rolling_feature_the_model_wants_is_actually_built():
    built = {
        f"{short}_roll{window}"
        for short in bf.ROLL_COLS.values()
        for window in bf.ROLL_WINDOWS
    }
    wanted = {c for c in tm.FEATURE_COLS if "_roll" in c and c != "def_fp_allowed_roll"}
    missing = wanted - built
    assert not missing, (
        f"train_model.FEATURE_COLS wants {sorted(missing)}, which "
        "build_features.ROLL_COLS/ROLL_WINDOWS never produces."
    )


def test_no_rolling_feature_is_built_and_then_ignored():
    built = {
        f"{short}_roll{window}"
        for short in bf.ROLL_COLS.values()
        for window in bf.ROLL_WINDOWS
    }
    unused = built - set(tm.FEATURE_COLS)
    assert not unused, (
        f"build_features produces {sorted(unused)} but FEATURE_COLS ignores them — "
        "either add them to the model or drop them from ROLL_COLS."
    )


def test_baseline_column_is_produced_by_the_feature_builder():
    shorts = set(bf.ROLL_COLS.values())
    assert tm.BASELINE_COL.replace("_shifted", "") in shorts


def test_target_matches_across_stages():
    assert tm.TARGET_COL == bf.TARGET_COL


def test_non_rolling_features_are_named_columns_build_features_creates():
    # These are produced outside the ROLL_COLS loop; guard against typos.
    produced = {
        "implied_team_total", "spread_line", "total_line", "is_home",
        "def_fp_allowed_roll", "games_played", "is_questionable",
        "practice_status_code",
    }
    non_rolling = {
        c for c in tm.FEATURE_COLS
        if not c.endswith(tuple(f"_roll{w}" for w in bf.ROLL_WINDOWS))
    }
    assert non_rolling == produced
