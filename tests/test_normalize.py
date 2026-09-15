"""Name/team normalization is what the lossy snap-count join hangs on."""

import pandas as pd
import pytest

from build_features import normalize_name, normalize_team


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Odell Beckham Jr.", "odell beckham"),
        ("Robert Griffin III", "robert griffin"),
        ("Ken Walker III", "ken walker"),
        ("A.J. Brown", "aj brown"),
        ("D'Andre Swift", "dandre swift"),
        ("Amon-Ra St. Brown", "amon ra st brown"),
        ("MARVIN HARRISON", "marvin harrison"),
        ("  Extra   Spaces  ", "extra spaces"),
        ("", ""),
    ],
)
def test_normalize_name(raw, expected):
    assert normalize_name(pd.Series([raw])).iloc[0] == expected


def test_normalize_name_strips_accents():
    # PFR and nflverse don't always agree on diacritics.
    assert normalize_name(pd.Series(["Chigoziem Okonkwo"])).iloc[0] == "chigoziem okonkwo"
    assert normalize_name(pd.Series(["José Borregales"])).iloc[0] == "jose borregales"


def test_normalize_name_handles_nulls():
    assert normalize_name(pd.Series([None])).iloc[0] == ""


def test_suffix_only_inside_word_is_kept():
    # "Ivy" must not lose a syllable to the roman-numeral "iv" rule.
    assert normalize_name(pd.Series(["Calvin Ivy"])).iloc[0] == "calvin ivy"


@pytest.mark.parametrize(
    "raw, expected",
    [("OAK", "LV"), ("SD", "LAC"), ("STL", "LA"), ("LAR", "LA"), ("JAC", "JAX")],
)
def test_normalize_team_maps_relocations(raw, expected):
    assert normalize_team(pd.Series([raw])).iloc[0] == expected


def test_normalize_team_passes_through_current_abbreviations():
    assert normalize_team(pd.Series(["kc"])).iloc[0] == "KC"
