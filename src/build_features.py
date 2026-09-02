"""Stage 3: join the raw nflverse tables into one modelling table.

Output is one row per (player, season, week) in ``data/processed/features.parquet``.

Leakage rules enforced here:
  * every rolling feature is ``shift(1)``ed before the window is applied, so a
    week's own production never appears in its own predictors;
  * the opponent adjustment is a *trailing* average, not a centered one.

``position_week_rank`` is computed as the ground-truth evaluation target. It is
NOT a model input.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

POSITIONS = ("QB", "RB", "WR", "TE")
TARGET_COL = "fantasy_points_ppr"

# Source column -> short name used in the rolling feature names.
# KEEP IN SYNC with FEATURE_COLS in train_model.py.
ROLL_COLS = {
    "fantasy_points_ppr": "fp_ppr",
    "targets": "tgt",
    "carries": "car",
    "receiving_yards": "rec_yds",
    "rushing_yards": "rush_yds",
    "passing_yards": "pass_yds",
    "offense_pct": "snap_pct",
    # Opportunity share. Snap share says a player was on the field; these say
    # the offense actually went to him, which is the stickier signal.
    "target_share": "tgt_share",
    "air_yards_share": "ay_share",
    "wopr": "wopr",
    "receiving_air_yards": "rec_ay",
}
ROLL_WINDOWS = (3, 5)
DEF_WINDOW = 4  # games of trailing defense-vs-position history

# Team abbreviations that changed hands; snap_counts (PFR) and player_stats
# (nflverse) don't always agree on the historical spelling.
TEAM_FIXES = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "LAR": "LA",
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "SL": "LA",
    "JAX": "JAX",
    "JAC": "JAX",
}

NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def resolve(df, table, *alternatives):
    """Return the first of ``alternatives`` present in ``df``, else raise."""
    for name in alternatives:
        if name in df.columns:
            return name
    raise KeyError(
        f"{table}: none of {alternatives} found. "
        f"Run src/inspect_schema.py — nflverse may have renamed this column. "
        f"Available: {sorted(df.columns)}"
    )


def normalize_name(series):
    """Lowercase, strip punctuation and generational suffixes: 'Odell Beckham Jr.' -> 'odell beckham'."""
    cleaned = (
        series.fillna("")
        .astype(str)
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.lower()
        .str.replace(r"[.'`]", "", regex=True)
        .str.replace(r"[^a-z ]", " ", regex=True)
        .str.split()
    )
    return cleaned.apply(
        lambda parts: " ".join(p for p in parts if p not in NAME_SUFFIXES)
    )


def practice_severity(series):
    """Map practice participation to an ordinal, increasing with severity.

    0 is reserved for "not on the injury report at all", which ``merge_injuries``
    fills in after the join — everything here is at least 1, because appearing
    on the report is itself mildly informative. Blank or malformed values (the
    source ships a few stray newlines) mean "listed, no limitation recorded",
    which is the same severity as full participation.
    """
    cleaned = series.fillna("").astype(str).str.strip().str.lower()
    code = pd.Series(1, index=series.index, dtype=int)
    code[cleaned.str.contains("limited", na=False)] = 2
    code[cleaned.str.contains("did not participate", na=False)] = 3
    return code


def normalize_team(series):
    return series.fillna("").astype(str).str.upper().replace(TEAM_FIXES)


def load_raw(raw_dir):
    tables = {}
    for name in ("player_stats", "schedules", "snap_counts", "injuries"):
        path = raw_dir / f"{name}.parquet"
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run src/pull_data.py first.")
        tables[name] = pd.read_parquet(path)
    return tables


def build_team_game_context(schedules):
    """Unpivot one row-per-game into two rows-per-team with Vegas context.

    ``spread_line`` is from the home team's perspective (positive = home
    favoured), so the implied total for each side is total/2 +/- spread/2.
    """
    season = resolve(schedules, "schedules", "season")
    week = resolve(schedules, "schedules", "week")
    home = resolve(schedules, "schedules", "home_team")
    away = resolve(schedules, "schedules", "away_team")
    spread = resolve(schedules, "schedules", "spread_line")
    total = resolve(schedules, "schedules", "total_line")

    base = schedules[[season, week, home, away, spread, total]].copy()
    base[spread] = pd.to_numeric(base[spread], errors="coerce")
    base[total] = pd.to_numeric(base[total], errors="coerce")

    home_rows = pd.DataFrame(
        {
            "season": base[season],
            "week": base[week],
            "team": base[home],
            "opponent": base[away],
            "is_home": 1,
            "spread_line": base[spread],
            "total_line": base[total],
            "implied_team_total": base[total] / 2 + base[spread] / 2,
        }
    )
    away_rows = pd.DataFrame(
        {
            "season": base[season],
            "week": base[week],
            "team": base[away],
            "opponent": base[home],
            "is_home": 0,
            "spread_line": -base[spread],
            "total_line": base[total],
            "implied_team_total": base[total] / 2 - base[spread] / 2,
        }
    )
    ctx = pd.concat([home_rows, away_rows], ignore_index=True)
    ctx["team"] = normalize_team(ctx["team"])
    ctx["opponent"] = normalize_team(ctx["opponent"])
    return ctx


def prepare_player_stats(player_stats):
    ps = player_stats.copy()
    team_col = resolve(ps, "player_stats", "team", "recent_team")
    name_col = resolve(ps, "player_stats", "player_display_name", "player_name")
    for required in ("player_id", "position", "season", "week", "opponent_team", TARGET_COL):
        resolve(ps, "player_stats", required)

    if "season_type" in ps.columns:
        ps = ps[ps["season_type"] == "REG"]

    ps = ps[ps["position"].isin(POSITIONS)].copy()
    # player_stats ships both player_name and player_display_name; renaming the
    # resolved one onto "player_name" would otherwise create a duplicate column.
    for canonical, source in (("team", team_col), ("player_name", name_col)):
        if source != canonical and canonical in ps.columns:
            ps = ps.drop(columns=[canonical])
    ps = ps.rename(columns={team_col: "team", name_col: "player_name"})

    # Columns that may be absent depending on the summary level nflverse served.
    for col in ROLL_COLS:
        if col not in ps.columns:
            ps[col] = np.nan
        ps[col] = pd.to_numeric(ps[col], errors="coerce")

    ps["team"] = normalize_team(ps["team"])
    ps["opponent_team"] = normalize_team(ps["opponent_team"])
    ps["name_key"] = normalize_name(ps["player_name"])
    ps["season"] = ps["season"].astype(int)
    ps["week"] = ps["week"].astype(int)
    return ps


def merge_snap_counts(df, snap_counts):
    """Join PFR snap counts on normalized name + team + season/week.

    snap_counts carries no gsis_id, so this join is inherently lossy — the
    unmatched % printed below is the diagnostic for how lossy. A crosswalk via
    ``nfl.load_ff_playerids()`` would be more robust.
    """
    sc = snap_counts.copy()
    player_col = resolve(sc, "snap_counts", "player", "player_name")
    team_col = resolve(sc, "snap_counts", "team")
    pct_col = resolve(sc, "snap_counts", "offense_pct")
    for required in ("season", "week"):
        resolve(sc, "snap_counts", required)

    if "game_type" in sc.columns:
        sc = sc[sc["game_type"] == "REG"]

    sc = sc.assign(
        name_key=normalize_name(sc[player_col]),
        team=normalize_team(sc[team_col]),
        season=sc["season"].astype(int),
        week=sc["week"].astype(int),
        offense_pct=pd.to_numeric(sc[pct_col], errors="coerce"),
    )
    # PFR reports percentages as fractions in some seasons, 0-100 in others.
    max_pct = sc["offense_pct"].max()
    if pd.notna(max_pct) and max_pct > 1.5:
        sc["offense_pct"] = sc["offense_pct"] / 100.0

    sc = sc[["season", "week", "team", "name_key", "offense_pct"]].drop_duplicates(
        subset=["season", "week", "team", "name_key"]
    )

    merged = df.drop(columns=["offense_pct"]).merge(
        sc, on=["season", "week", "team", "name_key"], how="left"
    )

    unmatched = merged["offense_pct"].isna().mean() * 100
    print(f"[build_features] snap_pct unmatched: {unmatched:.1f}% of player-weeks")
    if unmatched > 25:
        print(
            "[build_features] WARNING: high unmatched rate. Check name normalization "
            "(suffixes, accents) and team abbreviations, or switch to the "
            "nfl.load_ff_playerids() crosswalk."
        )
    return merged


def merge_injuries(df, injuries):
    """Attach the weekly injury report, joined on gsis_id where possible.

    Both signals here are published before kickoff — practice reports land
    Wednesday through Friday, the game designation Friday — so neither leaks.

    Note the row grain: a player ruled Out never plays, so he has no
    ``player_stats`` row and no feature row either. That makes an "is out" flag
    structurally dead (it fired 4 times in 40,330 rows), which is why the
    severity signal here comes from *practice participation* instead — that is
    recorded for players who go on to play, and reaches ~17% of rows against
    the game designation's ~4%.
    """
    inj = injuries.copy()
    status_col = resolve(inj, "injuries", "report_status")
    practice_col = resolve(inj, "injuries", "practice_status")
    id_col = "gsis_id" if "gsis_id" in inj.columns else None
    name_col = next(
        (c for c in ("full_name", "player_name", "player") if c in inj.columns), None
    )

    inj["season"] = inj["season"].astype(int)
    inj["week"] = inj["week"].astype(int)
    inj["report_status"] = inj[status_col].fillna("").astype(str).str.strip().str.title()
    inj["practice_code"] = practice_severity(inj[practice_col])
    inj["questionable"] = inj["report_status"].eq("Questionable").astype(int)

    keys = ["season", "week", "player_id"] if id_col else ["season", "week", "team", "name_key"]
    if id_col:
        inj = inj.rename(columns={id_col: "player_id"})
        inj = inj[inj["player_id"].notna()]
    elif name_col:
        inj["name_key"] = normalize_name(inj[name_col])
        inj["team"] = normalize_team(inj[resolve(inj, "injuries", "team")])
    else:
        raise KeyError("injuries: no gsis_id or name column to join on")

    # A handful of player-weeks carry two report rows. Take the worst of them
    # rather than whichever happens to sort first.
    keyed = inj.groupby(keys, as_index=False).agg(
        practice_code=("practice_code", "max"),
        is_questionable=("questionable", "max"),
        report_status=("report_status", "max"),
    )

    merged = df.merge(keyed, on=keys, how="left")
    merged["report_status"] = merged["report_status"].fillna("")
    merged["is_questionable"] = merged["is_questionable"].fillna(0).astype(int)
    # Not on the injury report at all is the healthiest state, hence 0.
    merged["practice_status_code"] = merged["practice_code"].fillna(0).astype(int)
    return merged.drop(columns=["practice_code"])


def build_opponent_adjustment(df):
    """Trailing average fantasy points a defense has allowed to a position.

    Trailing, never centered: the value attached to week W only uses weeks < W,
    so no future information reaches a given week's features.
    """
    allowed = (
        df.groupby(["season", "week", "opponent_team", "position"], as_index=False)[
            TARGET_COL
        ]
        .sum()
        .rename(columns={"opponent_team": "def_team", TARGET_COL: "fp_allowed"})
    )
    allowed = allowed.sort_values(["def_team", "position", "season", "week"])

    grouped = allowed.groupby(["def_team", "position"])["fp_allowed"]
    allowed["def_fp_allowed_roll"] = grouped.transform(
        lambda s: s.shift(1).rolling(DEF_WINDOW, min_periods=1).mean()
    )

    merged = df.merge(
        allowed[["season", "week", "def_team", "position", "def_fp_allowed_roll"]],
        left_on=["season", "week", "opponent_team", "position"],
        right_on=["season", "week", "def_team", "position"],
        how="left",
    ).drop(columns=["def_team"])
    return merged


def add_rolling_features(df):
    """Lag-then-roll: shift(1) first so the current week is never in its own window."""
    df = df.sort_values(["player_id", "season", "week"]).copy()
    grouped = df.groupby("player_id", sort=False)

    for source, short in ROLL_COLS.items():
        lagged = grouped[source].shift(1)
        df[f"{short}_shifted"] = lagged
        for window in ROLL_WINDOWS:
            df[f"{short}_roll{window}"] = lagged.groupby(df["player_id"]).transform(
                lambda s, w=window: s.rolling(w, min_periods=1).mean()
            )

    df["games_played"] = grouped.cumcount()
    return df


def add_position_week_rank(df):
    """Ground-truth rank within (season, week, position). Evaluation target only."""
    df["position_week_rank"] = (
        df.groupby(["season", "week", "position"])[TARGET_COL]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    tables = load_raw(args.raw_dir)

    df = prepare_player_stats(tables["player_stats"])
    print(f"[build_features] player-weeks after position/REG filter: {len(df):,}")

    ctx = build_team_game_context(tables["schedules"])
    df = df.merge(
        ctx[["season", "week", "team", "is_home", "spread_line", "total_line", "implied_team_total"]],
        on=["season", "week", "team"],
        how="left",
    )
    missing_vegas = df["implied_team_total"].isna().mean() * 100
    print(f"[build_features] implied_team_total missing: {missing_vegas:.1f}%")

    df = merge_snap_counts(df, tables["snap_counts"])
    df = merge_injuries(df, tables["injuries"])
    df = build_opponent_adjustment(df)
    df = add_rolling_features(df)
    df = add_position_week_rank(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[build_features] wrote {len(df):,} rows x {df.shape[1]} cols -> {out_path}")
    print("[build_features] next: python src/train_model.py --test-season 2024")


if __name__ == "__main__":
    main()
