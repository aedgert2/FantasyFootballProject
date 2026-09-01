"""Stage 2 (diagnostic): print the schema of each cached raw table.

``build_features.py`` hard-codes nflverse column names, and nflverse renames
columns from time to time. Run this after a fresh pull to confirm the columns
the join expects are actually present before trusting the features.
"""

import argparse
from pathlib import Path

import polars as pl

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Columns build_features.py depends on. Alternatives are listed together because
# nflverse has renamed a few of these over the years (recent_team -> team, etc.).
EXPECTED = {
    "player_stats": [
        ("player_id",),
        ("player_display_name", "player_name"),
        ("position",),
        ("team", "recent_team"),
        ("opponent_team",),
        ("season",),
        ("week",),
        ("season_type",),
        ("fantasy_points_ppr",),
        ("targets",),
        ("carries",),
        ("receiving_yards",),
        ("rushing_yards",),
        ("passing_yards",),
    ],
    "schedules": [
        ("season",),
        ("week",),
        ("game_type",),
        ("home_team",),
        ("away_team",),
        ("spread_line",),
        ("total_line",),
    ],
    "snap_counts": [
        ("season",),
        ("week",),
        ("player",),
        ("team",),
        ("position",),
        ("offense_pct",),
    ],
    "injuries": [
        ("season",),
        ("week",),
        ("team",),
        ("gsis_id",),
        ("full_name",),
        ("report_status",),
    ],
}


def check_expected(name, columns):
    missing = []
    for alternatives in EXPECTED.get(name, []):
        if not any(alt in columns for alt in alternatives):
            missing.append(" | ".join(alternatives))
    if missing:
        print(f"  !! MISSING expected columns: {', '.join(missing)}")
        print("     build_features.py will raise a KeyError on these.")
    else:
        print("  all expected columns present")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--rows", type=int, default=3, help="Sample rows to print")
    args = parser.parse_args()

    paths = sorted(args.raw_dir.glob("*.parquet"))
    if not paths:
        raise SystemExit(
            f"No parquet files in {args.raw_dir}. Run src/pull_data.py first."
        )

    for path in paths:
        name = path.stem
        df = pl.read_parquet(path)
        print("=" * 78)
        print(f"{name}: {df.height:,} rows x {df.width} cols  ({path})")
        print("-" * 78)
        for col, dtype in zip(df.columns, df.dtypes):
            print(f"  {col:<32} {dtype}")
        check_expected(name, set(df.columns))
        if "season" in df.columns:
            seasons = sorted(df["season"].unique().to_list())
            print(f"  seasons: {seasons}")
        print("-" * 78)
        with pl.Config(tbl_cols=12, tbl_width_chars=160):
            print(df.head(args.rows))
        print()


if __name__ == "__main__":
    main()
