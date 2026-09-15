"""Stage 1: pull raw nflverse tables and cache them as parquet.

Everything downstream reads from ``data/raw/*.parquet`` rather than hitting the
network again, so this is the only script that needs connectivity.
"""

import argparse
from pathlib import Path

import nflreadpy as nfl

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# name -> nflreadpy loader. Each loader takes a ``seasons`` argument and returns
# a polars DataFrame.
TABLES = {
    "player_stats": nfl.load_player_stats,
    "schedules": nfl.load_schedules,
    "snap_counts": nfl.load_snap_counts,
    "injuries": nfl.load_injuries,
    "participation": nfl.load_participation,
}

# Play-by-play is only needed to tell pass plays from run plays when deriving
# routes run, and the full table is enormous, so only these columns are cached.
# There is no usable shortcut: participation's own time_to_throw has 99.9%
# precision but misses 12% of pass plays (sacks and scrambles), and that gap is
# biased toward teams with mobile quarterbacks.
PBP_COLS = ["game_id", "play_id", "season", "week", "pass"]


def pull_table(name, loader, seasons):
    print(f"[pull_data] {name}: downloading seasons {seasons} ...")
    df = loader(seasons=seasons)
    # Some loaders (schedules) ignore the season filter and return everything.
    if "season" in df.columns:
        df = df.filter(df["season"].is_in(seasons))
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        required=True,
        help="Seasons to download, e.g. --seasons 2021 2022 2023 2024",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Where to cache the parquet files (default: {RAW_DIR})",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, loader in TABLES.items():
        df = pull_table(name, loader, args.seasons)
        path = args.out_dir / f"{name}.parquet"
        df.write_parquet(path)
        print(f"[pull_data] {name}: {df.height:,} rows x {df.width} cols -> {path}")

    print(f"[pull_data] pbp_pass: downloading seasons {args.seasons} (large) ...")
    pbp = nfl.load_pbp(seasons=args.seasons).select(PBP_COLS)
    path = args.out_dir / "pbp_pass.parquet"
    pbp.write_parquet(path)
    print(f"[pull_data] pbp_pass: {pbp.height:,} rows x {pbp.width} cols -> {path}")

    print("[pull_data] done. Next: python src/inspect_schema.py")


if __name__ == "__main__":
    main()
