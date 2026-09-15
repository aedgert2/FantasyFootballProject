"""Archive FantasyPros expert consensus rankings, weekly.

``nfl.load_ff_rankings()`` returns only a **current snapshot** — there is no
history in nflverse — so any week not captured is lost permanently. That makes
this the one piece of data in the project that cannot be regenerated, which is
why ``data/rankings/`` is committed while the rest of ``data/`` is gitignored.

The point is the benchmark. The model is currently compared against a 5-game
rolling average, which it does not beat. The question that actually decides
whether this project is worth using is whether it beats free public rankings,
and answering it requires a season of snapshots taken before kickoff.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RANKINGS_DIR = ROOT / "data" / "rankings"


def current_nfl_week(today):
    """Best-effort (season, week) for the slate this snapshot precedes.

    Derived from the schedule rather than the calendar, because week boundaries
    move. Returns (None, None) rather than raising — a snapshot with an unknown
    week is still worth keeping, since the week can be recovered later from the
    schedule, but a missed snapshot cannot.
    """
    try:
        season = today.year if today.month >= 3 else today.year - 1
        sched = nfl.load_schedules(seasons=[season]).to_pandas()
        sched = sched[sched["gameday"].notna()]
        upcoming = sched[sched["gameday"] >= today.strftime("%Y-%m-%d")]
        if upcoming.empty:
            return season, None
        return season, int(upcoming.sort_values("gameday").iloc[0]["week"])
    except Exception as exc:  # noqa: BLE001 - never let this lose a snapshot
        print(f"[archive_rankings] could not derive week ({exc}); storing anyway")
        return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RANKINGS_DIR)
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    season, week = current_nfl_week(now)

    df = nfl.load_ff_rankings()
    df = df.with_columns(
        pl.lit(now.isoformat()).alias("captured_at"),
        pl.lit(season, dtype=pl.Int64).alias("nfl_season"),
        pl.lit(week, dtype=pl.Int64).alias("nfl_week"),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # One file per capture date. Re-running the same day overwrites rather than
    # duplicating, so a retry after a failure is safe.
    path = args.out_dir / f"ecr_{now.strftime('%Y-%m-%d')}.parquet"
    df.write_parquet(path)

    label = f"season {season} week {week}" if week else "week unknown"
    print(f"[archive_rankings] {df.height:,} rows x {df.width} cols ({label}) -> {path}")
    print(f"[archive_rankings] archive now holds {len(list(args.out_dir.glob('ecr_*.parquet')))} snapshots")


if __name__ == "__main__":
    main()
