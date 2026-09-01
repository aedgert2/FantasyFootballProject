# Fantasy Start/Sit Predictor

Ranks fantasy players within a position for a given week, per the project
notes: rolling-average form, opponent-adjusted defense strength, Vegas
context, and a dumb baseline to confirm the model actually adds value.

## Setup

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Two things that will bite you otherwise:

- On macOS, LightGBM needs OpenMP at runtime — `brew install libomp`, or
  `import lightgbm` fails with a `libomp.dylib` load error.
- `scikit-learn` is in `requirements.txt` even though nothing here imports it
  directly; `lightgbm.sklearn` refuses to load without it.

Dependencies are pinned to the versions the results below were produced with.
Python version is recorded in `.python-version` (3.13).

## Run order

```bash
python src/pull_data.py --seasons 2021 2022 2023 2024   # downloads from nflverse, caches to data/raw/
python src/inspect_schema.py                              # sanity-check column names before trusting the join
python src/build_features.py                               # joins tables, builds rolling/opponent/Vegas features
python src/train_model.py --test-season 2024                # trains LightGBM per position, compares to baseline
```

## Tests

```bash
pytest
```

37 tests, no network and no pulled data required. They cover the parts most
likely to break silently rather than loudly:

- **Name/team normalization** — suffixes (`Jr.`, `III`), punctuation, accents,
  and relocated-franchise abbreviations. This is what the lossy snap-count join
  depends on.
- **Vegas context** — that the favoured side gets the higher implied total in
  both the home-favourite and away-favourite cases, that the two sides sum back
  to `total_line`, and that the spread sign is flipped for the away team.
- **Leakage** — that every rolling feature excludes its own week, that players
  do not bleed into each other, and that the opponent adjustment is trailing
  rather than centered. One test asserts the rolling output is *not* equal to
  the naive (leaky) window, so a regression here fails loudly.
- **Feature sync** — that `FEATURE_COLS` in `train_model.py` and
  `ROLL_COLS`/`ROLL_WINDOWS` in `build_features.py` agree in both directions.
  This is the drift the project notes call out as the standing footgun.

## Files

- `src/pull_data.py` — pulls `player_stats`, `schedules`, `snap_counts`, `injuries` via `nflreadpy` and caches as parquet in `data/raw/`.
- `src/inspect_schema.py` — prints columns/sample rows of each raw table. Run this first — nflverse schemas shift occasionally, and this catches a mismatch before it silently breaks a join.
- `src/build_features.py` — joins the raw tables on (player/team, season, week) and builds:
  - rolling averages (last 3 / 5 games) of points, targets, carries, yards, snap %, all lagged so the current week is never in its own average (no leakage)
  - opponent-adjusted defense strength (trailing average fantasy points allowed to that position)
  - Vegas context (implied team total, spread, home/away)
  - injury report status
  - Writes `data/processed/features.parquet`.
- `src/train_model.py` — dumb baseline (rank by last week's points) vs. a LightGBM regressor trained per position, evaluated by Spearman rank correlation within each (season, week, position) group — the metric that matters for start/sit, since getting the order right matters more than nailing exact point totals. Time-based split: trains on seasons before `--test-season`, tests on that season.

## Known gotchas (flagged in the original notes, confirmed while building this)

- **Player ID mismatch**: `player_stats` keys off `player_id` (gsis ID), but `snap_counts` comes from Pro Football Reference and only carries a player name + team. `build_features.py` joins snap counts on normalized name + team + season + week as a best-effort match — check the "snap_pct unmatched" print statement after running; if it's high, the name normalization needs work (suffixes like Jr./II, defunct team abbreviations after a trade, etc.). A more robust fix is to pull `nfl.load_ff_playerids()` and join through its ID crosswalk instead of matching on name. On 2021–2024 the name-based join lands at **1.1% unmatched**, which is low enough that the crosswalk isn't urgent.
- **Verified against live nflverse data** (2021–2024): `inspect_schema.py` reports all expected columns present, and the full pipeline runs end to end. Still run `inspect_schema.py` after any fresh pull — nflverse renames columns occasionally, and `build_features.py` raises a `KeyError` naming the exact table/column when one goes missing.
- Vegas lines (`total_line`, `spread_line`) aren't always populated for older seasons — on 2021–2024 `implied_team_total` came back 0% missing, but expect NaNs if you pull further back.

## Results (test season 2024, trained on 2021–2023)

Mean Spearman rank correlation within each (week, position) group — the metric
that matters for start/sit. The model beats the "start last week's higher
scorer" baseline at every position.

| Position | n    | MAE  | Model rho | Baseline rho | Lift   |
|----------|------|------|-----------|--------------|--------|
| QB       | 653  | 6.58 | 0.467     | 0.424        | +0.043 |
| RB       | 1504 | 4.58 | 0.668     | 0.628        | +0.041 |
| WR       | 2391 | 4.87 | 0.615     | 0.537        | +0.079 |
| TE       | 1198 | 3.89 | 0.545     | 0.439        | +0.106 |
| **Mean** |      |      | **0.574** | **0.507**    | **+0.067** |

## Next steps (from the original notes)

- Decide final target: currently trains on `fantasy_points_ppr`; swap for standard/half-PPR by changing `TARGET_COL` in `train_model.py`.
- Once real data confirms the join rate, consider replacing the name-based snap-count join with the `load_ff_playerids()` crosswalk for reliability.
- Try LightGBM's `LambdaRank` objective instead of plain regression + post-hoc ranking, now that the pipeline runs end-to-end.