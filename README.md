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
python src/pull_data.py --seasons 2019 2020 2021 2022 2023 2024 2025
python src/inspect_schema.py                    # sanity-check column names before trusting the join
python src/build_features.py                    # joins tables, builds rolling/opponent/Vegas features
python src/train_model.py --test-season 2025    # trains LightGBM per position, compares to baseline
python src/train_model.py --test-season 2025 --significance   # ...and tests whether the lift beats chance
```

**2019-2025 is the standard pull.** Training window was chosen by measurement,
not habit: holding 2024 and 2025 out and varying the training start year, the
lift rises monotonically with more history and never degrades.

| Train from | Lift on 2024 | Lift on 2025 |
|-----------:|-------------:|-------------:|
| 2022       | +0.060       | +0.064       |
| 2021       | +0.069       | +0.072       |
| 2020       | +0.068       | +0.083       |
| **2019**   | **+0.081**   | **+0.084**   |

The COVID-shortened 2020 season was the obvious thing to worry about — no
preseason, empty stadiums, home-field advantage worth roughly nothing — but
including it costs nothing measurable. Don't extend before 2019 without
re-checking: Vegas lines get sparse in older seasons.

## Tests

```bash
pytest
```

44 tests, no network and no pulled data required. They cover the parts most
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
- **Paired scoring** — that `group_rho_table()` drops a group from *both*
  score columns when it is unusable for either, so model and baseline rho are
  always averaged over identical groups and the lift compares like with like.

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

- **Player ID mismatch**: `player_stats` keys off `player_id` (gsis ID), but `snap_counts` comes from Pro Football Reference and only carries a player name + team. `build_features.py` joins snap counts on normalized name + team + season + week as a best-effort match — check the "snap_pct unmatched" print statement after running; if it's high, the name normalization needs work (suffixes like Jr./II, defunct team abbreviations after a trade, etc.). A more robust fix is to pull `nfl.load_ff_playerids()` and join through its ID crosswalk instead of matching on name. Across 2019-2025 the name-based join lands at **0.9% unmatched**, low enough that the crosswalk isn't urgent.
- **Verified against live nflverse data** (2019-2025): `inspect_schema.py` reports all expected columns present, and the full pipeline runs end to end. Still run `inspect_schema.py` after any fresh pull — nflverse renames columns occasionally, and `build_features.py` raises a `KeyError` naming the exact table/column when one goes missing.
- Vegas lines (`total_line`, `spread_line`) aren't always populated for older seasons — across 2019-2025 `implied_team_total` comes back 0% missing, but expect NaNs if you pull further back. That is the reason 2019 is the floor.
- **`is_doubtful_or_out` is dead by construction.** It fires 4 times in 40,330 rows. A player ruled Out doesn't play, so he has no `player_stats` row and therefore no feature row at all — the flag can only fire for someone listed Out/Doubtful who suited up anyway. `is_questionable` is alive but sparse (4.3%). Neither is a join bug; the feature is simply asking a question the row grain can't answer.
- **Rolling windows cross the season boundary.** Features group by `player_id` alone, so week 1 of a season carries form from the end of the previous one. Defensible — better than starting every season blind — but it means an offseason team change is invisible to the feature. `games_played` counts across seasons for the same reason.

## Results

Two held-out seasons, each trained on every prior season. The metric is mean
Spearman rho within each (week, position) group — the ordering is what you act
on, so the absolute point totals are deliberately discarded.

### 2025 — the primary result

Trained on 2019-2024 (33,067 rows), tested on 2025 (5,914 rows).

| Position | n    | MAE  | Model rho | Baseline rho | Lift   |
|----------|------|------|-----------|--------------|--------|
| QB       | 650  | 7.09 | 0.371     | 0.343        | +0.027 |
| RB       | 1539 | 4.64 | 0.704     | 0.626        | +0.078 |
| WR       | 2461 | 4.38 | 0.641     | 0.502        | +0.139 |
| TE       | 1264 | 3.86 | 0.564     | 0.473        | +0.092 |
| **Mean** |      |      | **0.570** | **0.486**    | **+0.084** |

In decisions rather than correlations: over all 275,564 same-week same-position
head-to-heads, the model picks the higher scorer **73.5%** of the time against
the baseline's **67.2%**.

### 2024 — second held-out season

Trained on 2019-2023. Mean model rho **0.587** vs. baseline **0.507**, lift
**+0.081**, ahead in 60 of 72 groups. Two independent seasons landing within
0.003 of each other is the reason to believe the edge is real.

### Is the lift significant?

`--significance` runs paired tests on the 72 per-group differences (each group
yields a model rho and a baseline rho over identical players, so they pair).

| | 2025 | 2024 |
|---|---|---|
| Mean lift | +0.084 | +0.081 |
| Groups won | 59/72 | 60/72 |
| Paired *t* | 8.04, p < 0.0001 | 6.36, p < 0.0001 |
| Bootstrap 95% CI | [+0.068, +0.103] | [+0.055, +0.109] |

The CI comes from resampling whole **weeks**, not groups, because positions
within a week share the same games and aren't independent.

**Quarterback is the exception.** The lift is +0.027 in 2025 (p = 0.30) and
+0.046 in 2024 (p = 0.25) — not significant in either. The model does not beat
"start last week's higher scorer" at QB, and that replicates across two
independent seasons. It fits the feature story: snap share is the model's
strongest signal for skill players and carries no information for quarterbacks,
who never leave the field.

One caveat on reading these numbers: rho is sensitive to how wide a group it
spans. WR week 5 of 2025 scores **+0.684** across all 126 receivers but
**-0.524** over the model's own top 8, where the spread is mostly noise. The
headline describes ordering the full position pool, not the close call you
actually agonize over on Sunday.

## Next steps (from the original notes)

- **Try `LambdaRank`.** The model optimizes squared error on points and the ordering is imposed afterwards, but only the ordering is graded. LightGBM's ranking objective with each (week, position) as a query group optimizes the actual target directly. This is the highest-value change available.
- **Fix or drop `is_doubtful_or_out`.** As above, it is a constant. Either drop it, or change what it asks — e.g. carry the *previous* week's status forward, which is knowable at lineup time and does apply to players who then play.
- **Work out why QB is flat.** No significant lift in either held-out season. QBs have no snap-share signal; the model may need quarterback-specific features (pressure rate, opponent pass defense, designed-run share) rather than the shared feature set.
- Decide final target: currently trains on `fantasy_points_ppr`; swap for standard/half-PPR by changing `TARGET_COL` in `train_model.py`.
- Consider the `load_ff_playerids()` crosswalk for the snap-count join — not urgent at 0.9% unmatched, but it removes a whole class of silent failure.