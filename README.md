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
| 2022       | +0.056       | +0.064       |
| 2021       | +0.069       | +0.077       |
| 2020       | +0.070       | +0.083       |
| **2019**   | **+0.077**   | **+0.090**   |

(Mean of 3 seeds, current feature set.)

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
- **The injury signal comes from practice participation, not game status.** An "is out" flag is structurally dead here: a player ruled Out doesn't play, so he has no `player_stats` row and therefore no feature row — the old `is_doubtful_or_out` fired 4 times in 40,330 rows. `practice_status_code` replaces it (0 not on the report, 1 listed/full, 2 limited, 3 did not participate), which is recorded for players who *do* play and reaches **17.3%** of rows against the game designation's 4.3%. Both are published before kickoff, so neither leaks. The signal is real and monotonic against a player's own 3-game average: **+0.08 / -0.39 / -1.05 / -1.62** PPR by code. It does *not*, however, measurably improve ranking accuracy — see below.
- **Rolling windows cross the season boundary.** Features group by `player_id` alone, so week 1 of a season carries form from the end of the previous one. Defensible — better than starting every season blind — but it means an offseason team change is invisible to the feature. `games_played` counts across seasons for the same reason.

## Results

Two held-out seasons, each trained on every prior season. The metric is mean
Spearman rho within each (week, position) group — the ordering is what you act
on, so the absolute point totals are deliberately discarded.

### 2025 — the primary result

Trained on 2019-2024 (33,067 rows), tested on 2025 (5,914 rows).

| Position | n    | MAE  | Model rho | Baseline rho | Lift   |
|----------|------|------|-----------|--------------|--------|
| QB       | 650  | 7.00 | 0.413     | 0.343        | +0.069 |
| RB       | 1539 | 4.64 | 0.701     | 0.626        | +0.075 |
| WR       | 2461 | 4.41 | 0.646     | 0.502        | +0.144 |
| TE       | 1264 | 3.85 | 0.569     | 0.473        | +0.096 |
| **Mean** |      |      | **0.582** | **0.486**    | **+0.096** |

In decisions rather than correlations: over all 275,564 same-week same-position
head-to-heads, the model picks the higher scorer **73.7%** of the time against
the baseline's **67.2%**.

### 2024 — second held-out season

Trained on 2019-2023. Mean model rho **0.587** vs. baseline **0.507**, lift
**+0.081**, ahead in 57 of 72 groups. Two independent seasons landing within
0.005 of each other is the reason to believe the edge is real.

### Is the lift significant?

`--significance` runs paired tests on the 72 per-group differences (each group
yields a model rho and a baseline rho over identical players, so they pair).

| | 2025 | 2024 |
|---|---|---|
| Mean lift | +0.096 | +0.081 |
| Groups won | 59/72 | 57/72 |
| Paired *t* | 9.22, p < 0.0001 | 6.36, p < 0.0001 |
| Bootstrap 95% CI | [+0.078, +0.117] | [+0.057, +0.108] |

The CI comes from resampling whole **weeks**, not groups, because positions
within a week share the same games and aren't independent.

**Quarterback is the exception, and don't trust a single run of it.** Measured
across 12 seeds, the QB lift is +0.057 on 2025 and +0.041 on 2024, and it clears
p < 0.05 in **4 of 12 seeds on 2025 and 0 of 12 on 2024** (2025 p ranges
0.017-0.146). The table above happens to show p = 0.027 for QB because the
documented command uses a fixed seed that lands on the favourable side — that is
not a finding. Treat QB as unresolved: possibly a small real edge on 2025, no
evidence of one on 2024. Its per-seed rho has sd ~0.010 against 0.002-0.006 at
the other positions, so always average seeds before concluding anything about it. It fits the feature story: snap share is the model's
strongest signal for skill players and carries no information for quarterbacks,
who never leave the field.

### The startable tier is harder, but the model still works there

Rho depends heavily on how wide a pool it spans, so the headline number is not
the whole story. Measured on a **neutral** tier — top-K by prior form, chosen
without reference to the model — for 2025:

| Signal | Top-12 | Top-24 | Full pool |
|--------|--------|--------|-----------|
| **Model** | **+0.208** | **+0.264** | **+0.582** |
| Baseline (last week's points) | +0.100 | +0.157 | +0.486 |
| Prior form (`fp_ppr_roll3`) | +0.159 | +0.221 | +0.562 |

Ordering a full position pool that runs from zero-point WR5s to a 40-point
ceiling is largely easy, and much of the headline comes from that easy part.
Among the dozen players you would genuinely consider starting the task is far
harder — but the model still roughly doubles the baseline there (+0.208 vs
+0.100), a lift comparable to its full-pool lift.

**How you define the tier matters enormously.** An earlier version of this
section measured the tier as "the model's own top 12" and reported a near-zero
correlation. That was a measurement artifact: conditioning on high predicted
values compresses predicted variance, so rho falls mechanically whether or not
the model is any good. Select the pool with something independent of the model
before drawing conclusions from it.

### How much headroom is left

| Oracle signal (not knowable pre-game) | Top-12 | Top-24 |
|---------------------------------------|--------|--------|
| Actual targets | **+0.572** | +0.569 |
| Actual snap share | +0.262 | +0.331 |

The tier is not noise-dominated. Knowing this week's target count would order it
at +0.572, roughly triple what the model manages. The binding constraint is
forecasting usage, not irreducible randomness — which is why the opportunity
features below were the next thing tried, and where further work should go.

## Opportunity features: the one change that moved the tier

`target_share`, `air_yards_share`, `wopr` and `receiving_air_yards` were already
in the raw nflverse pull and unused. They are opportunity-share metrics — snap
share says a player was on the field, target share says the offense actually
went to him — and they are 100% populated for every position. Adding them (7
rolling sources to 11, 22 features to 30) does this, over 6 seeds:

| | Top-12 | Top-24 | Full pool |
|---|--------|--------|-----------|
| 2024 | **+0.024** | +0.016 | +0.000 |
| 2025 | **+0.008** | +0.006 | +0.002 |

They pay in the startable tier and do nothing on the full pool, which is exactly
the shape you want — the full pool was never the hard part. By position on the
full pool, WR (+0.005) and TE (+0.005) gain in both seasons while RB loses
slightly, consistent with these being receiving-usage metrics.

This is the only change measured in this project so far that moved tier
performance. `racr` and the EPA columns were left out: they are efficiency
ratios rather than opportunity, they regress hard, and they are only 69-89%
populated for skill positions.

## Capacity concentration: the thing that doesn't work

Before adding features, the cheaper hypothesis was tested — that the model wastes
capacity on easy rows and should focus on the startable tier. It is wrong, and
consistently so:

| Scheme | 2024 top-12 | 2025 top-12 | Full pool (2025) |
|---|---|---|---|
| **Uniform (control)** | **+0.164** | **+0.182** | +0.576 |
| Weight x3 top-12 | +0.154 | +0.178 | +0.577 |
| Weight x10 top-12 | +0.157 | +0.175 | +0.582 |
| Train only top-24 | +0.130 | +0.152 | +0.484 |
| Train only top-12 | +0.129 | +0.160 | +0.190 |

All 10 comparisons negative, with a clean dose-response: the harder you
concentrate, the worse the tier gets. The "easy" rows are not wasted capacity —
they are what teaches the model the feature-to-points relationship that orders
the good players. This is also the main evidence against `LambdaRank` being the
next move, since it reallocates gradient toward the top of each list in a more
sophisticated version of the same idea.

## A live feature that doesn't pay

Replacing `is_doubtful_or_out` with `practice_status_code` turned a constant
into a real signal — and left model accuracy exactly where it was. Measured over
8 seeds, adding it is worth:

| Position | 2024 | 2025 |
|----------|------|------|
| QB       | +0.0020 | +0.0034 |
| RB       | -0.0014 | +0.0009 |
| WR       | +0.0014 | +0.0004 |
| TE       | -0.0012 | -0.0007 |

Every one of those is smaller than the seed-to-seed standard deviation of the
same measurement. Restricting to the head-to-heads where exactly one player is
on the injury report — precisely where it should help — gives 72.8% vs 72.9% on
2024 and 73.3% vs 73.2% on 2025. Nothing.

The likely reason is size, not validity: a limited-practice week costs about a
point against a player's own average, while the gap between two players in a
start/sit decision is usually several. A penalty that small rarely flips an
ordering, which is all rho measures.

Worth recording rather than quietly dropping, because "the feature is broken" and
"the feature is fine and doesn't matter" are different conclusions, and only the
second one is true here.

## Next steps (from the original notes)

- **More usage signal.** The oracle probe says knowing this week's targets would order the tier at +0.572 against the model's +0.208, so usage forecasting is where the remaining headroom is. Team-level pace and pass rate, red-zone share, and routes run are the obvious next candidates.
- **Position-specific feature sets.** The opportunity features help WR/TE and slightly hurt RB. One feature list for all four positions is leaving something on the table.
- **`LambdaRank`, but not yet.** It optimizes the ordering the model is actually graded on, which is a correctness argument on its own. Deprioritized because the capacity-concentration probe above suggests reweighting toward the top of the list hurts here. Worth revisiting once the features carry more tier signal.
- **Decide whether to keep `practice_status_code`.** It is a genuine signal that does not help the metric (see "A live feature that doesn't pay" below). Keeping it costs nothing and gives a better injury feature somewhere to grow from; dropping it takes the model to 21 features and loses nothing measurable.
- **Report averaged over seeds.** Single-seed numbers move by ~0.01 at QB, which is the same size as the effects being argued about. Averaging 5-10 seeds before recording a result would stop seed noise being read as signal.
- **Work out why QB is flat.** No significant lift in either held-out season. QBs have no snap-share signal; the model may need quarterback-specific features (pressure rate, opponent pass defense, designed-run share) rather than the shared feature set.
- Decide final target: currently trains on `fantasy_points_ppr`; swap for standard/half-PPR by changing `TARGET_COL` in `train_model.py`.
- Consider the `load_ff_playerids()` crosswalk for the snap-count join — not urgent at 0.9% unmatched, but it removes a whole class of silent failure.