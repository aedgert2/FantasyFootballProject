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

## Results — read this before quoting any number

**The model does not beat sorting players by their 5-game scoring average.**
That is the headline, and it took a baseline change to see it.

### The honest comparison

Baseline is `fp_ppr_roll5` — one of the model's own input features. Both seasons,
seed-42 run of the documented command:

| Test season | Model rho | Baseline rho | Lift | Groups won | p |
|---|---|---|---|---|---|
| 2025 | 0.583 | 0.572 | **+0.011** | 38/72 | 0.151 |
| 2024 | 0.590 | 0.593 | **-0.003** | 33/72 | 0.741 |

Neither is significant. The 2025 bootstrap CI is [-0.004, +0.027] and 2024's
straddles zero too. WR is the one position with a reliable edge (+0.019 on 2025,
p = 0.003, ahead in 15 of 18 weeks); RB and TE are flat or slightly negative in
both seasons.

Per position, over 10 seeds, with the number of seeds reaching p < 0.05:

| Position | 2024 lift | 2024 sig | 2025 lift | 2025 sig |
|---|---|---|---|---|
| QB | -0.009 | 0/10 | +0.016 | 0/10 |
| RB | **-0.016** | **7/10 (worse)** | +0.001 | 0/10 |
| WR | +0.008 | 0/10 | **+0.020** | **10/10** |
| TE | -0.008 | 0/10 | -0.013 | 0/10 |

WR is the only position with a positive lift in both seasons, and it is only
reliably significant in one. RB in 2024 is *significantly worse* than the
rolling average in 7 of 10 seeds.

### Against trivial heuristics, on identical rows

| Method | 2024 top-12 | 2025 top-12 | 2024 full | 2025 full |
|---|---|---|---|---|
| LightGBM (30 features) | +0.187 | +0.201 | +0.576 | +0.574 |
| **5-game average** | **+0.212** | **+0.222** | +0.581 | +0.570 |
| Season-to-date average | +0.180 | **+0.228** | **+0.588** | **+0.579** |
| 3-game average | +0.132 | +0.168 | +0.571 | +0.559 |
| Last week only (old baseline) | +0.059 | +0.101 | +0.505 | +0.487 |

In the startable tier — the only place a start/sit tool matters — a rolling mean
beats the model in both seasons.

### Why this was hidden for so long

The original baseline was "start whoever scored more last week," which scores
+0.059 to +0.101 in the tier against the rolling mean's +0.212 to +0.222. Every
result measured against it looked excellent: a +0.096 lift, p < 0.0001,
bootstrap CIs well clear of zero. All of that arithmetic was correct and none of
it meant what it appeared to. **A weak baseline is the most expensive mistake in
this project's history** — it validated months of work that was not adding value.

### When the model disagrees with the average, who wins?

If the model can't beat the rolling mean on aggregate, it might still be right
where it disagrees. Tested pairwise: for every pair of players the model and the
rolling mean order differently, who picked the higher scorer?

| Population | Disagreements | Model wins |
|---|---|---|
| All players, 2024 | 13.4% of pairs | 49.6% |
| All players, 2025 | 12.5% of pairs | 51.6% |
| **Startable tier, 2024** | 33.0% of pairs | **49.3%** |
| **Startable tier, 2025** | 30.8% of pairs | **48.9%** |

Coin flips, and slightly *below* even in the tier. Filtering to the model's most
confident quarter of disagreements lifts tier accuracy to ~53%, but pairs within
a week are not independent so that is weaker than its sample size suggests.

On the full pool, high-confidence disagreements do reach 56-58% — but the full
pool is the easy problem (separating startable players from WR5s), not a lineup
decision.

### What the project honestly is

A clean, well-tested pipeline that reproduces the accuracy of a 5-game rolling
average, with a possible small edge at wide receiver. Week-to-week
autocorrelation of PPR points is r = 0.487 (r-squared 0.237), so roughly
three-quarters of weekly fantasy scoring is not predictable from prior
production at all. That ceiling is the reason, not a modelling defect: dropping
the weakest feature changes nothing, and a minimal 4-feature model is worse.

## Routes run and TPRR: derived, validated, and it changed nothing

The best remaining idea was targets per route run — the metric that separates
"ran 30 routes for 3 looks" from "ran 8 routes for 3 looks", which neither snap
share nor target share can do. It is derivable: `load_participation` lists every
player on the field for each play and covers 2019-2025, and `load_pbp` says
which plays were passes. Both key off gsis id, so no name matching.

The derivation validates well. Median routes per game come out at WR 24-28, TE
18-20, RB 13-15 and QB 35-39 (dropbacks) across all seven seasons, no
player-week exceeds a TPRR of 1, and the leaderboards reproduce the known ones —
Michael Thomas .292 leading 2019 (his 149-catch year), Puka Nacua .341 leading
2025. Median WR TPRR is .174 in both 2019 and 2025.

It does not help. Over 8 seeds:

| | full pool | top-24 | top-12 |
|---|---|---|---|
| 2024 | +0.002 | +0.001 | +0.013 |
| 2025 | +0.004 | +0.006 | -0.001 |

By position the only consistent gain is **QB, +0.015 in both seasons** — because
for a quarterback "routes" is really dropbacks, a pass-volume signal. For the
pass catchers it was built for, it is +0.002 to -0.005. The model with routes
still loses to the 5-game average in the startable tier in both seasons (0.196
vs 0.215 on 2024; 0.188 vs 0.218 on 2025).

Kept because the derivation is correct, cheap and well-tested, and TPRR is worth
having in the feature table regardless. Not kept because it earned its place.

**Two traps worth knowing if you touch this code:** participation's
`offense_positions` column is 100% blank before 2023, so filtering on it
silently produces zero routes for four full seasons — the derivation therefore
counts every player on the field and lets the merge filter. And the pass flag
must come from pbp: participation's own `time_to_throw` has 99.9% precision but
misses 12% of pass plays (sacks and scrambles), biased toward mobile
quarterbacks.

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

- **Probably nothing in modelling.** Routes/TPRR was the best remaining public-data idea and it moved nothing. Combined with the r-squared 0.237 ceiling, the honest read is that public box-score data supports a rolling average and not much more. Remaining ideas need data we do not have: market player-prop lines, or archived weekly consensus rankings (start archiving `load_ff_rankings` now — it is a current snapshot only, so history is not recoverable later).
- **Position-specific feature sets.** The opportunity features help WR/TE and slightly hurt RB. One feature list for all four positions is leaving something on the table.
- **`LambdaRank`, but not yet.** It optimizes the ordering the model is actually graded on, which is a correctness argument on its own. Deprioritized because the capacity-concentration probe above suggests reweighting toward the top of the list hurts here. Worth revisiting once the features carry more tier signal.
- **Decide whether to keep `practice_status_code`.** It is a genuine signal that does not help the metric (see "A live feature that doesn't pay" below). Keeping it costs nothing and gives a better injury feature somewhere to grow from; dropping it takes the model to 21 features and loses nothing measurable.
- **Report averaged over seeds.** Single-seed numbers move by ~0.01 at QB, which is the same size as the effects being argued about. Averaging 5-10 seeds before recording a result would stop seed noise being read as signal.
- **Work out why QB is flat.** No significant lift in either held-out season. QBs have no snap-share signal; the model may need quarterback-specific features (pressure rate, opponent pass defense, designed-run share) rather than the shared feature set.
- Decide final target: currently trains on `fantasy_points_ppr`; swap for standard/half-PPR by changing `TARGET_COL` in `train_model.py`.
- Consider the `load_ff_playerids()` crosswalk for the snap-count join — not urgent at 0.9% unmatched, but it removes a whole class of silent failure.