"""Stage 4: train one LightGBM regressor per position and score the ranking.

Start/sit is a ranking problem, so the headline metric is the mean Spearman
correlation between predicted and actual order *within* each
(season, week, position) group. MAE is reported too, but a model that nails
point totals and gets the order wrong is useless here.

Every run is compared against ``baseline_rank`` — "start whoever has the better
5-game average". That baseline is deliberately strong: it is one of the model's
own input features, so the comparison asks whether 30 features and four gradient
boosters actually beat one line of pandas. Measured on 2024 and 2025 they
roughly tie, and the rolling mean wins inside the startable tier.

An earlier version used last week's points instead. That baseline is far too
weak — it scores ~+0.06 to +0.10 in the tier against the rolling mean's ~+0.21 —
and it flattered every result measured against it.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import binomtest, spearmanr, ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

POSITIONS = ("QB", "RB", "WR", "TE")
TARGET_COL = "fantasy_points_ppr"
BASELINE_COL = "fp_ppr_roll5"   # the bar to clear: sort by 5-game average

# KEEP IN SYNC with ROLL_COLS / ROLL_WINDOWS in build_features.py.
FEATURE_COLS = [
    "fp_ppr_roll3",
    "fp_ppr_roll5",
    "tgt_roll3",
    "tgt_roll5",
    "car_roll3",
    "car_roll5",
    "rec_yds_roll3",
    "rec_yds_roll5",
    "rush_yds_roll3",
    "rush_yds_roll5",
    "pass_yds_roll3",
    "pass_yds_roll5",
    "snap_pct_roll3",
    "snap_pct_roll5",
    "tgt_share_roll3",
    "tgt_share_roll5",
    "ay_share_roll3",
    "ay_share_roll5",
    "wopr_roll3",
    "wopr_roll5",
    "rec_ay_roll3",
    "rec_ay_roll5",
    "routes_roll3",
    "routes_roll5",
    "tprr_roll3",
    "tprr_roll5",
    "implied_team_total",
    "spread_line",
    "total_line",
    "is_home",
    "def_fp_allowed_roll",
    "games_played",
    "is_questionable",
    "practice_status_code",
]

LGBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    random_state=42,
    verbose=-1,
)


def group_rho_table(df, score_cols, target_col=TARGET_COL):
    """Spearman rho per (season, week, position), one column per score.

    A group is kept only if it is usable for *every* score column, so the rhos
    come out paired — each row compares the model and the baseline over exactly
    the same players. That pairing is what makes the lift testable.
    """
    rows = []
    for (season, week, position), group in df.groupby(["season", "week", "position"]):
        valid = group[list(score_cols) + [target_col]].dropna()
        if len(valid) < 3 or any(valid[c].nunique() < 2 for c in score_cols):
            continue
        rhos = {c: spearmanr(valid[c], valid[target_col]).statistic for c in score_cols}
        if any(np.isnan(r) for r in rhos.values()):
            continue
        rows.append(
            {"season": season, "week": week, "position": position, "n": len(valid), **rhos}
        )
    return pd.DataFrame(rows)


def _fmt_p(p):
    return "< 0.0001" if p < 0.0001 else f"  {p:.4f}"


def significance_report(table, model_col, base_col, n_boot=5000, seed=0):
    """Report whether the lift over the baseline is distinguishable from noise.

    Runs on the paired per-group differences. The bootstrap resamples whole
    weeks rather than individual groups, because the positions within one week
    share the same underlying games and are not independent of each other.
    """
    lift = table[model_col] - table[base_col]
    n = len(lift)
    print()
    print("=== Is the lift distinguishable from noise? ===")
    if n < 3 or np.allclose(lift, 0):
        print("  too few paired groups to test")
        return

    wins = int((lift > 0).sum())
    t_stat, t_p = ttest_rel(table[model_col], table[base_col])
    sign_p = binomtest(wins, n, 0.5).pvalue

    week_means = table.assign(lift=lift).groupby(["season", "week"])["lift"].mean().to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(week_means), size=(n_boot, len(week_means)))
    boot = week_means[draws].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"  paired groups     {n}   ({len(week_means)} weeks)")
    print(f"  mean lift         {lift.mean():+.4f}   (sd {lift.std():.4f})")
    print(f"  groups won        {wins} / {n}")
    print(f"  paired t-test     t = {t_stat:6.3f}   p = {_fmt_p(t_p)}")
    try:
        w_stat, w_p = wilcoxon(table[model_col], table[base_col])
        print(f"  Wilcoxon signed   W = {w_stat:6.0f}   p = {_fmt_p(w_p)}")
    except ValueError:
        pass
    print(f"  sign test                        p = {_fmt_p(sign_p)}")
    print(
        f"  cluster bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]"
        f"   ({n_boot:,} reps, resampling whole weeks)"
    )

    print("\n  per position:")
    for position, group in table.groupby("position"):
        g_lift = group[model_col] - group[base_col]
        if len(g_lift) < 3 or np.allclose(g_lift, 0):
            print(f"    {position}  {g_lift.mean():+.4f}   (too few groups to test)")
            continue
        _, p_pos = ttest_rel(group[model_col], group[base_col])
        flag = "   not significant" if p_pos > 0.05 else ""
        print(
            f"    {position}  {g_lift.mean():+.4f}   "
            f"{int((g_lift > 0).sum())}/{len(g_lift)}   p = {_fmt_p(p_pos)}{flag}"
        )

    print(
        "\n  Groups are not fully independent — the same players recur week to\n"
        "  week — so the p-values overstate certainty. The bootstrap clusters by\n"
        "  week to absorb the dependence between positions sharing a game."
    )


def check_feature_sync(df):
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(
            "features.parquet is missing: "
            + ", ".join(missing)
            + "\nFEATURE_COLS here and ROLL_COLS in build_features.py have drifted "
            "out of sync — update both, then re-run build_features.py."
        )


def train_position(train_df, test_df, position, target_col):
    train_pos = train_df[train_df["position"] == position]
    test_pos = test_df[test_df["position"] == position].copy()

    if train_pos.empty or test_pos.empty:
        print(f"[train_model] {position}: no data, skipping")
        return None, None

    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(train_pos[FEATURE_COLS], train_pos[target_col])
    test_pos["predicted_points"] = model.predict(test_pos[FEATURE_COLS])
    test_pos["predicted_rank"] = (
        test_pos.groupby(["season", "week"])["predicted_points"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    # "Start whoever has the better 5-game average."
    test_pos["baseline_rank"] = (
        test_pos.groupby(["season", "week"])[BASELINE_COL]
        .rank(ascending=False, method="min")
    )
    return model, test_pos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-season",
        type=int,
        required=True,
        help="Hold this season out; train on every earlier season.",
    )
    parser.add_argument("--features", type=Path, default=PROCESSED_DIR / "features.parquet")
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--target",
        default=TARGET_COL,
        help="Scoring column to train on (e.g. fantasy_points for standard).",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=1,
        help="Drop player-weeks with fewer than this many prior games.",
    )
    parser.add_argument(
        "--significance",
        action="store_true",
        help="Also test whether the lift over the baseline beats chance.",
    )
    args = parser.parse_args()

    if not args.features.exists():
        raise SystemExit(f"Missing {args.features}. Run src/build_features.py first.")

    df = pd.read_parquet(args.features)
    check_feature_sync(df)

    df = df[df["games_played"] >= args.min_games]
    df = df.dropna(subset=[args.target, BASELINE_COL])

    train_df = df[df["season"] < args.test_season]
    test_df = df[df["season"] == args.test_season]
    if train_df.empty:
        raise SystemExit(f"No seasons before {args.test_season} in the feature table.")
    if test_df.empty:
        raise SystemExit(f"Season {args.test_season} is not in the feature table.")

    print(
        f"[train_model] train: {len(train_df):,} rows "
        f"(seasons {sorted(int(s) for s in train_df['season'].unique())})  |  "
        f"test: {len(test_df):,} rows (season {args.test_season})"
    )

    predictions = []
    for position in POSITIONS:
        model, scored = train_position(train_df, test_df, position, args.target)
        if scored is None:
            continue
        predictions.append(scored)

    out = pd.concat(predictions, ignore_index=True)
    table = group_rho_table(out, ["predicted_points", BASELINE_COL], args.target)

    rows = []
    for position in POSITIONS:
        scored = out[out["position"] == position]
        groups = table[table["position"] == position]
        if scored.empty or groups.empty:
            continue
        model_rho = groups["predicted_points"].mean()
        base_rho = groups[BASELINE_COL].mean()
        rows.append(
            {
                "position": position,
                "n_test": len(scored),
                "mae": float(np.mean(np.abs(scored["predicted_points"] - scored[args.target]))),
                "model_rho": model_rho,
                "baseline_rho": base_rho,
                "lift": model_rho - base_rho,
                "weeks": len(groups),
            }
        )

    summary = pd.DataFrame(rows)
    print()
    print(f"=== Season {args.test_season} — rank correlation within (week, position) ===")
    print(
        summary.to_string(
            index=False,
            float_format=lambda v: f"{v:6.3f}",
        )
    )
    print()
    print(
        f"mean model rho:    {summary['model_rho'].mean():.3f}\n"
        f"mean baseline rho: {summary['baseline_rho'].mean():.3f}\n"
        f"mean lift:         {summary['lift'].mean():+.3f}"
    )

    if args.significance:
        significance_report(table, "predicted_points", BASELINE_COL)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"predictions_{args.test_season}.parquet"
    keep = [
        "player_id",
        "player_name",
        "position",
        "team",
        "opponent_team",
        "season",
        "week",
        args.target,
        "position_week_rank",
        "predicted_points",
        "predicted_rank",
        "baseline_rank",
    ]
    out[[c for c in keep if c in out.columns]].to_parquet(out_path, index=False)
    print(f"\n[train_model] wrote {len(out):,} predictions -> {out_path}")


if __name__ == "__main__":
    main()
