"""Stage 4: train one LightGBM regressor per position and score the ranking.

Start/sit is a ranking problem, so the headline metric is the mean Spearman
correlation between predicted and actual order *within* each
(season, week, position) group. MAE is reported too, but a model that nails
point totals and gets the order wrong is useless here.

Every run is compared against ``dumb_baseline_rank`` — "start whoever scored
more last week" — built from the same lagged column the features come from.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

POSITIONS = ("QB", "RB", "WR", "TE")
TARGET_COL = "fantasy_points_ppr"
BASELINE_COL = "fp_ppr_shifted"

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
    "implied_team_total",
    "spread_line",
    "total_line",
    "is_home",
    "def_fp_allowed_roll",
    "games_played",
    "is_questionable",
    "is_doubtful_or_out",
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


def mean_group_spearman(df, score_col):
    """Mean Spearman rho between ``score_col`` and actual points, per week/position."""
    rhos = []
    for _, group in df.groupby(["season", "week", "position"]):
        valid = group[[score_col, TARGET_COL]].dropna()
        if len(valid) < 3 or valid[score_col].nunique() < 2:
            continue
        rho = spearmanr(valid[score_col], valid[TARGET_COL]).statistic
        if not np.isnan(rho):
            rhos.append(rho)
    if not rhos:
        return np.nan, 0
    return float(np.mean(rhos)), len(rhos)


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
    # "Start whoever scored more last week."
    test_pos["dumb_baseline_rank"] = (
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
    rows = []
    for position in POSITIONS:
        model, scored = train_position(train_df, test_df, position, args.target)
        if scored is None:
            continue
        predictions.append(scored)

        mae = float(np.mean(np.abs(scored["predicted_points"] - scored[args.target])))
        model_rho, n_groups = mean_group_spearman(scored, "predicted_points")
        base_rho, _ = mean_group_spearman(scored, BASELINE_COL)
        rows.append(
            {
                "position": position,
                "n_test": len(scored),
                "mae": mae,
                "model_rho": model_rho,
                "baseline_rho": base_rho,
                "lift": model_rho - base_rho,
                "weeks": n_groups,
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

    out = pd.concat(predictions, ignore_index=True)
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
        "dumb_baseline_rank",
    ]
    out[[c for c in keep if c in out.columns]].to_parquet(out_path, index=False)
    print(f"\n[train_model] wrote {len(out):,} predictions -> {out_path}")


if __name__ == "__main__":
    main()
