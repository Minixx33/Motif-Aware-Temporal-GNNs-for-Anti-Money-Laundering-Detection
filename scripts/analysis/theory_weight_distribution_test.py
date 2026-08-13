"""
theory_weight_distribution_test.py
---------------------------------------------------------------------------
Falsification / placebo control (review feedback Aug 4 2026, sec 4.3):
"randomized theory weights vs. our current hand-set weights" -- done as a
genuine DISTRIBUTIONAL test rather than a single random draw.

Full GNN retraining for every random weight draw is not feasible on this
project's compute budget. This script gets a real distribution for free
instead: the RAT/SLT composite score is just a weighted sum of already-
computed component columns, so for each of N random weight draws we can
recompute the composite score and its own raw AUPR against the label in
pure numpy/pandas -- no model training, seconds per thousand draws -- and
see where the theory-motivated weights rank within that whole distribution.

This raw-score AUPR is a weaker signal than a trained GNN's AUPR (it's an
unlearned linear ranking, not a fitted classifier), but it is a legitimate,
well-powered placebo test on its own: if the theory-motivated weights don't
even outperform a large random-weight distribution at the raw-score level,
that is strong evidence against the specific weighting being load-bearing.
Recommended follow-up (small, bounded cost): retrain the actual GNN on
2-3 representative random-weight draws (e.g. the median and the
best-performing one from this distribution) via --random_weights on the
injectors, to confirm the raw-score ranking predicts GNN-level results in
the same direction. That is a separate, deliberately small step -- this
script is the free, well-powered part.

Usage:
    python scripts/analysis/theory_weight_distribution_test.py \
        --theory rat \
        --input_csv ibm_transcations_datasets/RAT/HI-Small_Trans_RAT_pristine.csv \
        --n_draws 1000 --seed 7 \
        --output_json results_placebo/rat_weight_distribution.json

    python scripts/analysis/theory_weight_distribution_test.py \
        --theory slt \
        --input_csv ibm_transcations_datasets/SLT/HI-Small_Trans_SLT_pristine.csv \
        --n_draws 1000 --seed 7 \
        --output_json results_placebo/slt_weight_distribution.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

EPS = 1e-8

LABEL_COL = "Is Laundering"

# ---- RAT composite-score component columns (must match rat_injector.py) ----
RAT_OFFENDER_COLS = [
    "RAT_src_amount_z_pos", "RAT_src_out_deg_norm", "RAT_src_burst_norm",
    "RAT_is_off_hours", "RAT_src_entity_acct_norm",
]
RAT_TARGET_COLS = [
    "RAT_dst_amount_z_pos", "RAT_dst_in_deg_norm", "RAT_dst_age_norm_inv",  # computed below
    "RAT_dst_entity_acct_norm",
]
RAT_GUARDIAN_COLS = [
    "RAT_is_off_hours", "RAT_is_weekend", "RAT_is_cross_bank",
    "RAT_combined_burst", "RAT_same_entity",
]

# ---- SLT composite-score component columns (must match slt_injector.py) ----
SLT_SRC_COLS = [
    "SLT_src_susp_nbr_ratio_lag1", "SLT_src_susp_amt_share_lag1",
    "SLT_src_strong_tie_susp_ratio_lag1", "SLT_src_exposure_delta_pos",  # computed below
    "SLT_src_cum_exposure_7d_norm",
]
SLT_DST_COLS = [
    "SLT_dst_susp_nbr_ratio_lag1", "SLT_dst_susp_amt_share_lag1",
    "SLT_dst_strong_tie_susp_ratio_lag1", "SLT_dst_exposure_delta_pos",
    "SLT_dst_cum_exposure_7d_norm",
]


def _norm_by_quantile(series, q=0.95):
    s = series.astype(float)
    qv = s.quantile(q)
    if not np.isfinite(qv) or qv <= 0:
        qv = s.max()
    if not np.isfinite(qv) or qv <= 0:
        return pd.Series(0.0, index=series.index)
    return (s / qv).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)


def load_rat(input_csv):
    df = pd.read_csv(input_csv, low_memory=False)
    # RAT_target_score's age term is (1 - norm_by_quantile(dst_age_days)),
    # not a column rat_injector.py saves directly -- reconstruct it here.
    df["RAT_dst_age_norm_inv"] = 1 - _norm_by_quantile(df["dst_age_days"].fillna(0))
    offender = df[RAT_OFFENDER_COLS].to_numpy(dtype=np.float64)
    target = df[RAT_TARGET_COLS].to_numpy(dtype=np.float64)
    guardian = df[RAT_GUARDIAN_COLS].to_numpy(dtype=np.float64)
    y = df[LABEL_COL].to_numpy()
    return offender, target, guardian, y


def rat_score(offender_mat, target_mat, guardian_mat, w_off, w_tar, w_gua):
    off = offender_mat @ w_off
    tar = target_mat @ w_tar
    gua = guardian_mat @ w_gua
    score = ((off + EPS) * (tar + EPS) * (gua + EPS)) ** (1 / 3)
    return np.clip(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0), 0, 1)


def load_slt(input_csv):
    df = pd.read_csv(input_csv, low_memory=False)
    df["SLT_src_exposure_delta_pos"] = df["SLT_src_exposure_delta"].clip(lower=0)
    df["SLT_dst_exposure_delta_pos"] = df["SLT_dst_exposure_delta"].clip(lower=0)
    src = df[SLT_SRC_COLS].to_numpy(dtype=np.float64)
    dst = df[SLT_DST_COLS].to_numpy(dtype=np.float64)
    y = df[LABEL_COL].to_numpy()
    return src, dst, y


def slt_score(src_mat, dst_mat, w_src, w_dst):
    src_score = np.clip(src_mat @ w_src, 0, 1)
    dst_score = np.clip(dst_mat @ w_dst, 0, 1)
    score = 0.55 * src_score + 0.45 * dst_score
    return np.clip(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0), 0, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Distributional placebo test: raw composite-score AUPR "
                     "across many random weight draws vs. the theory-motivated weights."
    )
    parser.add_argument("--theory", choices=["rat", "slt"], required=True)
    parser.add_argument("--input_csv", type=str, required=True,
                        help="A CSV with the theory's component columns already "
                             "computed (e.g. the --dump_pristine snapshot).")
    parser.add_argument("--n_draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.theory == "rat":
        print(f"Loading RAT component columns from: {args.input_csv}")
        offender, target, guardian, y = load_rat(args.input_csv)

        # Theory-motivated (renormalized post-pattern-flag-removal) weights,
        # must match rat_injector.py's defaults.
        w_off_theory = np.array([0.30, 0.20, 0.20, 0.10, 0.10]) / 0.90
        w_tar_theory = np.array([0.35, 0.25, 0.15, 0.15]) / 0.90
        w_gua_theory = np.array([0.30, 0.20, 0.20, 0.20, 0.10])

        theory_scores = rat_score(offender, target, guardian, w_off_theory, w_tar_theory, w_gua_theory)
        theory_aupr = float(average_precision_score(y, theory_scores))

        print(f"Theory-weighted raw-score AUPR: {theory_aupr:.4f}")
        print(f"Drawing {args.n_draws} random weight vectors (Dirichlet)...")

        random_auprs = np.empty(args.n_draws, dtype=np.float64)
        for i in range(args.n_draws):
            w_off = rng.dirichlet(np.ones(len(RAT_OFFENDER_COLS)))
            w_tar = rng.dirichlet(np.ones(len(RAT_TARGET_COLS)))
            w_gua = rng.dirichlet(np.ones(len(RAT_GUARDIAN_COLS)))
            s = rat_score(offender, target, guardian, w_off, w_tar, w_gua)
            random_auprs[i] = average_precision_score(y, s)

    else:
        print(f"Loading SLT component columns from: {args.input_csv}")
        src, dst, y = load_slt(args.input_csv)

        w_src_theory = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
        w_dst_theory = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

        theory_scores = slt_score(src, dst, w_src_theory, w_dst_theory)
        theory_aupr = float(average_precision_score(y, theory_scores))

        print(f"Theory-weighted raw-score AUPR: {theory_aupr:.4f}")
        print(f"Drawing {args.n_draws} random weight vectors (Dirichlet)...")

        random_auprs = np.empty(args.n_draws, dtype=np.float64)
        for i in range(args.n_draws):
            w_src = rng.dirichlet(np.ones(len(SLT_SRC_COLS)))
            w_dst = rng.dirichlet(np.ones(len(SLT_DST_COLS)))
            s = slt_score(src, dst, w_src, w_dst)
            random_auprs[i] = average_precision_score(y, s)

    percentile_rank = float((random_auprs < theory_aupr).mean() * 100)
    result = {
        "theory": args.theory,
        "input_csv": args.input_csv,
        "n_draws": args.n_draws,
        "seed": args.seed,
        "theory_weighted_raw_aupr": theory_aupr,
        "random_draws_raw_aupr": {
            "mean": float(random_auprs.mean()),
            "std": float(random_auprs.std()),
            "min": float(random_auprs.min()),
            "max": float(random_auprs.max()),
            "p50": float(np.percentile(random_auprs, 50)),
            "p95": float(np.percentile(random_auprs, 95)),
            "p99": float(np.percentile(random_auprs, 99)),
        },
        "theory_weight_percentile_within_random_distribution": percentile_rank,
    }

    print()
    print(f"Random draws: mean AUPR={result['random_draws_raw_aupr']['mean']:.4f}  "
          f"std={result['random_draws_raw_aupr']['std']:.4f}  "
          f"max={result['random_draws_raw_aupr']['max']:.4f}")
    print(f"Theory weights land at the {percentile_rank:.1f}th percentile "
          f"of the random-weight distribution.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
