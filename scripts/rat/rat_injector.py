"""
RAT + Motif Feature Injection for IBM HI-Small (Multiplicative Version)
Fixed Version – No NaN Threshold, Correct Injection
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ===================== CLI ARGS =====================

_parser = argparse.ArgumentParser(description="RAT feature injector")
_parser.add_argument("--data_dir",    type=str, default=None,
                     help="Path to ibm_transcations_datasets/ (default: auto-resolved)")
_parser.add_argument("--trans_file",  type=str, default="HI-Small_Trans.csv",
                     help="Transaction CSV filename inside data_dir (default: HI-Small_Trans.csv)")
_parser.add_argument("--acct_file",   type=str, default="HI-Small_accounts.csv",
                     help="Accounts CSV filename inside data_dir (default: HI-Small_accounts.csv)")
_parser.add_argument("--patterns_file", type=str, default="HI-Small_Patterns.txt",
                     help="Patterns txt filename inside data_dir (default: HI-Small_Patterns.txt)")
_parser.add_argument("--output_dir",  type=str, default=None,
                     help="Output directory for injected CSVs (default: <data_dir>/RAT)")
_parser.add_argument("--dump_pristine", action="store_true",
                     help="Also write a snapshot of the data BEFORE any intensity "
                          "boosting is applied (RAT_*/motif_* features at their "
                          "pristine, never-boosted values). Same row count/order "
                          "as the low/medium/high CSVs -- used for post-hoc "
                          "leakage/robustness checks, not part of normal training.")

# ---- Component weights (offender / target / guardian-weakness) ----
# Pattern-flag terms (RAT_src_pattern_flag / RAT_dst_pattern_flag) have been
# REMOVED from these formulas: they were derived from the AMLworld simulator's
# ground-truth laundering-pattern export, which would not be available to a
# real investigator before prediction (review feedback, Aug 4 2026, sec 3.1).
# The remaining weights below are the original values renormalized to sum to
# 1 within each component (offender/target lost a 0.10 term each; guardian
# weakness never used the pattern flag, so its weights are unchanged).
_parser.add_argument("--w_off_amt",      type=float, default=0.30 / 0.90)
_parser.add_argument("--w_off_outdeg",   type=float, default=0.20 / 0.90)
_parser.add_argument("--w_off_burst",    type=float, default=0.20 / 0.90)
_parser.add_argument("--w_off_offhours", type=float, default=0.10 / 0.90)
_parser.add_argument("--w_off_entity",   type=float, default=0.10 / 0.90)
_parser.add_argument("--w_tar_amt",      type=float, default=0.35 / 0.90)
_parser.add_argument("--w_tar_indeg",    type=float, default=0.25 / 0.90)
_parser.add_argument("--w_tar_age",      type=float, default=0.15 / 0.90)
_parser.add_argument("--w_tar_entity",   type=float, default=0.15 / 0.90)
_parser.add_argument("--w_gua_offhours",  type=float, default=0.30)
_parser.add_argument("--w_gua_weekend",   type=float, default=0.20)
_parser.add_argument("--w_gua_crossbank", type=float, default=0.20)
_parser.add_argument("--w_gua_burst",     type=float, default=0.20)
_parser.add_argument("--w_gua_entity",    type=float, default=0.10)

# ---- Falsification / placebo controls (review feedback sec 4.3) ----
_parser.add_argument("--random_weights", action="store_true",
                     help="Placebo control: ignore the --w_* values above and "
                          "draw random positive weights (Dirichlet, summing to "
                          "1 within each of the offender/target/guardian "
                          "components) instead of the theory-motivated weights. "
                          "Reproducible via --weight_seed. Tests whether the "
                          "specific theory-derived weighting matters versus any "
                          "reasonable combination of the same features.")
_parser.add_argument("--weight_seed", type=int, default=42)
_parser.add_argument("--selection", choices=["score", "random"], default="score",
                     help="Placebo control: 'score' (default) selects the top "
                          "laundering transactions by RAT_score, as before. "
                          "'random' selects a random sample of the same size "
                          "instead (same boost mechanics applied afterward), to "
                          "test whether score-based selection is doing anything "
                          "beyond boosting an arbitrary subset of positives.")
_parser.add_argument("--selection_seed", type=int, default=123)
_parser.add_argument("--intensities", type=str, default="low,medium,high",
                     help="Comma/space separated subset of intensities to produce "
                          "(default: all three, e.g. --intensities medium). "
                          "Matches slt_injector.py's --intensities flag.")

_args = _parser.parse_args()

if _args.random_weights:
    _wrng = np.random.default_rng(_args.weight_seed)
    _W_OFF = _wrng.dirichlet(np.ones(5))
    _W_TAR = _wrng.dirichlet(np.ones(4))
    _W_GUA = _wrng.dirichlet(np.ones(5))
    print(f"[RAT] --random_weights set (seed={_args.weight_seed}):")
    print(f"  offender (amt,outdeg,burst,offhours,entity)        = {_W_OFF}")
    print(f"  target   (amt,indeg,age,entity)                    = {_W_TAR}")
    print(f"  guardian (offhours,weekend,crossbank,burst,entity) = {_W_GUA}")
else:
    _W_OFF = np.array([_args.w_off_amt, _args.w_off_outdeg, _args.w_off_burst,
                        _args.w_off_offhours, _args.w_off_entity])
    _W_TAR = np.array([_args.w_tar_amt, _args.w_tar_indeg, _args.w_tar_age, _args.w_tar_entity])
    _W_GUA = np.array([_args.w_gua_offhours, _args.w_gua_weekend, _args.w_gua_crossbank,
                        _args.w_gua_burst, _args.w_gua_entity])

W_OFF_AMT, W_OFF_OUTDEG, W_OFF_BURST, W_OFF_OFFHOURS, W_OFF_ENTITY = _W_OFF
W_TAR_AMT, W_TAR_INDEG, W_TAR_AGE, W_TAR_ENTITY = _W_TAR
W_GUA_OFFHOURS, W_GUA_WEEKEND, W_GUA_CROSSBANK, W_GUA_BURST, W_GUA_ENTITY = _W_GUA

# Output filenames get a suffix when a placebo control is active, so they
# never collide with the main score-selected / theory-weighted CSVs.
_suffix = ""
if _args.selection == "random":
    _suffix += "_randselect"
if _args.random_weights:
    _suffix += "_randweights"

# ===================== CONFIG =====================

# Resolve project root from this file's location so paths work on Windows / Linux / macOS.
# scripts/rat/rat_injector.py  →  parents[2] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = _args.data_dir if _args.data_dir else str(PROJECT_ROOT / "ibm_transcations_datasets")

TX_CSV_PATH       = os.path.join(BASE_DIR, _args.trans_file)
ACCOUNTS_CSV_PATH = os.path.join(BASE_DIR, _args.acct_file)
PATTERNS_TXT_PATH = os.path.join(BASE_DIR, _args.patterns_file)

OUTPUT_DIR = _args.output_dir if _args.output_dir else os.path.join(BASE_DIR, "RAT")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TS_COL      = "Timestamp"
SRC_COL     = "Account"
DST_COL     = "Account.1"
FROM_BANK   = "From Bank"
TO_BANK     = "To Bank"
AMT_PAID    = "Amount Paid"
AMT_REC     = "Amount Received"
LABEL_COL   = "Is Laundering"

ACCT_ID_COL      = "Account Number"
ACCT_ENTITY_ID   = "Entity ID"
ACCT_ENTITY_NAME = "Entity Name"

INTENSITIES = {"low": 0.05, "medium": 0.10, "high": 0.20}

# Restrict to the requested subset (default: all three). Mirrors
# slt_injector.py's --intensities flag -- useful for placebo-control runs
# that only need one intensity level to demonstrate the control.
_requested_intensities = [
    x.strip() for x in _args.intensities.replace(",", " ").split() if x.strip()
]
for _ri in _requested_intensities:
    if _ri not in INTENSITIES:
        raise ValueError(f"Unknown intensity '{_ri}'. Valid: {list(INTENSITIES.keys())}")
INTENSITIES = {k: v for k, v in INTENSITIES.items() if k in _requested_intensities}
print(f"[RAT] Producing intensities: {list(INTENSITIES.keys())}")

EPS = 1e-8

# ===================== HELPERS =====================

def safe_zscore(x, mean, std):
    return (x - mean) / (std.replace(0, np.nan) + EPS)

def norm_by_quantile(series, q=0.95):
    s = series.astype(float)
    qv = s.quantile(q)
    if not np.isfinite(qv) or qv <= 0:
        qv = s.max()
    if not np.isfinite(qv) or qv <= 0:
        return pd.Series(0.0, index=series.index)
    out = (s / qv).replace([np.inf, -np.inf], np.nan).fillna(0)
    return out.clip(0, 1)

def clip_positive(series):
    return series.clip(lower=0.0)

# ===================== LOAD TRANSACTIONS =====================

print(f"Loading transactions from: {TX_CSV_PATH}")
df = pd.read_csv(TX_CSV_PATH, low_memory=False).reset_index(drop=True)
df[TS_COL] = pd.to_datetime(df[TS_COL], errors="raise")

df[AMT_PAID] = pd.to_numeric(df[AMT_PAID], errors="coerce", downcast="float")
df[AMT_REC]  = pd.to_numeric(df[AMT_REC],  errors="coerce", downcast="float")
df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce", downcast="integer").fillna(0).astype(np.int8)

df[SRC_COL] = df[SRC_COL].astype(str)
df[DST_COL] = df[DST_COL].astype(str)


# ===================== LOAD ACCOUNTS =====================

print(f"Loading accounts from: {ACCOUNTS_CSV_PATH}")
df_acct = pd.read_csv(ACCOUNTS_CSV_PATH, low_memory=False)

# Make join key consistent BEFORE setting index
df_acct[ACCT_ID_COL] = df_acct[ACCT_ID_COL].astype(str)
df_acct = df_acct.set_index(ACCT_ID_COL)

# Optional pattern accounts
pattern_accounts = set()
if os.path.exists(PATTERNS_TXT_PATH):
    with open(PATTERNS_TXT_PATH, "r") as f:
        for line in f:
            acc = line.strip()
            if acc:
                pattern_accounts.add(acc)
print(f"Pattern accounts loaded: {len(pattern_accounts)}")

# ===================== PER-ACCOUNT STATS =====================

src_group = df.groupby(SRC_COL)
dst_group = df.groupby(DST_COL)

df["src_out_degree"] = src_group[DST_COL].nunique().reindex(df[SRC_COL]).values
df["dst_in_degree"]  = dst_group[SRC_COL].nunique().reindex(df[DST_COL]).values

df["src_amt_mean"] = src_group[AMT_PAID].mean().reindex(df[SRC_COL]).values
df["src_amt_std"]  = src_group[AMT_PAID].std().reindex(df[SRC_COL]).values
df["dst_amt_mean"] = dst_group[AMT_REC].mean().reindex(df[DST_COL]).values
df["dst_amt_std"]  = dst_group[AMT_REC].std().reindex(df[DST_COL]).values

df["src_first_seen"] = src_group[TS_COL].min().reindex(df[SRC_COL]).values
df["dst_first_seen"] = dst_group[TS_COL].min().reindex(df[DST_COL]).values

df["src_age_days"] = (df[TS_COL] - df["src_first_seen"]).dt.total_seconds() / (3600*24)
df["dst_age_days"] = (df[TS_COL] - df["dst_first_seen"]).dt.total_seconds() / (3600*24)

df["src_age_days"] = df["src_age_days"].fillna(0)
df["dst_age_days"] = df["dst_age_days"].fillna(0)

# ===================== BURSTINESS =====================

df["date_only"] = df[TS_COL].dt.date
df["src_day_tx_count"] = df.groupby([SRC_COL,"date_only"])[AMT_PAID].transform("count")
df["dst_day_tx_count"] = df.groupby([DST_COL,"date_only"])[AMT_REC].transform("count")

# ===================== TIME CONTEXT =====================

df["hour"] = df[TS_COL].dt.hour
df["weekday"] = df[TS_COL].dt.weekday

df["RAT_is_off_hours"]  = ((df["hour"] < 8) | (df["hour"] > 18)).astype(int)
df["RAT_is_weekend"]    = (df["weekday"] >= 5).astype(int)
df["RAT_is_cross_bank"] = (df[FROM_BANK] != df[TO_BANK]).astype(int)

# ===================== AMOUNT Z-SCORES =====================

df["RAT_src_amount_z_pos"] = clip_positive(safe_zscore(df[AMT_PAID], df["src_amt_mean"], df["src_amt_std"]))
df["RAT_dst_amount_z_pos"] = clip_positive(safe_zscore(df[AMT_REC],  df["dst_amt_mean"], df["dst_amt_std"]))

# ===================== NORMALIZE STRUCTURAL =====================

df["RAT_src_out_deg_norm"] = norm_by_quantile(df["src_out_degree"].fillna(0))
df["RAT_dst_in_deg_norm"]  = norm_by_quantile(df["dst_in_degree"].fillna(0))
df["RAT_src_burst_norm"]   = norm_by_quantile(df["src_day_tx_count"].fillna(0))
df["RAT_dst_burst_norm"]   = norm_by_quantile(df["dst_day_tx_count"].fillna(0))
df["RAT_combined_burst"]   = norm_by_quantile(df["src_day_tx_count"].fillna(0) +
                                              df["dst_day_tx_count"].fillna(0)

)

# ===================== MERGE ENTITY INFO =====================

df = df.join(df_acct.add_prefix("srcacct_"), on=SRC_COL)
df = df.join(df_acct.add_prefix("dstacct_"), on=DST_COL)

df["src_entity_id"] = df["srcacct_" + ACCT_ENTITY_ID]
df["dst_entity_id"] = df["dstacct_" + ACCT_ENTITY_ID]

df["RAT_same_entity"] = (df["src_entity_id"].astype(str) == df["dst_entity_id"].astype(str)).astype(int)

entity_acct_count = df_acct.reset_index().groupby(ACCT_ENTITY_ID)[ACCT_ID_COL].nunique()
df["RAT_src_entity_accounts"] = df["src_entity_id"].map(entity_acct_count).fillna(1)
df["RAT_dst_entity_accounts"] = df["dst_entity_id"].map(entity_acct_count).fillna(1)

df["RAT_src_entity_acct_norm"] = norm_by_quantile(df["RAT_src_entity_accounts"])
df["RAT_dst_entity_acct_norm"] = norm_by_quantile(df["RAT_dst_entity_accounts"])

# ===================== PATTERN FLAGS (REMOVED) =====================
# RAT_src_pattern_flag / RAT_dst_pattern_flag used to be derived here from
# `pattern_accounts` (loaded from the AMLworld simulator's ground-truth
# laundering-pattern export). That's information a real investigator would
# not have before prediction, so it has been removed as a model input --
# see review feedback Aug 4 2026, sec 3.1. `pattern_accounts` is still
# loaded above but is no longer used to construct any feature.

# ===================== MUTUAL FLOW =====================

edge_counts = df.groupby([SRC_COL, DST_COL]).size().reset_index(name="count")
rev = edge_counts.rename(columns={SRC_COL:"DST_tmp", DST_COL:"SRC_tmp"})

mutual = edge_counts.merge(
    rev,
    left_on=[SRC_COL, DST_COL],
    right_on=["DST_tmp", "SRC_tmp"],
    how="inner"
)[[SRC_COL, DST_COL]].drop_duplicates()

mutual["RAT_mutual_flag"] = 1
df = df.merge(mutual, on=[SRC_COL, DST_COL], how="left")
df["RAT_mutual_flag"] = df["RAT_mutual_flag"].fillna(0)

# ===================== MOTIF FEATURES =====================

src_outdeg_by_acct = df.groupby(SRC_COL)[DST_COL].nunique()
df["dst_out_degree"] = df[DST_COL].map(src_outdeg_by_acct)

df["dst_out_deg_norm"]  = norm_by_quantile(df["dst_out_degree"].fillna(0))

df["motif_fanin"]   = df["RAT_dst_in_deg_norm"]
df["motif_fanout"]  = df["RAT_src_out_deg_norm"]
df["motif_chain"]   = np.sqrt(df["RAT_dst_in_deg_norm"] * df["dst_out_deg_norm"])

df["motif_cycle"] = (
    0.5 * df["RAT_mutual_flag"] +
    0.3 * df["RAT_same_entity"] +
    0.2 * df["RAT_combined_burst"]
)

# ===================== RAT SUB-SCORES =====================

df["RAT_offender_score"] = (
    W_OFF_AMT*df["RAT_src_amount_z_pos"] +
    W_OFF_OUTDEG*df["RAT_src_out_deg_norm"] +
    W_OFF_BURST*df["RAT_src_burst_norm"] +
    W_OFF_OFFHOURS*df["RAT_is_off_hours"] +
    W_OFF_ENTITY*df["RAT_src_entity_acct_norm"]
)

df["RAT_target_score"] = (
    W_TAR_AMT*df["RAT_dst_amount_z_pos"] +
    W_TAR_INDEG*df["RAT_dst_in_deg_norm"] +
    W_TAR_AGE*(1 - norm_by_quantile(df["dst_age_days"].fillna(0))) +
    W_TAR_ENTITY*df["RAT_dst_entity_acct_norm"]
)

df["RAT_guardian_weakness_score"] = (
    W_GUA_OFFHOURS*df["RAT_is_off_hours"] +
    W_GUA_WEEKEND*df["RAT_is_weekend"] +
    W_GUA_CROSSBANK*df["RAT_is_cross_bank"] +
    W_GUA_BURST*df["RAT_combined_burst"] +
    W_GUA_ENTITY*df["RAT_same_entity"]
)

# ===================== MULTIPLICATIVE RAT SCORE =====================

print("Computing multiplicative RAT score...")

df["RAT_score"] = (
    (df["RAT_offender_score"] + EPS) *
    (df["RAT_target_score"] + EPS) *
    (df["RAT_guardian_weakness_score"] + EPS)
) ** (1/3)

df["RAT_score"] = df["RAT_score"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0,1)

# ===================== CREATE INTENSITY DATASETS =====================

launder_mask = df[LABEL_COL] == 1

# IMPORTANT FIX: DROP NaN RAT SCORES
launder_scores = (
    df.loc[launder_mask, "RAT_score"]
      .replace([np.inf, -np.inf], np.nan)
      .dropna()
      .astype(float)
      .values
)

print(f"Valid laundering RAT_scores: {len(launder_scores)}")

if len(launder_scores) < 10:
    raise RuntimeError("Too few valid laundering RAT scores — check calculations.")

float_cols = [c for c in df.columns if c.startswith(("RAT_", "motif_")) or c.endswith("_norm")]
df[float_cols] = df[float_cols].astype(np.float32)

# ===================== INJECTION RULES (FIX) =====================
# Previously the intensity loop only set the RAT_injected metadata flag, so the
# low/medium/high CSVs had IDENTICAL feature values (the flag is excluded from
# model inputs downstream). This fix makes injection real: for the selected
# laundering rows, continuous RAT/motif feature values are boosted toward their
# global 95th percentile, and the composite scores are recomputed, so each
# intensity level differs in feature space.

BOOST_ALPHA = 0.7  # blend strength toward the 95th-percentile value
BOOST_COLS = [
    "RAT_src_amount_z_pos", "RAT_dst_amount_z_pos",
    "RAT_src_out_deg_norm", "RAT_dst_in_deg_norm",
    "RAT_src_burst_norm", "RAT_dst_burst_norm", "RAT_combined_burst",
    "RAT_src_entity_acct_norm", "RAT_dst_entity_acct_norm",
    "motif_fanin", "motif_fanout", "motif_chain", "motif_cycle",
]
SCORE_COLS = ["RAT_offender_score", "RAT_target_score",
              "RAT_guardian_weakness_score", "RAT_score"]

# pristine copies so boosts never accumulate across intensity iterations
_originals = {c: df[c].copy() for c in BOOST_COLS + SCORE_COLS}
_q95 = {c: float(np.nanquantile(df[c].values.astype(float), 0.95)) for c in BOOST_COLS}
_dst_age_norm = norm_by_quantile(df["dst_age_days"].fillna(0))

# df is still fully pristine here -- nothing in the intensity loop below has
# touched it yet. Dumping now (before any boosting) gives a CSV with the
# exact same rows/order as every HI-Small_Trans_RAT_<intensity>.csv, but
# with RAT_*/motif_* features at their never-boosted values.
if _args.dump_pristine:
    out_base = os.path.splitext(os.path.basename(_args.trans_file))[0]
    pristine_path = os.path.join(OUTPUT_DIR, f"{out_base}_RAT_pristine{_suffix}.csv")
    print(f"Saving pristine (un-boosted) snapshot: {pristine_path}")
    df.to_csv(pristine_path, index=False)
    print(f"Saved {pristine_path} [0 injected rows by construction]")

for name, frac in INTENSITIES.items():
    # restore pristine feature values before applying this intensity's boost
    for c in BOOST_COLS + SCORE_COLS:
        df[c] = _originals[c].copy()

    if _args.selection == "score":
        threshold = float(np.quantile(launder_scores, 1 - frac))
        print(f"{name}: threshold = {threshold:.4f}")

        # selection uses the pristine RAT_score (as before)
        df["RAT_injected"] = (
            (df[LABEL_COL] == 1) &
            (df["RAT_score"] >= threshold)
        ).astype(np.int8)
    else:
        # Placebo control: select a random sample of laundering rows of the
        # same size the score-based threshold would have selected, instead of
        # ranking by RAT_score. Boost mechanics afterward are identical.
        _sel_rng = np.random.default_rng(
            _args.selection_seed + {"low": 0, "medium": 1, "high": 2}[name]
        )
        n_select = int(round(frac * int(launder_mask.sum())))
        launder_idx = df.index[launder_mask]
        chosen_idx = _sel_rng.choice(launder_idx, size=min(n_select, len(launder_idx)), replace=False)
        print(f"{name}: random selection = {n_select} of {len(launder_idx)} laundering rows")

        df["RAT_injected"] = np.int8(0)
        df.loc[chosen_idx, "RAT_injected"] = 1

    inj = df["RAT_injected"] == 1

    # boost continuous RAT features for injected rows (upward only)
    for c in BOOST_COLS:
        cur = df.loc[inj, c].astype(float)
        df.loc[inj, c] = np.maximum(cur, cur + BOOST_ALPHA * (_q95[c] - cur)).astype(np.float32)

    # recompute composite scores from the boosted features (same formulas as above)
    df["RAT_offender_score"] = (
        W_OFF_AMT*df["RAT_src_amount_z_pos"] +
        W_OFF_OUTDEG*df["RAT_src_out_deg_norm"] +
        W_OFF_BURST*df["RAT_src_burst_norm"] +
        W_OFF_OFFHOURS*df["RAT_is_off_hours"] +
        W_OFF_ENTITY*df["RAT_src_entity_acct_norm"]
    ).astype(np.float32)

    df["RAT_target_score"] = (
        W_TAR_AMT*df["RAT_dst_amount_z_pos"] +
        W_TAR_INDEG*df["RAT_dst_in_deg_norm"] +
        W_TAR_AGE*(1 - _dst_age_norm) +
        W_TAR_ENTITY*df["RAT_dst_entity_acct_norm"]
    ).astype(np.float32)

    df["RAT_guardian_weakness_score"] = (
        W_GUA_OFFHOURS*df["RAT_is_off_hours"] +
        W_GUA_WEEKEND*df["RAT_is_weekend"] +
        W_GUA_CROSSBANK*df["RAT_is_cross_bank"] +
        W_GUA_BURST*df["RAT_combined_burst"] +
        W_GUA_ENTITY*df["RAT_same_entity"]
    ).astype(np.float32)

    df["RAT_score"] = ((
        (df["RAT_offender_score"] + EPS) *
        (df["RAT_target_score"] + EPS) *
        (df["RAT_guardian_weakness_score"] + EPS)
    ) ** (1/3)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1).astype(np.float32)

    df["RAT_intensity_level"] = (df["RAT_injected"] * {"low": 1, "medium": 2, "high": 3}[name]).astype(np.int8)

    out_base = os.path.splitext(os.path.basename(_args.trans_file))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{out_base}_RAT_{name}{_suffix}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} [{int(df['RAT_injected'].sum())} injected rows]")


print("DONE.")