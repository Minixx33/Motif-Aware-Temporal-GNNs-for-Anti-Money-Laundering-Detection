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

# Sort chronologically BEFORE any feature engineering. Everything below this
# point (degree, amount mean/std, burstiness, mutual-flow) is computed with
# cumulative/expanding operations that only make sense -- and are only
# causal -- if rows are in time order first. This also matches
# motif_graph_builder_static.py's own sort, so the graph builder's re-sort
# is idempotent on this output.
df = df.sort_values(TS_COL, kind="mergesort").reset_index(drop=True)


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

# ===================== PER-ACCOUNT STATS (POINT-IN-TIME / CAUSAL) =====================
# Everything in this section used to be computed with a plain groupby over the
# WHOLE dataset (e.g. src_group[DST_COL].nunique()), which means a
# transaction's "out-degree"/"average amount"/etc. feature reflected that
# account's ENTIRE history -- including transactions that happen after it,
# possibly in the val/test period. That is leakage: a real investigator
# scoring a transaction at time t cannot see what the account does at t+1.
#
# Below, every one of these features is rebuilt to be strictly a function of
# transactions STRICTLY BEFORE the current row (row order = chronological,
# guaranteed by the sort right after loading). Two features -- first_seen and
# the age derived from it -- were already causal even under the old global
# groupby (an account's first-ever transaction time doesn't change depending
# on what "future" data you can see), so they're left as-is.
#
# Known remaining, disclosed limitation: the *normalization scale* used below
# (norm_by_quantile's global P95, and the boost target's global P95 further
# down) is still computed from the full dataset. That is a much weaker form
# of dependence than the raw feature values themselves (it's a fixed rescaling
# constant, not a per-row leak of that account's own future activity), but it
# is not fully point-in-time. A fully rigorous fix would use an expanding
# quantile instead; not implemented here due to the added complexity/runtime
# cost of a per-row expanding quantile over a multi-million-row file.

is_new_pair = ~df.duplicated(subset=[SRC_COL, DST_COL], keep="first")
_is_new_pair_int = is_new_pair.astype(np.int32)

# Running (causal) out-degree / in-degree: count of DISTINCT counterparties
# this account has transacted with STRICTLY BEFORE this row. A repeat
# transaction with an already-seen counterparty does not increase degree.
_src_new_cumsum = _is_new_pair_int.groupby(df[SRC_COL]).cumsum()
_dst_new_cumsum = _is_new_pair_int.groupby(df[DST_COL]).cumsum()
df["src_out_degree"] = (_src_new_cumsum - _is_new_pair_int).astype(np.int32)
df["dst_in_degree"]  = (_dst_new_cumsum - _is_new_pair_int).astype(np.int32)

# Running (causal) amount mean/std: computed from this account's PRIOR
# transactions only (population formula via cumulative sum / sum-of-squares,
# excluding the current row). First transaction for an account has no prior
# history -> NaN mean/std -> z-score fillna(0) below (no baseline yet, so
# "not anomalous" is the only defensible default).
_amt_paid_sq = df[AMT_PAID].astype(np.float64) ** 2
_amt_rec_sq  = df[AMT_REC].astype(np.float64) ** 2

# NOTE: computed in float64 throughout (not the float32 AMT_PAID/AMT_REC
# columns) -- the E[X^2] - E[X]^2 variance formula is numerically unstable
# under catastrophic cancellation, and float32 precision was leaving a
# spurious nonzero residual even in cases where the true variance is exactly
# zero (e.g. an account's 2nd-ever transaction, with only 1 prior value).
_amt_paid_f64 = df[AMT_PAID].astype(np.float64)
_amt_rec_f64  = df[AMT_REC].astype(np.float64)

_src_cumsum   = _amt_paid_f64.groupby(df[SRC_COL]).cumsum()
_src_cumsumsq = _amt_paid_sq.groupby(df[SRC_COL]).cumsum()
_src_cumcount = df.groupby(SRC_COL).cumcount()  # # of PRIOR rows in group (excludes current)

_dst_cumsum   = _amt_rec_f64.groupby(df[DST_COL]).cumsum()
_dst_cumsumsq = _amt_rec_sq.groupby(df[DST_COL]).cumsum()
_dst_cumcount = df.groupby(DST_COL).cumcount()

_src_prior_sum   = _src_cumsum - _amt_paid_f64
_src_prior_sumsq = _src_cumsumsq - _amt_paid_sq
_src_prior_n     = _src_cumcount.replace(0, np.nan)

_dst_prior_sum   = _dst_cumsum - _amt_rec_f64
_dst_prior_sumsq = _dst_cumsumsq - _amt_rec_sq
_dst_prior_n     = _dst_cumcount.replace(0, np.nan)

df["src_amt_mean"] = _src_prior_sum / _src_prior_n
_src_amt_var = (_src_prior_sumsq / _src_prior_n) - df["src_amt_mean"] ** 2
df["src_amt_std"] = np.sqrt(_src_amt_var.clip(lower=0))

df["dst_amt_mean"] = _dst_prior_sum / _dst_prior_n
_dst_amt_var = (_dst_prior_sumsq / _dst_prior_n) - df["dst_amt_mean"] ** 2
df["dst_amt_std"] = np.sqrt(_dst_amt_var.clip(lower=0))

# first_seen / age_days: already causal (an account's first-ever transaction
# time is the same value whether or not you can see the future), no change.
src_group = df.groupby(SRC_COL)
dst_group = df.groupby(DST_COL)
df["src_first_seen"] = src_group[TS_COL].transform("min")
df["dst_first_seen"] = dst_group[TS_COL].transform("min")

df["src_age_days"] = (df[TS_COL] - df["src_first_seen"]).dt.total_seconds() / (3600*24)
df["dst_age_days"] = (df[TS_COL] - df["dst_first_seen"]).dt.total_seconds() / (3600*24)

df["src_age_days"] = df["src_age_days"].fillna(0)
df["dst_age_days"] = df["dst_age_days"].fillna(0)

# ===================== BURSTINESS (POINT-IN-TIME / CAUSAL) =====================
# "How many transactions has this account made TODAY, up to and including
# this one" -- a running same-day count instead of the old
# groupby(...).transform("count"), which counted the WHOLE day's activity
# (including transactions that happen later that same day).

df["date_only"] = df[TS_COL].dt.date
df["src_day_tx_count"] = (df.groupby([SRC_COL, "date_only"]).cumcount() + 1).astype(np.int32)
df["dst_day_tx_count"] = (df.groupby([DST_COL, "date_only"]).cumcount() + 1).astype(np.int32)

# ===================== TIME CONTEXT =====================

df["hour"] = df[TS_COL].dt.hour
df["weekday"] = df[TS_COL].dt.weekday

df["RAT_is_off_hours"]  = ((df["hour"] < 8) | (df["hour"] > 18)).astype(int)
df["RAT_is_weekend"]    = (df["weekday"] >= 5).astype(int)
df["RAT_is_cross_bank"] = (df[FROM_BANK] != df[TO_BANK]).astype(int)

# ===================== AMOUNT Z-SCORES =====================

df["RAT_src_amount_z_pos"] = clip_positive(safe_zscore(df[AMT_PAID], df["src_amt_mean"], df["src_amt_std"])).fillna(0)
df["RAT_dst_amount_z_pos"] = clip_positive(safe_zscore(df[AMT_REC],  df["dst_amt_mean"], df["dst_amt_std"])).fillna(0)

# ===================== NORMALIZE STRUCTURAL =====================
# (see the disclosed-limitation note above: the P95 scale itself is global)

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

# ===================== MUTUAL FLOW (POINT-IN-TIME / CAUSAL) =====================
# Old version: flag (src,dst) if the REVERSE (dst,src) pair occurs ANYWHERE
# in the dataset -- including after this transaction. That lets a
# transaction's feature depend on a reversal that hasn't happened yet at
# prediction time. Causal version: flag only if dst had ALREADY sent to src
# at some point STRICTLY BEFORE this transaction's timestamp. This requires
# a genuinely sequential scan (an "as of this instant, have I seen the
# reverse edge yet" query), so it's an explicit single pass rather than a
# vectorized groupby -- same pattern already used for log_time_since_src/dst
# in motif_graph_builder_static.py at this dataset's scale.
_seen_pairs = set()
_mutual_flag = np.zeros(len(df), dtype=np.int8)
_src_vals = df[SRC_COL].to_numpy()
_dst_vals = df[DST_COL].to_numpy()
for _i in range(len(df)):
    _s, _d = _src_vals[_i], _dst_vals[_i]
    if (_d, _s) in _seen_pairs:
        _mutual_flag[_i] = 1
    _seen_pairs.add((_s, _d))
df["RAT_mutual_flag"] = _mutual_flag
del _seen_pairs

# ===================== MOTIF FEATURES (POINT-IN-TIME / CAUSAL) =====================
# dst_out_degree = "as if the DST account were a source, how many distinct
# counterparties has IT sent money to, as of just before this transaction."
# That history lives on OTHER rows (the ones where this account appears as
# SRC_COL), so a simple positional cumsum doesn't reach it -- use merge_asof
# to look up, for each row's dst account, the most recent (strictly prior)
# causal out-degree value computed for that account back when it acted as a
# source.
#
# Match on row POSITION (_seq), not the raw Timestamp value: every other
# causal feature above (src_out_degree, dst_in_degree, day counts,
# mutual_flag) treats "before" as "earlier row position", using row order as
# a deterministic tiebreak when timestamps are equal (this dataset has many
# exact-timestamp ties). merge_asof on the raw Timestamp instead would treat
# tied rows as simultaneous and exclude them from matching -- a different,
# inconsistent definition of "before" from the rest of this file. Row
# position has no ties by construction, so this keeps the convention uniform.
_seq = np.arange(len(df))
_src_role_lookup = pd.DataFrame({
    "acct": df[SRC_COL].to_numpy(),
    "_seq": _seq,
    "out_degree_after": _src_new_cumsum.to_numpy(),  # includes this row's own new-pair contribution
})
_dst_lookup_keys = pd.DataFrame({
    "acct": df[DST_COL].to_numpy(),
    "_seq": _seq,
})
_dst_asof = pd.merge_asof(
    _dst_lookup_keys, _src_role_lookup,
    on="_seq", by="acct",
    direction="backward", allow_exact_matches=False,  # strictly earlier row position
)
df["dst_out_degree"] = _dst_asof["out_degree_after"].fillna(0).to_numpy().astype(np.int32)
del _src_role_lookup, _dst_lookup_keys, _dst_asof, _seq

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