"""
Optimized SLT + Peer Exposure Feature Injection for IBM HI-Small / 5M transactions
----------------------------------------------------------------------------------

This version keeps the same SLT idea, but is rewritten to be much safer for large
transaction files (e.g., ~5M rows) on a laptop.

Core SLT logic:
1. Build an UNSUPERVISED peer-risk score per account (no labels used here).
2. Use a percentile threshold tau (e.g., P95) to mark "high-risk peers".
3. Compute account-day exposure to high-risk peers:
      - suspicious neighbor ratio
      - suspicious amount share
      - suspicious transaction share
      - suspicious strong-tie ratio
4. Add temporal / lagged exposure:
      - lag1
      - lag2
      - delta
      - cumulative recent exposure
5. Merge those features back into the transaction table.
6. Use labels ONLY at the end to create low/medium/high SLT-injected datasets.

Compared with the earlier prototype:
- Reads only required columns
- Downcasts datatypes
- Avoids duplicating the full transaction table into a 10M-row exposure table
- Computes source-side and destination-side exposure summaries separately
"""

import argparse
import os
import gc
from pathlib import Path

import numpy as np
import pandas as pd

# ===================== CLI ARGS =====================

_parser = argparse.ArgumentParser(description="SLT feature injector with configurable score weights")
_parser.add_argument("--variant",      type=str,   default="current",
                     help="Ablation variant name (used in output filenames)")
_parser.add_argument("--w_neighbor",   type=float, default=0.30, help="Weight: suspicious neighbour ratio")
_parser.add_argument("--w_amount",     type=float, default=0.25, help="Weight: suspicious amount share")
_parser.add_argument("--w_strong_tie", type=float, default=0.20, help="Weight: strong-tie suspicious ratio")
_parser.add_argument("--w_delta",      type=float, default=0.15, help="Weight: exposure delta")
_parser.add_argument("--w_cum",        type=float, default=0.10, help="Weight: cumulative 7-day exposure")
_parser.add_argument("--data_dir",     type=str,   default=None,
                     help="Path to ibm_transcations_datasets/ (default: auto-resolved from script location)")
_parser.add_argument("--output_dir",   type=str,   default=None,
                     help="Output directory for injected CSVs (default: <data_dir>/SLT[/<variant>])")
_parser.add_argument("--intensities",  type=str,   default="low,medium,high",
                     help="Comma/space separated subset of intensities to produce "
                          "(default: all three, e.g. --intensities medium)")
_parser.add_argument("--dump_pristine", action="store_true",
                     help="Also write a snapshot of the data BEFORE any intensity "
                          "boosting is applied (SLT_* features at their pristine, "
                          "never-boosted values). Same row count/order as the "
                          "low/medium/high CSVs -- used for post-hoc leakage/"
                          "robustness checks, not part of normal training.")

# ---- Falsification / placebo controls (review feedback sec 4.3) ----
_parser.add_argument("--random_weights", action="store_true",
                     help="Placebo control: ignore --w_neighbor/--w_amount/"
                          "--w_strong_tie/--w_delta/--w_cum and draw random "
                          "positive weights (Dirichlet, summing to 1) instead "
                          "of the theory-motivated weights for the SLT_src_score/"
                          "SLT_dst_score formulas. Reproducible via --weight_seed.")
_parser.add_argument("--weight_seed", type=int, default=42)
_parser.add_argument("--selection", choices=["score", "random"], default="score",
                     help="Placebo control: 'score' (default) selects the top "
                          "laundering transactions by SLT_score, as before. "
                          "'random' selects a random sample of the same size "
                          "instead (same boost mechanics applied afterward).")
_parser.add_argument("--selection_seed", type=int, default=123)

_args = _parser.parse_args()

VARIANT      = _args.variant

if _args.random_weights:
    _wrng = np.random.default_rng(_args.weight_seed)
    W_NEIGHBOR, W_AMOUNT, W_STRONG_TIE, W_DELTA, W_CUM = _wrng.dirichlet(np.ones(5))
    print(f"[SLT] --random_weights set (seed={_args.weight_seed}): "
          f"({W_NEIGHBOR}, {W_AMOUNT}, {W_STRONG_TIE}, {W_DELTA}, {W_CUM})")
else:
    W_NEIGHBOR   = _args.w_neighbor
    W_AMOUNT     = _args.w_amount
    W_STRONG_TIE = _args.w_strong_tie
    W_DELTA      = _args.w_delta
    W_CUM        = _args.w_cum

# Output filenames get a suffix when a placebo control is active, so they
# never collide with the main score-selected / theory-weighted CSVs.
_suffix = ""
if _args.selection == "random":
    _suffix += "_randselect"
if _args.random_weights:
    _suffix += "_randweights"

print(f"[SLT] Variant: {VARIANT}  weights=({W_NEIGHBOR}, {W_AMOUNT}, {W_STRONG_TIE}, {W_DELTA}, {W_CUM})")

# ===================== CONFIG =====================

# Resolve project root from this file's location so paths work on Windows / Linux / macOS.
# scripts/SLT/slt_injector.py  →  parents[2] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = _args.data_dir if _args.data_dir else str(PROJECT_ROOT / "ibm_transcations_datasets")

TX_CSV_PATH       = os.path.join(BASE_DIR, "HI-Small_Trans.csv")
ACCOUNTS_CSV_PATH = os.path.join(BASE_DIR, "HI-Small_accounts.csv")
PATTERNS_TXT_PATH = os.path.join(BASE_DIR, "HI-Small_Patterns.txt")   # optional

# "current" uses the flat SLT/ folder with the default naming convention.
# All other variants get their own subfolder.
if _args.output_dir:
    OUTPUT_DIR = _args.output_dir
elif VARIANT == "current":
    OUTPUT_DIR = os.path.join(BASE_DIR, "SLT")
else:
    OUTPUT_DIR = os.path.join(BASE_DIR, "SLT", VARIANT)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column names in the transaction file
TS_COL      = "Timestamp"
SRC_COL     = "Account"
DST_COL     = "Account.1"
FROM_BANK   = "From Bank"
TO_BANK     = "To Bank"
AMT_PAID    = "Amount Paid"
AMT_REC     = "Amount Received"
LABEL_COL   = "Is Laundering"

# Column names in the accounts file
ACCT_ID_COL      = "Account Number"
ACCT_ENTITY_ID   = "Entity ID"
ACCT_ENTITY_NAME = "Entity Name"

# How much of the laundering rows we mark as injected
INTENSITIES = {
    "low": 0.05,
    "medium": 0.10,
    "high": 0.20,
}

# Restrict to the requested subset (default: all three). Useful when only one
# intensity is actually needed downstream -- e.g. ablation variants that only
# ever train at medium -- so the injector doesn't waste a full pass over the
# dataset producing CSVs nothing will read.
_requested_intensities = [
    x.strip() for x in _args.intensities.replace(",", " ").split() if x.strip()
]
for _ri in _requested_intensities:
    if _ri not in INTENSITIES:
        raise ValueError(f"Unknown intensity '{_ri}'. Valid: {list(INTENSITIES.keys())}")
INTENSITIES = {k: v for k, v in INTENSITIES.items() if k in _requested_intensities}
print(f"[SLT] Producing intensities: {list(INTENSITIES.keys())}")

# Top 5% by unsupervised peer-risk score are considered "high-risk peers"
PEER_RISK_PERCENTILE = 0.95

EPS = 1e-8


# ===================== HELPERS =====================

def safe_divide(a, b):
    """Safe division that returns 0 where denominator is 0."""
    a = pd.to_numeric(a, errors="coerce").fillna(0).astype(np.float32)
    b = pd.to_numeric(b, errors="coerce").fillna(0).astype(np.float32)
    return np.where(np.abs(b) < EPS, 0.0, a / (b + EPS)).astype(np.float32)


def robust_norm(series, q=0.95, fit_series=None):
    """
    Normalize by a robust upper quantile instead of max.
    This keeps huge outliers from dominating the scale.
    Output is clipped to [0,1].

    If fit_series is given, the quantile (the scale CONSTANT) is fit from
    fit_series instead of the full `series` -- used to fit on a train-period
    subset only while still transforming every row, so the constant doesn't
    depend on val/test-period values. See NORM_TRAIN_FRAC / _NORM_FIT_N.
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(np.float32)
    if fit_series is not None:
        fit_s = pd.to_numeric(fit_series, errors="coerce").fillna(0).astype(np.float32)
    else:
        fit_s = s
    qv = float(fit_s.quantile(q))
    if (not np.isfinite(qv)) or (qv <= 0):
        qv = float(fit_s.max())
    if (not np.isfinite(qv)) or (qv <= 0):
        return pd.Series(0.0, index=series.index, dtype=np.float32)
    out = (s / qv).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)
    return out.astype(np.float32)


def load_pattern_accounts(path):
    """Load pattern accounts from the patterns txt file, if present."""
    pattern_accounts = set()
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                acc = line.strip()
                if acc:
                    pattern_accounts.add(acc)
    return pattern_accounts


def print_mem_hint(df, name):
    """Small helper to print approximate dataframe memory usage."""
    try:
        mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        print(f"[MEM] {name}: ~{mb:.2f} MB")
    except Exception:
        pass


# ===================== LOAD MINIMAL DATA =====================

print(f"Loading transactions from: {TX_CSV_PATH}")

RCURR = "Receiving Currency"
PCURR = "Payment Currency"
PFORMAT = "Payment Format"

tx_usecols = [
    TS_COL, SRC_COL, DST_COL, FROM_BANK, TO_BANK,
    AMT_PAID, AMT_REC, LABEL_COL,
    RCURR, PCURR, PFORMAT
]

df = pd.read_csv(
    TX_CSV_PATH,
    usecols=tx_usecols,
    low_memory=False
).reset_index(drop=True)

print_mem_hint(df, "raw transactions")

print(f"Loading accounts from: {ACCOUNTS_CSV_PATH}")
df_acct = pd.read_csv(ACCOUNTS_CSV_PATH, low_memory=False)

# Optional pattern accounts
pattern_accounts = load_pattern_accounts(PATTERNS_TXT_PATH)
print(f"Pattern accounts loaded: {len(pattern_accounts)}")

# ===================== CLEAN / DOWNCAST TRANSACTION DATA =====================

print("Cleaning and downcasting transaction data...")

# Parse timestamp once
df[TS_COL] = pd.to_datetime(df[TS_COL], errors="raise")

# Sort chronologically BEFORE any feature engineering. The causal/running
# account-behavior features below (txn count, amount flow, unique
# counterparties, peer-risk score -- all computed "as of just before this
# transaction") only make sense, and are only causal, in time order.
df = df.sort_values(TS_COL, kind="mergesort").reset_index(drop=True)

# Used to fit normalization scale constants (robust_norm's P95, the
# peer-risk tau threshold) on a train-period-only window, so they don't
# reflect val/test-period distributional information. The per-row values
# feeding into these are already point-in-time causal; this just keeps the
# rescaling CONSTANTS point-in-time too, matching rat_injector.py's fix.
# Independent of how splits are eventually assigned (stratified-random vs.
# chronological) -- a conservative causal cutoff, not a claim about the real
# train/val/test boundary.
NORM_TRAIN_FRAC = 0.60  # matches create_splits.py's default train_ratio
_NORM_FIT_N = int(len(df) * NORM_TRAIN_FRAC)
# Same cutoff expressed as a timestamp, for the day_exposure table further
# down (grouped by account+date, not globally date-sorted, so a row-index
# cutoff doesn't apply there -- a date cutoff does).
_TRAIN_CUTOFF_TS = df[TS_COL].iloc[min(_NORM_FIT_N, len(df) - 1)].floor("D")

# Convert account IDs to string first, then to category to save memory
df[SRC_COL] = df[SRC_COL].astype(str)
df[DST_COL] = df[DST_COL].astype(str)

df[RCURR] = df[RCURR].astype(str).astype("category")
df[PCURR] = df[PCURR].astype(str).astype("category")
df[PFORMAT] = df[PFORMAT].astype(str).astype("category")

# Compare bank codes while still plain strings, then convert to category for RAM savings.
df[FROM_BANK] = df[FROM_BANK].astype(str)
df[TO_BANK] = df[TO_BANK].astype(str)
df["is_cross_bank"] = (df[FROM_BANK] != df[TO_BANK]).astype(np.int8)

# Banks as category to reduce RAM after cross-bank flag is built
df[FROM_BANK] = df[FROM_BANK].astype("category")
df[TO_BANK] = df[TO_BANK].astype("category")

# Amounts and label downcast
df[AMT_PAID] = pd.to_numeric(df[AMT_PAID], errors="coerce").fillna(0).astype(np.float32)
df[AMT_REC]  = pd.to_numeric(df[AMT_REC], errors="coerce").fillna(0).astype(np.float32)
df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(np.int8)

# Add time context
df["date_only"] = df[TS_COL].dt.floor("D")
df["hour"] = df[TS_COL].dt.hour.astype(np.int8)
df["weekday"] = df[TS_COL].dt.weekday.astype(np.int8)

# Base context flags
df["is_off_hours"]  = ((df["hour"] < 8) | (df["hour"] > 18)).astype(np.int8)
df["is_weekend"]    = (df["weekday"] >= 5).astype(np.int8)

# Converting account columns to category after string coercion can save a lot of RAM
df[SRC_COL] = df[SRC_COL].astype("category")
df[DST_COL] = df[DST_COL].astype("category")

print_mem_hint(df, "cleaned transactions")

# ===================== CLEAN ACCOUNTS DATA =====================

print("Cleaning accounts table...")

df_acct[ACCT_ID_COL] = df_acct[ACCT_ID_COL].astype(str)

# The real accounts file can have duplicate ACCT_ID_COL values (this ID
# isn't necessarily globally unique across banks in this dataset). The
# entity merge further below (df.merge(src_acct_meta, on="src_account_key",
# how="left")) silently expands df's row count if the right side has
# duplicate keys -- every step before that merge assumes len(df) is fixed
# at the transaction count (the causal per-account arrays computed earlier
# are indexed 0..len(df)-1), so a row-count change there would corrupt
# everything and desync the pristine CSV from the intensity-boosted CSVs'
# row order. Deduplicate up front so the merge can only ever be
# many-to-one, never many-to-many.
_n_acct_before = len(df_acct)
df_acct = df_acct.drop_duplicates(subset=[ACCT_ID_COL], keep="first")
if len(df_acct) < _n_acct_before:
    print(f"[accounts] Dropped {_n_acct_before - len(df_acct)} duplicate "
          f"'{ACCT_ID_COL}' rows (keeping first) so the entity merge below "
          f"can't change the transaction row count.")

df_acct = df_acct.set_index(ACCT_ID_COL)

# Keep account metadata lightweight
# We only really need entity info if available.
if ACCT_ENTITY_ID in df_acct.columns:
    df_acct[ACCT_ENTITY_ID] = df_acct[ACCT_ENTITY_ID].astype(str)

if ACCT_ENTITY_NAME in df_acct.columns:
    df_acct[ACCT_ENTITY_NAME] = df_acct[ACCT_ENTITY_NAME].astype(str)

print_mem_hint(df_acct, "accounts table")

# ===================== ACCOUNT-LEVEL BEHAVIORAL RISK FEATURES (POINT-IN-TIME / CAUSAL) =====================
# Old version: one peer_risk_score PER ACCOUNT, computed from that account's
# ENTIRE transaction history (both directions, summed over the whole
# dataset), then mapped onto every one of that account's transactions --
# meaning its very FIRST transaction got the same "risk" value as its
# 1000th. That's leakage: an account's future total activity level shouldn't
# be visible when scoring an earlier transaction.
#
# Causal version: peer_risk_score becomes a PER-TRANSACTION quantity --
# "this account's behavioral risk, using only activity STRICTLY BEFORE this
# transaction." An account's behavior is tracked across BOTH roles it plays
# (sender and receiver) combined, via a single running per-account state.
#
# Known remaining, disclosed limitation (same as rat_injector.py): the
# normalization scale (robust_norm's global P95) and the tau threshold below
# are calibrated from the full dataset's distribution of causal scores, not
# an expanding/point-in-time one. That's a fixed rescaling constant, not a
# per-row leak of any specific account's own future activity, but it is not
# fully point-in-time either. Not implemented here due to added complexity.

print("Computing account-level behavioral statistics (point-in-time / causal)...")

_n = len(df)
_orig_idx = np.arange(_n)

# Every transaction contributes two "touches": one for its SRC account
# (outgoing) and one for its DST account (incoming). Combining both into one
# table lets an account's running stats accumulate correctly regardless of
# which role it plays on any given transaction.
_touches = pd.concat([
    pd.DataFrame({
        "account": df[SRC_COL].to_numpy(dtype=str),
        "counterparty": df[DST_COL].to_numpy(dtype=str),
        "amount": df[AMT_PAID].to_numpy(dtype=np.float64),
        "is_cross_bank": df["is_cross_bank"].to_numpy(),
        "orig_idx": _orig_idx,
    }),
    pd.DataFrame({
        "account": df[DST_COL].to_numpy(dtype=str),
        "counterparty": df[SRC_COL].to_numpy(dtype=str),
        "amount": df[AMT_REC].to_numpy(dtype=np.float64),
        "is_cross_bank": df["is_cross_bank"].to_numpy(),
        "orig_idx": _orig_idx,
    }),
], ignore_index=True).sort_values("orig_idx", kind="mergesort").reset_index(drop=True)

_is_new_cpty = (~_touches.duplicated(subset=["account", "counterparty"], keep="first")).astype(np.int32)

# Running per-account state "after" each touch (includes that touch's own
# contribution) -- the merge_asof lookup below, with allow_exact_matches=
# False, is what turns this into a strictly-prior ("before") value per
# original transaction row.
_touches["cum_txn_count"]   = _touches.groupby("account").cumcount() + 1
_touches["cum_amount"]      = _touches.groupby("account")["amount"].cumsum()
_touches["cum_unique_cpty"] = _is_new_cpty.groupby(_touches["account"]).cumsum()
_touches["cum_cross_bank"]  = _touches.groupby("account")["is_cross_bank"].cumsum()

_touch_state_cols = ["account", "orig_idx", "cum_txn_count", "cum_amount", "cum_unique_cpty", "cum_cross_bank"]

def _asof_account_state(role_account_col):
    """For each original row, look up that row's account's running state as
    of the last touch (either role) with orig_idx STRICTLY LESS than this
    row's own orig_idx -- which excludes this transaction's own two touches
    (both share the same orig_idx)."""
    keys = pd.DataFrame({
        "account": df[role_account_col].to_numpy(dtype=str),
        "orig_idx": _orig_idx,
    })
    return pd.merge_asof(
        keys, _touches[_touch_state_cols],
        on="orig_idx", by="account",
        direction="backward", allow_exact_matches=False,
    )

_src_state = _asof_account_state(SRC_COL)
_dst_state = _asof_account_state(DST_COL)
del _touches, _is_new_cpty

# first_seen is safe to compute globally (unlike everything else here): an
# account's first-ever transaction time is the same value regardless of what
# future data you can see, since it's a minimum. Combines both roles.
_acct_first_seen = pd.concat([
    df.groupby(SRC_COL, observed=True)[TS_COL].min(),
    df.groupby(DST_COL, observed=True)[TS_COL].min(),
], axis=1).min(axis=1)
_src_first_seen_ts = df[SRC_COL].map(_acct_first_seen)
_dst_first_seen_ts = df[DST_COL].map(_acct_first_seen)

def _build_causal_peer_inputs(state, first_seen_ts):
    txn_count   = state["cum_txn_count"].fillna(0).astype(np.float32)
    amount_flow = state["cum_amount"].fillna(0).astype(np.float32)
    unique_cpty = state["cum_unique_cpty"].fillna(0).astype(np.float32)
    cross_bank_ratio = safe_divide(state["cum_cross_bank"].fillna(0), txn_count)
    active_days = ((df[TS_COL] - first_seen_ts).dt.total_seconds() / (3600 * 24)).fillna(0).clip(lower=0).astype(np.float32)
    txn_per_active_day = safe_divide(txn_count, active_days + 1)
    return txn_count, amount_flow, unique_cpty, cross_bank_ratio, txn_per_active_day

(_src_txn_count, _src_amount_flow, _src_unique_cpty,
 _src_cross_bank_ratio, _src_txn_per_active_day) = _build_causal_peer_inputs(_src_state, _src_first_seen_ts)
(_dst_txn_count, _dst_amount_flow, _dst_unique_cpty,
 _dst_cross_bank_ratio, _dst_txn_per_active_day) = _build_causal_peer_inputs(_dst_state, _dst_first_seen_ts)
del _src_state, _dst_state

# Normalize each component using the pooled (src-role + dst-role) causal
# distribution, so both roles share one consistent scale, then combine into
# the UNSUPERVISED peer-risk score. This is NOT a laundering label -- it's a
# behavioral abnormality / risk proxy. Weights are the original
# 0.25/0.25/0.20/0.15/0.10 values (which summed to 0.95 alongside the
# now-removed 0.05 pattern-flag term) renormalized to sum to 1.0.
def _pooled_norm(a, b):
    a = pd.Series(a).reset_index(drop=True)
    b = pd.Series(b).reset_index(drop=True)
    pooled = pd.concat([a, b], ignore_index=True)
    # Fit the P95 scale constant from only the train-period-chronological
    # rows of EACH role (a and b are both df-row-order-aligned), not the
    # full pooled series -- see NORM_TRAIN_FRAC / _NORM_FIT_N above.
    fit_pooled = pd.concat([a.iloc[:_NORM_FIT_N], b.iloc[:_NORM_FIT_N]], ignore_index=True)
    normed = robust_norm(pooled, fit_series=fit_pooled)
    half = len(a)
    return normed.iloc[:half].reset_index(drop=True), normed.iloc[half:].reset_index(drop=True)

_norm_txn_count_s, _norm_txn_count_d = _pooled_norm(_src_txn_count, _dst_txn_count)
_norm_amount_s, _norm_amount_d = _pooled_norm(_src_amount_flow, _dst_amount_flow)
_norm_cpty_s, _norm_cpty_d = _pooled_norm(_src_unique_cpty, _dst_unique_cpty)
_norm_tpad_s, _norm_tpad_d = _pooled_norm(_src_txn_per_active_day, _dst_txn_per_active_day)

def _peer_risk(norm_txn, norm_amt, norm_cpty, norm_tpad, cross_bank_ratio):
    # safe_divide returns a plain numpy array; the norm_* Series are
    # converted to numpy explicitly so this adds cleanly regardless of index.
    return (
        (0.25 / 0.95) * norm_txn.to_numpy() +
        (0.25 / 0.95) * norm_amt.to_numpy() +
        (0.20 / 0.95) * norm_cpty.to_numpy() +
        (0.15 / 0.95) * norm_tpad.to_numpy() +
        (0.10 / 0.95) * np.clip(np.asarray(cross_bank_ratio), 0, 1)
    ).clip(0, 1).astype(np.float32)

df["src_peer_risk_score"] = _peer_risk(_norm_txn_count_s, _norm_amount_s, _norm_cpty_s, _norm_tpad_s, _src_cross_bank_ratio)
df["dst_peer_risk_score"] = _peer_risk(_norm_txn_count_d, _norm_amount_d, _norm_cpty_d, _norm_tpad_d, _dst_cross_bank_ratio)

# Tau = threshold for "high-risk peer", calibrated from the pooled
# distribution of causal peer-risk scores across both roles. Fit on the
# train-period-only window (first _NORM_FIT_N chronological rows of each
# role), then applied to every row -- same fix as robust_norm above, so tau
# doesn't reflect val/test-period score distribution either.
_pooled_scores = np.concatenate([df["src_peer_risk_score"].to_numpy(), df["dst_peer_risk_score"].to_numpy()])
_pooled_scores_fit = np.concatenate([
    df["src_peer_risk_score"].to_numpy()[:_NORM_FIT_N],
    df["dst_peer_risk_score"].to_numpy()[:_NORM_FIT_N],
])
tau = float(np.quantile(_pooled_scores_fit, PEER_RISK_PERCENTILE))
if not np.isfinite(tau):
    tau = float(np.nanmax(_pooled_scores_fit))

df["src_is_high_risk_peer"] = (df["src_peer_risk_score"] >= tau).astype(np.int8)
df["dst_is_high_risk_peer"] = (df["dst_peer_risk_score"] >= tau).astype(np.int8)

print(f"Peer-risk threshold tau (P{int(PEER_RISK_PERCENTILE * 100)}, train-period-fit): {tau:.4f}")
print(f"High-risk peer (account, transaction) instances: "
      f"src={int(df['src_is_high_risk_peer'].sum())}  dst={int(df['dst_is_high_risk_peer'].sum())}")
print_mem_hint(df, "transactions after causal peer-risk computation")

# ===================== SOURCE-SIDE DAILY EXPOSURE =====================

print("Building source-side SLT exposure summaries...")

# Instead of creating a giant doubled exposure table, we summarize source exposure directly:
# account = source
# peer_account = destination

src_pair_day = (
    df.groupby([SRC_COL, "date_only", DST_COL], observed=True, as_index=False)
      .agg(
          pair_txn_count=(DST_COL, "size"),
          pair_amount=(AMT_PAID, "sum"),
          peer_is_high_risk=("dst_is_high_risk_peer", "max")
      )
)

# Strong tie definition:
# repeated interaction in a day OR above-median pair amount
src_pair_amt_median = float(src_pair_day["pair_amount"].median())
src_pair_day["is_strong_tie"] = (
    (src_pair_day["pair_txn_count"] > 1) |
    (src_pair_day["pair_amount"] > src_pair_amt_median)
).astype(np.int8)

# Precompute "high-risk only" numeric columns to avoid expensive groupby lambdas
src_pair_day["high_risk_pair_amount"] = (
    src_pair_day["pair_amount"] * src_pair_day["peer_is_high_risk"]
).astype(np.float32)

src_pair_day["high_risk_pair_txn_count"] = (
    src_pair_day["pair_txn_count"] * src_pair_day["peer_is_high_risk"]
).astype(np.int32)

src_pair_day["strong_tie_high_risk"] = (
    src_pair_day["is_strong_tie"] * src_pair_day["peer_is_high_risk"]
).astype(np.int8)

src_day_exposure = (
    src_pair_day.groupby([SRC_COL, "date_only"], observed=True, as_index=False)
               .agg(
                   total_neighbors=(DST_COL, "nunique"),
                   high_risk_neighbors=("peer_is_high_risk", "sum"),
                   total_tie_amount=("pair_amount", "sum"),
                   high_risk_tie_amount=("high_risk_pair_amount", "sum"),
                   total_ties=("pair_txn_count", "sum"),
                   high_risk_ties=("high_risk_pair_txn_count", "sum"),
                   strong_ties_total=("is_strong_tie", "sum"),
                   strong_ties_high_risk=("strong_tie_high_risk", "sum"),
               )
               .rename(columns={SRC_COL: "account"})
)

# Clean up big intermediate table once done
del src_pair_day
gc.collect()

# ===================== DESTINATION-SIDE DAILY EXPOSURE =====================

print("Building destination-side SLT exposure summaries...")

# Destination-side exposure:
# account = destination
# peer_account = source

dst_pair_day = (
    df.groupby([DST_COL, "date_only", SRC_COL], observed=True, as_index=False)
      .agg(
          pair_txn_count=(SRC_COL, "size"),
          pair_amount=(AMT_REC, "sum"),
          peer_is_high_risk=("src_is_high_risk_peer", "max")
      )
)

dst_pair_amt_median = float(dst_pair_day["pair_amount"].median())
dst_pair_day["is_strong_tie"] = (
    (dst_pair_day["pair_txn_count"] > 1) |
    (dst_pair_day["pair_amount"] > dst_pair_amt_median)
).astype(np.int8)

dst_pair_day["high_risk_pair_amount"] = (
    dst_pair_day["pair_amount"] * dst_pair_day["peer_is_high_risk"]
).astype(np.float32)

dst_pair_day["high_risk_pair_txn_count"] = (
    dst_pair_day["pair_txn_count"] * dst_pair_day["peer_is_high_risk"]
).astype(np.int32)

dst_pair_day["strong_tie_high_risk"] = (
    dst_pair_day["is_strong_tie"] * dst_pair_day["peer_is_high_risk"]
).astype(np.int8)

dst_day_exposure = (
    dst_pair_day.groupby([DST_COL, "date_only"], observed=True, as_index=False)
               .agg(
                   total_neighbors=(SRC_COL, "nunique"),
                   high_risk_neighbors=("peer_is_high_risk", "sum"),
                   total_tie_amount=("pair_amount", "sum"),
                   high_risk_tie_amount=("high_risk_pair_amount", "sum"),
                   total_ties=("pair_txn_count", "sum"),
                   high_risk_ties=("high_risk_pair_txn_count", "sum"),
                   strong_ties_total=("is_strong_tie", "sum"),
                   strong_ties_high_risk=("strong_tie_high_risk", "sum"),
               )
               .rename(columns={DST_COL: "account"})
)

del dst_pair_day
gc.collect()

# ===================== COMBINE BOTH SIDES INTO ACCOUNT-DAY EXPOSURE =====================

print("Combining source-side and destination-side exposure...")

# Concatenate the already-aggregated account-day summaries.
# This is much smaller than duplicating the raw transaction table first.
day_exposure = pd.concat([src_day_exposure, dst_day_exposure], ignore_index=True)

# Some account-days may appear in both source-side and destination-side summaries.
# Sum them together.
day_exposure = (
    day_exposure.groupby(["account", "date_only"], observed=True, as_index=False)
                .agg({
                    "total_neighbors": "sum",
                    "high_risk_neighbors": "sum",
                    "total_tie_amount": "sum",
                    "high_risk_tie_amount": "sum",
                    "total_ties": "sum",
                    "high_risk_ties": "sum",
                    "strong_ties_total": "sum",
                    "strong_ties_high_risk": "sum"
                })
)

del src_day_exposure, dst_day_exposure
gc.collect()

print_mem_hint(day_exposure, "day exposure table")

# Core SLT exposure features
day_exposure["SLT_susp_nbr_ratio"] = safe_divide(
    day_exposure["high_risk_neighbors"],
    day_exposure["total_neighbors"]
)

day_exposure["SLT_susp_amt_share"] = safe_divide(
    day_exposure["high_risk_tie_amount"],
    day_exposure["total_tie_amount"]
)

day_exposure["SLT_susp_txn_share"] = safe_divide(
    day_exposure["high_risk_ties"],
    day_exposure["total_ties"]
)

day_exposure["SLT_strong_tie_susp_ratio"] = safe_divide(
    day_exposure["strong_ties_high_risk"],
    day_exposure["strong_ties_total"]
)

for col in [
    "SLT_susp_nbr_ratio",
    "SLT_susp_amt_share",
    "SLT_susp_txn_share",
    "SLT_strong_tie_susp_ratio"
]:
    day_exposure[col] = pd.to_numeric(day_exposure[col], errors="coerce").fillna(0).clip(0, 1).astype(np.float32)

# ===================== TEMPORAL / LAGGED SLT FEATURES =====================

print("Computing lagged and cumulative SLT features...")

day_exposure = day_exposure.sort_values(["account", "date_only"]).reset_index(drop=True)

base_slt_cols = [
    "SLT_susp_nbr_ratio",
    "SLT_susp_amt_share",
    "SLT_susp_txn_share",
    "SLT_strong_tie_susp_ratio"
]

# lag1 and lag2 per account
for base_col in base_slt_cols:
    day_exposure[f"{base_col}_lag1"] = (
        day_exposure.groupby("account", sort=False)[base_col]
                    .shift(1)
                    .fillna(0)
                    .astype(np.float32)
    )
    day_exposure[f"{base_col}_lag2"] = (
        day_exposure.groupby("account", sort=False)[base_col]
                    .shift(2)
                    .fillna(0)
                    .astype(np.float32)
    )

# Temporal shift / acceleration of risky exposure
day_exposure["SLT_exposure_delta"] = (
    day_exposure["SLT_susp_nbr_ratio_lag1"] - day_exposure["SLT_susp_nbr_ratio_lag2"]
).fillna(0).astype(np.float32)

# Rolling cumulative recent exposure (7 prior days, excluding current day)
day_exposure["SLT_cum_exposure_7d"] = (
    day_exposure.groupby("account", sort=False)["SLT_susp_nbr_ratio"]
                .transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum())
                .fillna(0)
                .astype(np.float32)
)

day_exposure["SLT_cum_amt_exposure_7d"] = (
    day_exposure.groupby("account", sort=False)["SLT_susp_amt_share"]
                .transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum())
                .fillna(0)
                .astype(np.float32)
)

# Normalize cumulative features. day_exposure is sorted by (account, date),
# not globally by date, so a row-index cutoff doesn't isolate the train
# period the way it does for the transaction-level df above -- use the date
# cutoff (_TRAIN_CUTOFF_TS) instead to build the fit subset, same idea as
# robust_norm's fit_series elsewhere in this file.
_day_exposure_fit_mask = day_exposure["date_only"] < _TRAIN_CUTOFF_TS
day_exposure["SLT_cum_exposure_7d_norm"] = robust_norm(
    day_exposure["SLT_cum_exposure_7d"],
    fit_series=day_exposure.loc[_day_exposure_fit_mask, "SLT_cum_exposure_7d"],
)
day_exposure["SLT_cum_amt_exposure_7d_norm"] = robust_norm(
    day_exposure["SLT_cum_amt_exposure_7d"],
    fit_series=day_exposure.loc[_day_exposure_fit_mask, "SLT_cum_amt_exposure_7d"],
)

# ===================== MERGE SLT FEATURES BACK TO TRANSACTIONS =====================

print("Merging SLT account-day features back onto transactions...")

# Prepare source-side feature table
src_day_features = day_exposure.copy()
src_day_features = src_day_features.add_prefix("src_")
src_day_features = src_day_features.rename(columns={
    "src_account": SRC_COL,
    "src_date_only": "date_only"
})

# Prepare destination-side feature table
dst_day_features = day_exposure.copy()
dst_day_features = dst_day_features.add_prefix("dst_")
dst_day_features = dst_day_features.rename(columns={
    "dst_account": DST_COL,
    "dst_date_only": "date_only"
})

# Merge source features
df = df.merge(src_day_features, on=[SRC_COL, "date_only"], how="left")

# Merge destination features
df = df.merge(dst_day_features, on=[DST_COL, "date_only"], how="left")

# Clean up
del src_day_features, dst_day_features, day_exposure
gc.collect()

# Fill missing SLT feature values with 0
slt_cols = [c for c in df.columns if "SLT_" in c]
for col in slt_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

print_mem_hint(df, "transactions after SLT merge")

# ===================== OPTIONAL ENTITY CONTEXT =====================

print("Merging optional entity metadata...")

# Add entity info if available using explicit merge keys.
# This avoids pandas creating duplicate anonymous key columns on repeated joins.
df["src_account_key"] = df[SRC_COL].astype(str)
df["dst_account_key"] = df[DST_COL].astype(str)

src_acct_meta = (
    df_acct.add_prefix("srcacct_")
           .reset_index()
           .rename(columns={ACCT_ID_COL: "src_account_key"})
)

dst_acct_meta = (
    df_acct.add_prefix("dstacct_")
           .reset_index()
           .rename(columns={ACCT_ID_COL: "dst_account_key"})
)

df = df.merge(src_acct_meta, on="src_account_key", how="left")
df = df.merge(dst_acct_meta, on="dst_account_key", how="left")

df = df.drop(columns=["src_account_key", "dst_account_key"])

if ACCT_ENTITY_ID in df_acct.columns:
    src_entity_col = f"srcacct_{ACCT_ENTITY_ID}"
    dst_entity_col = f"dstacct_{ACCT_ENTITY_ID}"
    if src_entity_col in df.columns and dst_entity_col in df.columns:
        df["src_entity_id"] = df[src_entity_col]
        df["dst_entity_id"] = df[dst_entity_col]
        df["SLT_same_entity"] = (df["src_entity_id"].astype(str) == df["dst_entity_id"].astype(str)).astype(np.int8)
    else:
        df["SLT_same_entity"] = 0
else:
    df["SLT_same_entity"] = 0

# ===================== FINAL TRANSACTION-LEVEL SLT SCORE =====================

print("Computing final transaction-level SLT score...")

# Source score: how much the sender has recently been exposed to risky peers
# Weights are set by CLI args (--w_neighbor, --w_amount, --w_strong_tie, --w_delta, --w_cum)
df["SLT_src_score"] = (
    W_NEIGHBOR   * df["src_SLT_susp_nbr_ratio_lag1"] +
    W_AMOUNT     * df["src_SLT_susp_amt_share_lag1"] +
    W_STRONG_TIE * df["src_SLT_strong_tie_susp_ratio_lag1"] +
    W_DELTA      * df["src_SLT_exposure_delta"].clip(lower=0) +
    W_CUM        * df["src_SLT_cum_exposure_7d_norm"]
).clip(0, 1).astype(np.float32)

# Destination score: same idea for the receiver
df["SLT_dst_score"] = (
    W_NEIGHBOR   * df["dst_SLT_susp_nbr_ratio_lag1"] +
    W_AMOUNT     * df["dst_SLT_susp_amt_share_lag1"] +
    W_STRONG_TIE * df["dst_SLT_strong_tie_susp_ratio_lag1"] +
    W_DELTA      * df["dst_SLT_exposure_delta"].clip(lower=0) +
    W_CUM        * df["dst_SLT_cum_exposure_7d_norm"]
).clip(0, 1).astype(np.float32)

# Final transaction score WITHOUT CROSS-BANK!!!
df["SLT_score"] = (
    0.55 * df["SLT_src_score"] +
    0.45 * df["SLT_dst_score"]
).clip(0, 1).astype(np.float32)

df["SLT_score"] = df["SLT_score"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1).astype(np.float32)

# Keep source/destination semantics while making every SLT feature discoverable
# by graph builders that select columns with the "SLT_" prefix.
slt_rename_map = {}
for col in df.columns:
    if col.startswith("src_SLT_"):
        slt_rename_map[col] = col.replace("src_SLT_", "SLT_src_", 1)
    elif col.startswith("dst_SLT_"):
        slt_rename_map[col] = col.replace("dst_SLT_", "SLT_dst_", 1)

if slt_rename_map:
    df = df.rename(columns=slt_rename_map)
    print(f"Renamed {len(slt_rename_map)} source/destination SLT columns to SLT_* prefix.")

# ===================== LOW / MEDIUM / HIGH INJECTION OUTPUTS =====================

print("Creating low / medium / high SLT injected datasets...")

# IMPORTANT:
# Labels are only used HERE, at the final step, to decide which laundering rows
# get the SLT injection flag. This mirrors the RAT injector design.
launder_mask = df[LABEL_COL] == 1

launder_scores = (
    df.loc[launder_mask, "SLT_score"]
      .replace([np.inf, -np.inf], np.nan)
      .dropna()
      .astype(float)
      .values
)

print(f"Valid laundering SLT_scores: {len(launder_scores)}")

if len(launder_scores) < 10:
    raise RuntimeError("Too few valid laundering SLT scores — check calculations.")

# Cast SLT columns to float32 / int8 for smaller output
for col in df.columns:
    if col.startswith("SLT_") or "_SLT_" in col:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(np.int8)

# ===================== INJECTION RULES (FIX) =====================
# Previously this loop only set the SLT_injected metadata flag, so the
# low/medium/high CSVs had IDENTICAL feature values (the flag is excluded from
# model inputs downstream). This fix makes injection real: for the selected
# laundering rows, the SLT exposure features are boosted toward a high-exposure
# profile, and the SLT scores are recomputed, so each intensity level actually
# differs in feature space. Intensity = prevalence (how many laundering rows
# carry the boosted pattern); the boost strength is constant.

BOOST_ALPHA = 0.7  # blend strength toward the high-exposure target value

_base = ["susp_nbr_ratio", "susp_amt_share", "susp_txn_share", "strong_tie_susp_ratio"]
BOOST_COLS = []
for _s in ["src", "dst"]:
    for _b in _base:
        BOOST_COLS += [f"SLT_{_s}_{_b}", f"SLT_{_s}_{_b}_lag1", f"SLT_{_s}_{_b}_lag2"]
    BOOST_COLS += [f"SLT_{_s}_cum_exposure_7d_norm", f"SLT_{_s}_cum_amt_exposure_7d_norm",
                   f"SLT_{_s}_exposure_delta"]
BOOST_COLS = [c for c in BOOST_COLS if c in df.columns]
SCORE_COLS = ["SLT_src_score", "SLT_dst_score", "SLT_score"]

# High-exposure target per feature: 95th percentile of its POSITIVE values
# (these exposure features are sparse, so the overall q95 is often 0).
_hi = {}
for _c in BOOST_COLS:
    _v = df[_c].values
    _pos = _v[_v > 0]
    _hi[_c] = float(np.quantile(_pos, 0.95)) if len(_pos) else 1.0

# df is still fully pristine here -- nothing below has boosted it yet.
# Dumping now gives a CSV with the exact same rows/order as every
# HI-Small_Trans_SLT_<intensity>.csv, but with SLT_* features at their
# never-boosted values.
if _args.dump_pristine:
    pristine_path = os.path.join(OUTPUT_DIR, f"HI-Small_Trans_SLT_pristine{_suffix}.csv") \
        if VARIANT == "current" else \
        os.path.join(OUTPUT_DIR, f"HI-Small_Trans_SLT_{VARIANT}_pristine{_suffix}.csv")
    print(f"Saving pristine (un-boosted) snapshot: {pristine_path}")
    # Explicit date_format: pandas' to_csv formats a whole datetime64 column
    # as date-only ("YYYY-MM-DD", no time) if EVERY value in it happens to
    # land exactly at midnight, and with full "YYYY-MM-DD HH:MM:SS"
    # otherwise -- a column-wide decision, not per-row. Forcing the format
    # here guarantees this CSV's Timestamp column is unambiguous, so
    # downstream readers (graph builders) that call pd.to_datetime()
    # without an explicit format can't hit a format-inference mismatch.
    df.to_csv(pristine_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"Saved {pristine_path} [0 injected rows by construction]")

for name, frac in INTENSITIES.items():
    if _args.selection == "score":
        threshold = float(np.quantile(launder_scores, 1 - frac))
        print(f"{name}: threshold = {threshold:.4f}")

        # selection uses the pristine SLT_score (as before)
        df["SLT_injected"] = (
            (df[LABEL_COL] == 1) &
            (df["SLT_score"] > threshold)
        ).astype(np.int8)
    else:
        # Placebo control: select a random sample of laundering rows of the
        # same size the score-based threshold would have selected, instead of
        # ranking by SLT_score. Boost mechanics afterward are identical.
        _sel_rng = np.random.default_rng(
            _args.selection_seed + {"low": 0, "medium": 1, "high": 2}[name]
        )
        n_select = int(round(frac * int(launder_mask.sum())))
        launder_idx = df.index[launder_mask]
        chosen_idx = _sel_rng.choice(launder_idx, size=min(n_select, len(launder_idx)), replace=False)
        print(f"{name}: random selection = {n_select} of {len(launder_idx)} laundering rows")

        df["SLT_injected"] = np.int8(0)
        df.loc[chosen_idx, "SLT_injected"] = 1

    inj = df["SLT_injected"] == 1

    # snapshot pristine values of the rows we are about to boost
    _saved = {c: df.loc[inj, c].copy() for c in BOOST_COLS + SCORE_COLS}

    # boost exposure features for injected rows (upward only)
    for c in BOOST_COLS:
        cur = df.loc[inj, c].astype(float)
        df.loc[inj, c] = np.maximum(cur, cur + BOOST_ALPHA * (_hi[c] - cur)).astype(np.float32)

    # recompute SLT scores from the boosted features (same formulas as above)
    df["SLT_src_score"] = (
        W_NEIGHBOR   * df["SLT_src_susp_nbr_ratio_lag1"] +
        W_AMOUNT     * df["SLT_src_susp_amt_share_lag1"] +
        W_STRONG_TIE * df["SLT_src_strong_tie_susp_ratio_lag1"] +
        W_DELTA      * df["SLT_src_exposure_delta"].clip(lower=0) +
        W_CUM        * df["SLT_src_cum_exposure_7d_norm"]
    ).clip(0, 1).astype(np.float32)

    df["SLT_dst_score"] = (
        W_NEIGHBOR   * df["SLT_dst_susp_nbr_ratio_lag1"] +
        W_AMOUNT     * df["SLT_dst_susp_amt_share_lag1"] +
        W_STRONG_TIE * df["SLT_dst_strong_tie_susp_ratio_lag1"] +
        W_DELTA      * df["SLT_dst_exposure_delta"].clip(lower=0) +
        W_CUM        * df["SLT_dst_cum_exposure_7d_norm"]
    ).clip(0, 1).astype(np.float32)

    df["SLT_score"] = (
        0.55 * df["SLT_src_score"] +
        0.45 * df["SLT_dst_score"]
    ).clip(0, 1).astype(np.float32)

    df["SLT_intensity_level"] = (
        df["SLT_injected"] * {"low": 1, "medium": 2, "high": 3}[name]
    ).astype(np.int8)

    if VARIANT == "current":
        out_path = os.path.join(OUTPUT_DIR, f"HI-Small_Trans_SLT_{name}{_suffix}.csv")
    else:
        out_path = os.path.join(OUTPUT_DIR, f"HI-Small_Trans_SLT_{VARIANT}_{name}{_suffix}.csv")
    print(f"Saving: {out_path}")
    df.to_csv(out_path, index=False, date_format="%Y-%m-%d %H:%M:%S")  # see pristine dump's comment above
    print(f"Saved {out_path} [{int(df['SLT_injected'].sum())} injected rows]")

    # restore pristine values so boosts never accumulate across intensities
    for c in BOOST_COLS + SCORE_COLS:
        df.loc[inj, c] = _saved[c]

print("DONE.")
