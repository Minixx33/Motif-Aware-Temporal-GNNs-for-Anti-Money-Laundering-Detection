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

import os
import gc
from pathlib import Path

import numpy as np
import pandas as pd

# ===================== CONFIG =====================

# Resolve project root from this file's location so paths work on Windows / Linux / macOS.
# scripts/SLT/slt_injector.py  →  parents[2] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = str(PROJECT_ROOT / "ibm_transcations_datasets")

TX_CSV_PATH       = os.path.join(BASE_DIR, "HI-Small_Trans.csv")
ACCOUNTS_CSV_PATH = os.path.join(BASE_DIR, "HI-Small_accounts.csv")
PATTERNS_TXT_PATH = os.path.join(BASE_DIR, "HI-Small_patterns.txt")   # optional

OUTPUT_DIR = os.path.join(BASE_DIR, "SLT")
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
    "low": 0.15,
    "medium": 0.30,
    "high": 0.60,
}

# Top 5% by unsupervised peer-risk score are considered "high-risk peers"
PEER_RISK_PERCENTILE = 0.95

EPS = 1e-8


# ===================== HELPERS =====================

def safe_divide(a, b):
    """Safe division that returns 0 where denominator is 0."""
    a = pd.to_numeric(a, errors="coerce").fillna(0).astype(np.float32)
    b = pd.to_numeric(b, errors="coerce").fillna(0).astype(np.float32)
    return np.where(np.abs(b) < EPS, 0.0, a / (b + EPS)).astype(np.float32)


def robust_norm(series, q=0.95):
    """
    Normalize by a robust upper quantile instead of max.
    This keeps huge outliers from dominating the scale.
    Output is clipped to [0,1].
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(np.float32)
    qv = float(s.quantile(q))
    if (not np.isfinite(qv)) or (qv <= 0):
        qv = float(s.max())
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
df_acct = df_acct.set_index(ACCT_ID_COL)

# Keep account metadata lightweight
# We only really need entity info if available.
if ACCT_ENTITY_ID in df_acct.columns:
    df_acct[ACCT_ENTITY_ID] = df_acct[ACCT_ENTITY_ID].astype(str)

if ACCT_ENTITY_NAME in df_acct.columns:
    df_acct[ACCT_ENTITY_NAME] = df_acct[ACCT_ENTITY_NAME].astype(str)

print_mem_hint(df_acct, "accounts table")

# ===================== ACCOUNT-LEVEL BEHAVIORAL RISK FEATURES =====================

print("Computing account-level behavioral statistics...")

# Outgoing stats per account
src_stats = df.groupby(SRC_COL, observed=True).agg(
    src_txn_count=(SRC_COL, "size"),
    src_total_paid=(AMT_PAID, "sum"),
    src_avg_paid=(AMT_PAID, "mean"),
    src_unique_dsts=(DST_COL, "nunique"),
    src_cross_bank_ratio=("is_cross_bank", "mean"),
    src_off_hours_ratio=("is_off_hours", "mean"),
    src_weekend_ratio=("is_weekend", "mean"),
    src_first_seen=(TS_COL, "min"),
    src_last_seen=(TS_COL, "max"),
)

# Incoming stats per account
dst_stats = df.groupby(DST_COL, observed=True).agg(
    dst_txn_count=(DST_COL, "size"),
    dst_total_received=(AMT_REC, "sum"),
    dst_avg_received=(AMT_REC, "mean"),
    dst_unique_srcs=(SRC_COL, "nunique"),
    dst_first_seen=(TS_COL, "min"),
    dst_last_seen=(TS_COL, "max"),
)

# Merge into a single account stats table
acct_stats = src_stats.join(dst_stats, how="outer").fillna(0)

# First/last seen over both directions
src_first = acct_stats["src_first_seen"].replace(0, pd.NaT)
dst_first = acct_stats["dst_first_seen"].replace(0, pd.NaT)
src_last = acct_stats["src_last_seen"].replace(0, pd.NaT)
dst_last = acct_stats["dst_last_seen"].replace(0, pd.NaT)

acct_stats["first_seen"] = pd.concat([src_first, dst_first], axis=1).min(axis=1)
acct_stats["last_seen"]  = pd.concat([src_last, dst_last], axis=1).max(axis=1)

acct_stats["active_days"] = (
    (acct_stats["last_seen"] - acct_stats["first_seen"]).dt.total_seconds() / (3600 * 24)
).fillna(0).clip(lower=0).astype(np.float32)

# Combine incoming + outgoing behavior
acct_stats["total_txn_count"] = (
    pd.to_numeric(acct_stats["src_txn_count"], errors="coerce").fillna(0).astype(np.int32) +
    pd.to_numeric(acct_stats["dst_txn_count"], errors="coerce").fillna(0).astype(np.int32)
)

acct_stats["total_amount_flow"] = (
    pd.to_numeric(acct_stats["src_total_paid"], errors="coerce").fillna(0).astype(np.float32) +
    pd.to_numeric(acct_stats["dst_total_received"], errors="coerce").fillna(0).astype(np.float32)
)

acct_stats["total_unique_counterparties"] = (
    pd.to_numeric(acct_stats["src_unique_dsts"], errors="coerce").fillna(0).astype(np.int32) +
    pd.to_numeric(acct_stats["dst_unique_srcs"], errors="coerce").fillna(0).astype(np.int32)
)

# Approximate burstiness / intensity of activity over life of account
acct_stats["txn_per_active_day"] = safe_divide(
    acct_stats["total_txn_count"],
    acct_stats["active_days"] + 1
)

# Optional pattern flag
acct_stats["pattern_flag"] = acct_stats.index.astype(str).isin(pattern_accounts).astype(np.int8)

# Normalize the core components
acct_stats["norm_total_txn_count"] = robust_norm(acct_stats["total_txn_count"])
acct_stats["norm_total_amount_flow"] = robust_norm(acct_stats["total_amount_flow"])
acct_stats["norm_unique_counterparties"] = robust_norm(acct_stats["total_unique_counterparties"])
acct_stats["norm_txn_per_active_day"] = robust_norm(acct_stats["txn_per_active_day"])
acct_stats["norm_cross_bank_ratio"] = pd.to_numeric(acct_stats["src_cross_bank_ratio"], errors="coerce").fillna(0).clip(0, 1).astype(np.float32)
acct_stats["norm_pattern_flag"] = acct_stats["pattern_flag"].astype(np.float32)

# Build UNSUPERVISED peer-risk score
# This is NOT a laundering label. It is a behavioral abnormality / risk proxy.
acct_stats["peer_risk_score"] = (
    0.25 * acct_stats["norm_total_txn_count"] +
    0.25 * acct_stats["norm_total_amount_flow"] +
    0.20 * acct_stats["norm_unique_counterparties"] +
    0.15 * acct_stats["norm_txn_per_active_day"] +
    0.10 * acct_stats["norm_cross_bank_ratio"] +
    0.05 * acct_stats["norm_pattern_flag"]
).astype(np.float32).clip(0, 1)

# Tau = threshold for defining a high-risk peer
tau = float(acct_stats["peer_risk_score"].quantile(PEER_RISK_PERCENTILE))
if (not np.isfinite(tau)):
    tau = float(acct_stats["peer_risk_score"].max())

acct_stats["is_high_risk_peer"] = (acct_stats["peer_risk_score"] >= tau).astype(np.int8)

print(f"Peer-risk threshold tau (P{int(PEER_RISK_PERCENTILE * 100)}): {tau:.4f}")
print(f"High-risk peer accounts: {int(acct_stats['is_high_risk_peer'].sum())}")
print_mem_hint(acct_stats, "account stats")

# Attach peer-risk info to each transaction row
print("Mapping peer-risk metadata onto transactions...")
df["src_peer_risk_score"] = df[SRC_COL].astype(str).map(acct_stats["peer_risk_score"]).fillna(0).astype(np.float32)
df["dst_peer_risk_score"] = df[DST_COL].astype(str).map(acct_stats["peer_risk_score"]).fillna(0).astype(np.float32)

df["src_is_high_risk_peer"] = df[SRC_COL].astype(str).map(acct_stats["is_high_risk_peer"]).fillna(0).astype(np.int8)
df["dst_is_high_risk_peer"] = df[DST_COL].astype(str).map(acct_stats["is_high_risk_peer"]).fillna(0).astype(np.int8)

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
    day_exposure.groupby(["account", "date_only"], as_index=False)
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

# Normalize cumulative features
day_exposure["SLT_cum_exposure_7d_norm"] = robust_norm(day_exposure["SLT_cum_exposure_7d"])
day_exposure["SLT_cum_amt_exposure_7d_norm"] = robust_norm(day_exposure["SLT_cum_amt_exposure_7d"])

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
df["SLT_src_score"] = (
    0.30 * df["src_SLT_susp_nbr_ratio_lag1"] +
    0.25 * df["src_SLT_susp_amt_share_lag1"] +
    0.20 * df["src_SLT_strong_tie_susp_ratio_lag1"] +
    0.15 * df["src_SLT_exposure_delta"].clip(lower=0) +
    0.10 * df["src_SLT_cum_exposure_7d_norm"]
).clip(0, 1).astype(np.float32)

# Destination score: same idea for the receiver
df["SLT_dst_score"] = (
    0.30 * df["dst_SLT_susp_nbr_ratio_lag1"] +
    0.25 * df["dst_SLT_susp_amt_share_lag1"] +
    0.20 * df["dst_SLT_strong_tie_susp_ratio_lag1"] +
    0.15 * df["dst_SLT_exposure_delta"].clip(lower=0) +
    0.10 * df["dst_SLT_cum_exposure_7d_norm"]
).clip(0, 1).astype(np.float32)

# Final transaction score WITHOUT CROSS-BANK!!!
df["SLT_score"] = (
    0.55 * df["SLT_src_score"] +
    0.45 * df["SLT_dst_score"]
).clip(0, 1).astype(np.float32)

df["SLT_score"] = df["SLT_score"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1).astype(np.float32)

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

for name, frac in INTENSITIES.items():
    threshold = float(np.quantile(launder_scores, 1 - frac))
    print(f"{name}: threshold = {threshold:.4f}")

    df["SLT_injected"] = (
        (df[LABEL_COL] == 1) &
        (df["SLT_score"] > threshold)
    ).astype(np.int8)

    df["SLT_intensity_level"] = (
        df["SLT_injected"] * {"low": 1, "medium": 2, "high": 3}[name]
    ).astype(np.int8)

    out_path = os.path.join(OUTPUT_DIR, f"HI-Small_Trans_SLT_{name}.csv")
    print(f"Saving: {out_path}")
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} [{int(df['SLT_injected'].sum())} injected rows]")

print("DONE.")
