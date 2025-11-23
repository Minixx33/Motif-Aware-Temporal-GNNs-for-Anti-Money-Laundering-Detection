"""
motif_feature_extractor_ibm_small.py
------------------------------------
Build per-account (node) features for the IBM HI-Small_Trans_RAT dataset.

Features include:
  - fan_in, fan_out
  - mean / std in/out amount
  - total_tx
  - activity window (first/last timestamp, active_days, tx_per_day)
  - motif role counts (fanin / chain / cycle as src/dst)
  - motif_participation, motif_ratio

Input:
  - HI-Small_Trans_RAT.csv (RAT-injected IBM HI-Small)

Output:
  - HI-Small_motif_features.csv (one row per account)
"""

import os
import numpy as np
import pandas as pd

# ===================== CONFIG =====================

INPUT_PATH = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets\RAT\HI-Small_Trans_RAT.csv"

OUTPUT_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets\RAT"
OUTPUT_FILE = "HI-Small_motif_features.csv"

SRC_COL = "Account"
DST_COL = "Account.1"
AMT_REC_COL = "Amount Received"
AMT_PAID_COL = "Amount Paid"
TS_COL = "Timestamp"
LABEL_COL = "Is Laundering"
MOTIF_COL = "injected_motif"

# ===================== LOAD DATA =====================

print(f"Loading RAT-injected IBM HI-Small from:\n  {INPUT_PATH}")
tx = pd.read_csv(INPUT_PATH, low_memory=False)

print(f"Loaded {len(tx):,} transactions")

required_cols = [
    SRC_COL, DST_COL, AMT_REC_COL, AMT_PAID_COL,
    TS_COL, LABEL_COL, MOTIF_COL
]
for c in required_cols:
    if c not in tx.columns:
        raise ValueError(f"Required column '{c}' not found in dataset.")

# Ensure account IDs are strings
tx[SRC_COL] = tx[SRC_COL].astype(str)
tx[DST_COL] = tx[DST_COL].astype(str)

# Parse timestamps
tx[TS_COL] = pd.to_datetime(tx[TS_COL], errors="coerce")
if tx[TS_COL].isna().any():
    bad = tx[tx[TS_COL].isna()].head()
    raise ValueError(f"Some timestamps could not be parsed. Example bad rows:\n{bad}")

print(f"Timestamp range: {tx[TS_COL].min()} -> {tx[TS_COL].max()}")

# ===================== BUILD ACCOUNT UNIVERSE =====================

accounts_src = tx[SRC_COL].unique()
accounts_dst = tx[DST_COL].unique()
all_accounts = pd.Index(accounts_src).append(pd.Index(accounts_dst)).unique()

print(f"Unique accounts: {len(all_accounts):,}")

accounts_df = pd.DataFrame({"acct_id": all_accounts})

# ===================== BASIC DEGREE / AMOUNT FEATURES =====================

print("Computing fan_in / fan_out...")

fan_out = tx.groupby(SRC_COL).size()            # out-degree
fan_in = tx.groupby(DST_COL).size()             # in-degree

accounts_df["fan_out"] = accounts_df["acct_id"].map(fan_out).fillna(0).astype(float)
accounts_df["fan_in"] = accounts_df["acct_id"].map(fan_in).fillna(0).astype(float)
accounts_df["total_tx"] = accounts_df["fan_in"] + accounts_df["fan_out"]

print("Computing amount statistics...")

# Outgoing amounts: Amount Paid by source
out_amt_mean = tx.groupby(SRC_COL)[AMT_PAID_COL].mean()
out_amt_std = tx.groupby(SRC_COL)[AMT_PAID_COL].std()

# Incoming amounts: Amount Received by destination
in_amt_mean = tx.groupby(DST_COL)[AMT_REC_COL].mean()
in_amt_std = tx.groupby(DST_COL)[AMT_REC_COL].std()

accounts_df["mean_out_amt"] = accounts_df["acct_id"].map(out_amt_mean).fillna(0.0)
accounts_df["std_out_amt"] = accounts_df["acct_id"].map(out_amt_std).fillna(0.0)
accounts_df["mean_in_amt"] = accounts_df["acct_id"].map(in_amt_mean).fillna(0.0)
accounts_df["std_in_amt"] = accounts_df["acct_id"].map(in_amt_std).fillna(0.0)

# ===================== ACTIVITY WINDOW FEATURES =====================

print("Computing activity window features (first/last timestamp, active_days, tx_per_day)...")

# Build per-account activity: combine src/dst appearances
act_src = tx[[SRC_COL, TS_COL]].rename(columns={SRC_COL: "acct_id"})
act_dst = tx[[DST_COL, TS_COL]].rename(columns={DST_COL: "acct_id"})
activity = pd.concat([act_src, act_dst], ignore_index=True)

first_ts = activity.groupby("acct_id")[TS_COL].min()
last_ts = activity.groupby("acct_id")[TS_COL].max()

accounts_df["first_ts"] = accounts_df["acct_id"].map(first_ts)
accounts_df["last_ts"] = accounts_df["acct_id"].map(last_ts)

# Active days (at least 1)
delta = (accounts_df["last_ts"] - accounts_df["first_ts"]).dt.days
accounts_df["active_days"] = delta.fillna(0).astype(int)
accounts_df.loc[accounts_df["active_days"] < 1, "active_days"] = 1

accounts_df["tx_per_day"] = accounts_df["total_tx"] / accounts_df["active_days"]

# ===================== MOTIF PARTICIPATION FEATURES =====================

print("Computing motif participation features...")

motif_tx = tx[tx[MOTIF_COL].notna()].copy()
motif_tx[MOTIF_COL] = motif_tx[MOTIF_COL].astype(str)

if motif_tx.empty:
    print("Warning: No injected motifs found in dataset. Motif features will be zero.")
    motif_types = []
else:
    motif_types = sorted(motif_tx[MOTIF_COL].unique())
    print(f"Found {len(motif_types)} motif types: {motif_types}")
    print("Motif counts:")
    print(motif_tx[MOTIF_COL].value_counts())

# Initialize motif role columns
for m in motif_types:
    accounts_df[f"{m}_as_src"] = 0.0
    accounts_df[f"{m}_as_dst"] = 0.0

# Fill motif role counts
for m in motif_types:
    sub = motif_tx[motif_tx[MOTIF_COL] == m]

    src_counts = sub.groupby(SRC_COL).size()
    dst_counts = sub.groupby(DST_COL).size()

    accounts_df[f"{m}_as_src"] = accounts_df["acct_id"].map(src_counts).fillna(0.0)
    accounts_df[f"{m}_as_dst"] = accounts_df["acct_id"].map(dst_counts).fillna(0.0)

# Total motif participation across all motif types and roles
motif_cols = [c for c in accounts_df.columns if c.endswith("_as_src") or c.endswith("_as_dst")]
if motif_cols:
    accounts_df["motif_participation"] = accounts_df[motif_cols].sum(axis=1)
else:
    accounts_df["motif_participation"] = 0.0

# Ratio of motif-related transactions to total transactions
accounts_df["motif_ratio"] = 0.0
nonzero_tx_mask = accounts_df["total_tx"] > 0
accounts_df.loc[nonzero_tx_mask, "motif_ratio"] = (
    accounts_df.loc[nonzero_tx_mask, "motif_participation"] / accounts_df.loc[nonzero_tx_mask, "total_tx"]
)

# ===================== SAVE FEATURES =====================

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

# Sort by account id for consistency
accounts_df = accounts_df.sort_values("acct_id").reset_index(drop=True)

accounts_df.to_csv(output_path, index=False)
print(f"Saved motif-aware node features to:\n  {output_path}")
print(f"Number of accounts: {len(accounts_df):,}")
print("Top 5 rows:")
print(accounts_df.head())
