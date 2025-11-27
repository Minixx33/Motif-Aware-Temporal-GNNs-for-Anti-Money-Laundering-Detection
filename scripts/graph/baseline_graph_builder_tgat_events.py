# baseline_tgat_builder_FIXED.py
# ============================================================
# Baseline TGAT Event Graph Builder for IBM HI-Small (FIXED)
#
# FIXES:
# - Proper edge feature engineering (~17 features)
# - Rich node features with degrees (~9-21 features)  
# - Standard TGAT file names (src_nodes.pt, dst_nodes.pt, etc.)
# - No label leakage
#
# Produces event-based graph suitable for TGAT:
#   - src_nodes.pt
#   - dst_nodes.pt
#   - timestamps.pt
#   - edge_attr.pt
#   - y_edge.pt
#   - x_node.pt
#   - node_mapping.json
#   - edge_attr_cols.json
#   - feature_stats.json
#   - graph_stats.json
#
# Uses:
#   HI-Small_Trans.csv
#   HI-Small_accounts.csv
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import torch

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets"
OUT_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\tgat_graphs"

TRANS_FILE = "HI-Small_Trans.csv"
ACCOUNTS_FILE = "HI-Small_accounts.csv"

OUT_DIR = os.path.join(OUT_DIR, "HI-Small_Trans")
os.makedirs(OUT_DIR, exist_ok=True)

# Column names
TS_COL = "Timestamp"
FROM_BANK_COL = "From Bank"
TO_BANK_COL = "To Bank"
SRC_COL = "Account"
DST_COL = "Account.1"
AMT_REC_COL = "Amount Received"
AMT_PAID_COL = "Amount Paid"
CUR_REC_COL = "Receiving Currency"
CUR_PAY_COL = "Payment Currency"
PAY_FMT_COL = "Payment Format"
LABEL_COL = "Is Laundering"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def log1p_safe(x):
    """Safe log1p on numpy array."""
    x = np.asarray(x, dtype=np.float64)
    x = np.where(x < 0, 0.0, x)  # clamp negatives
    return np.log1p(x)


def parse_entity_type(entity_name: str) -> str:
    """
    Extract entity type from 'Entity Name' like:
      'Sole Proprietorship #50438' -> 'Sole Proprietorship'
      'Corporation #33520'         -> 'Corporation'
      'Partnership #35397'         -> 'Partnership'
    """
    if not isinstance(entity_name, str):
        return "Unknown"
    if "#" in entity_name:
        return entity_name.split("#")[0].strip()
    parts = entity_name.split()
    if len(parts) <= 1:
        return entity_name.strip()
    return " ".join(parts[:-1]).strip()


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BASELINE TGAT EVENT STREAM BUILDER (FIXED)")
print("=" * 70)

print("\nLoading transactions...")
trans_path = os.path.join(BASE_DIR, TRANS_FILE)
df = pd.read_csv(trans_path, low_memory=False)
print(f"  Loaded {len(df):,} transaction rows")

print("\nLoading accounts...")
accounts_path = os.path.join(BASE_DIR, ACCOUNTS_FILE)
df_acct = pd.read_csv(accounts_path, low_memory=False)
print(f"  Loaded {len(df_acct):,} account rows")

# Ensure correct types
df[SRC_COL] = df[SRC_COL].astype(str)
df[DST_COL] = df[DST_COL].astype(str)
df[TS_COL] = pd.to_datetime(df[TS_COL], errors="raise")

# Sort events chronologically (CRITICAL for TGAT!)
df = df.sort_values(TS_COL).reset_index(drop=True)
print(f"\nTimestamp range: {df[TS_COL].min()} -> {df[TS_COL].max()}")

# ============================================================
# NODE MAPPING (Account → integer ID)
# ============================================================

print("\n" + "=" * 70)
print("NODE MAPPING")
print("=" * 70)

all_accounts = pd.concat([df[SRC_COL], df[DST_COL]]).unique()
acct2id = {acct: i for i, acct in enumerate(all_accounts)}
num_nodes = len(acct2id)
print(f"Unique accounts (nodes): {num_nodes:,}")

src_ids = df[SRC_COL].map(acct2id).values
dst_ids = df[DST_COL].map(acct2id).values

print(f"Total events: {len(df):,}")

# ============================================================
# TEMPORAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL VALIDATION")
print("=" * 70)

timestamps = (df[TS_COL].astype("int64") // 10**9).values  # UNIX seconds
time_diffs = np.diff(timestamps)

if (time_diffs < 0).any():
    raise ValueError(" TGAT requires strictly non-decreasing timestamps!")

unique_ts = len(np.unique(timestamps))
print(f" Timestamps are non-decreasing")
print(f"Unique timestamps: {unique_ts:,} / {len(timestamps):,}")
print(f"Time range (UNIX): {timestamps.min()} -> {timestamps.max()}")

# ============================================================
# EDGE LABELS (TARGETS ONLY, NOT FEATURES)
# ============================================================

print("\n" + "=" * 70)
print("EDGE LABELS")
print("=" * 70)

y_edge = df[LABEL_COL].astype(int).values
num_laund = int(y_edge.sum())
pct_laund = 100.0 * y_edge.mean()

print(f"Laundering events: {num_laund:,} ({pct_laund:.4f}%)")
print(f"Normal events:     {len(y_edge) - num_laund:,} ({100.0 - pct_laund:.4f}%)")

# ============================================================
# EDGE FEATURES (PROPER FEATURE ENGINEERING)
# ============================================================

print("\n" + "=" * 70)
print("EDGE FEATURES (ENGINEERED)")
print("=" * 70)

# Log-transform amounts
amt_rec = df[AMT_REC_COL].astype(float).values
amt_paid = df[AMT_PAID_COL].astype(float).values

log_amt_rec = log1p_safe(amt_rec).astype(np.float32)
log_amt_paid = log1p_safe(amt_paid).astype(np.float32)

# Binary features
same_bank = (df[FROM_BANK_COL].astype(str) == df[TO_BANK_COL].astype(str)).astype(np.float32)
same_currency = (df[CUR_REC_COL].astype(str) == df[CUR_PAY_COL].astype(str)).astype(np.float32)

# Temporal features
hour_of_day = df[TS_COL].dt.hour.values.astype(np.float32)
day_of_week = df[TS_COL].dt.dayofweek.values.astype(np.float32)
is_weekend = (day_of_week >= 5).astype(np.float32)

# Normalized timestamps (0 to 1 scale)
ts_normalized = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-10)
ts_normalized = ts_normalized.astype(np.float32)

# Time since last transaction (per account)
# For source account: time since last outgoing transaction
time_since_last_src = np.zeros(len(df), dtype=np.float32)
last_src_time = {}
for i, (src, ts) in enumerate(zip(src_ids, timestamps)):
    if src in last_src_time:
        time_since_last_src[i] = float(ts - last_src_time[src])
    else:
        time_since_last_src[i] = 0.0  # First transaction for this account
    last_src_time[src] = ts

# For destination account: time since last incoming transaction
time_since_last_dst = np.zeros(len(df), dtype=np.float32)
last_dst_time = {}
for i, (dst, ts) in enumerate(zip(dst_ids, timestamps)):
    if dst in last_dst_time:
        time_since_last_dst[i] = float(ts - last_dst_time[dst])
    else:
        time_since_last_dst[i] = 0.0  # First transaction for this account
    last_dst_time[dst] = ts

# Log-transform time gaps (many transactions happen in quick succession)
log_time_since_src = np.log1p(time_since_last_src).astype(np.float32)
log_time_since_dst = np.log1p(time_since_last_dst).astype(np.float32)

# One-hot encoding for categorical features
cur_recv_dummies = pd.get_dummies(df[CUR_REC_COL].astype(str), prefix="cur_recv")
fmt_dummies = pd.get_dummies(df[PAY_FMT_COL].astype(str), prefix="fmt")

print(f"Receiving currencies: {len(cur_recv_dummies.columns)} distinct")
print(f"Payment formats:      {len(fmt_dummies.columns)} distinct")
print(" Added temporal features: hour, day, weekend, normalized_ts, time_since_last")

# Combine all edge features
edge_feat_df = pd.DataFrame({
    "log_amt_rec": log_amt_rec,
    "log_amt_paid": log_amt_paid,
    "same_bank": same_bank,
    "same_currency": same_currency,
    "hour_of_day": hour_of_day,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "ts_normalized": ts_normalized,
    "log_time_since_src": log_time_since_src,
    "log_time_since_dst": log_time_since_dst,
})

edge_feat_df = pd.concat([edge_feat_df, cur_recv_dummies, fmt_dummies], axis=1)
edge_feature_cols = list(edge_feat_df.columns)

# Convert to array
edge_attr = edge_feat_df.replace([np.inf, -np.inf], 0.0).fillna(0.0).values
edge_attr = edge_attr.astype(np.float32)

print(f"Total edge features: {len(edge_feature_cols)}")
print(f"edge_attr shape: {edge_attr.shape}")

# Validation
nan_edge = np.isnan(edge_attr).sum()
inf_edge = np.isinf(edge_attr).sum()
if nan_edge > 0 or inf_edge > 0:
    print(f"WARNING: edge_attr has NaN={nan_edge}, Inf={inf_edge}")
else:
    print(" No NaN/Inf in edge_attr")

# Feature statistics for normalization
edge_attr_means = edge_attr.mean(axis=0)
edge_attr_stds = edge_attr.std(axis=0)
edge_attr_mins = edge_attr.min(axis=0)
edge_attr_maxs = edge_attr.max(axis=0)

# ============================================================
# NODE FEATURES (STRUCTURE + METADATA, NO LABELS)
# ============================================================

print("\n" + "=" * 70)
print("NODE FEATURES (SAFE)")
print("=" * 70)

# Compute graph-based features (degrees)
out_deg_counts = pd.Series(src_ids).value_counts()
in_deg_counts = pd.Series(dst_ids).value_counts()

node_df = pd.DataFrame({
    "account": list(acct2id.keys()),
    "node_id": list(acct2id.values())
})

# Add degree features
node_df["out_degree"] = node_df["node_id"].map(out_deg_counts).fillna(0).astype(float)
node_df["in_degree"] = node_df["node_id"].map(in_deg_counts).fillna(0).astype(float)
node_df["total_degree"] = node_df["out_degree"] + node_df["in_degree"]

# Log-normalized degrees
node_df["log_out_degree"] = np.log1p(node_df["out_degree"])
node_df["log_in_degree"] = np.log1p(node_df["in_degree"])
node_df["log_total_degree"] = np.log1p(node_df["total_degree"])

# Merge account metadata
df_acct["Account Number"] = df_acct["Account Number"].astype(str)
df_acct["entity_type"] = df_acct["Entity Name"].apply(parse_entity_type)

node_df = node_df.merge(
    df_acct[["Account Number", "entity_type"]], 
    how="left",
    left_on="account",
    right_on="Account Number"
)

# One-hot encode entity types
node_df["entity_type"] = node_df["entity_type"].fillna("Unknown")
ent_dummies = pd.get_dummies(node_df["entity_type"], prefix="ent")

print(f"Entity types: {list(ent_dummies.columns)}")

# Combine all node features
node_feat_df = pd.concat([
    node_df[["out_degree", "in_degree", "total_degree",
             "log_out_degree", "log_in_degree", "log_total_degree"]],
    ent_dummies
], axis=1)

node_feature_cols = list(node_feat_df.columns)
node_features = node_feat_df.replace([np.inf, -np.inf], 0.0).fillna(0.0).values
node_features = node_features.astype(np.float32)

print(f"Total node features: {len(node_feature_cols)}")
print(f"x_node shape: {node_features.shape}")

# Validation
nan_node = np.isnan(node_features).sum()
inf_node = np.isinf(node_features).sum()
if nan_node > 0 or inf_node > 0:
    print(f"WARNING: x_node has NaN={nan_node}, Inf={inf_node}")
else:
    print(" No NaN/Inf in x_node")

# ============================================================
# SAVE OUTPUT FILES (STANDARD TGAT FORMAT)
# ============================================================

print("\n" + "=" * 70)
print("SAVING TENSORS (STANDARD TGAT FORMAT)")
print("=" * 70)
print(f"Output directory: {OUT_DIR}\n")

# Save tensors with standard TGAT names
torch.save(torch.tensor(src_ids, dtype=torch.long), 
           os.path.join(OUT_DIR, "src_nodes.pt"))
print("  - src_nodes.pt")

torch.save(torch.tensor(dst_ids, dtype=torch.long), 
           os.path.join(OUT_DIR, "dst_nodes.pt"))
print("  - dst_nodes.pt")

torch.save(torch.tensor(timestamps, dtype=torch.long), 
           os.path.join(OUT_DIR, "timestamps.pt"))
print("  - timestamps.pt")

torch.save(torch.tensor(edge_attr, dtype=torch.float32), 
           os.path.join(OUT_DIR, "edge_attr.pt"))
print("  - edge_attr.pt")

torch.save(torch.tensor(y_edge, dtype=torch.long), 
           os.path.join(OUT_DIR, "y_edge.pt"))
print("  - y_edge.pt")

torch.save(torch.tensor(node_features, dtype=torch.float32), 
           os.path.join(OUT_DIR, "x_node.pt"))
print("  - x_node.pt")

# Save metadata
with open(os.path.join(OUT_DIR, "node_mapping.json"), "w", encoding="utf-8") as f:
    json.dump(acct2id, f)
print("  - node_mapping.json")

with open(os.path.join(OUT_DIR, "edge_attr_cols.json"), "w", encoding="utf-8") as f:
    json.dump(edge_feature_cols, f, indent=2)
print("  - edge_attr_cols.json")

# Feature statistics
feature_stats = {
    "edge_attr_means": edge_attr_means.tolist(),
    "edge_attr_stds": edge_attr_stds.tolist(),
    "edge_attr_mins": edge_attr_mins.tolist(),
    "edge_attr_maxs": edge_attr_maxs.tolist(),
    "edge_attr_cols": edge_feature_cols,
    "node_feature_cols": node_feature_cols,
}

with open(os.path.join(OUT_DIR, "feature_stats.json"), "w", encoding="utf-8") as f:
    json.dump(feature_stats, f, indent=2)
print("  - feature_stats.json")

# Graph statistics
graph_stats = {
    "dataset_type": "Baseline",
    "num_nodes": int(num_nodes),
    "num_events": int(len(df)),
    "unique_timestamps": int(unique_ts),
    "num_edge_features": len(edge_feature_cols),
    "num_node_features": node_features.shape[1],
    "pct_laundering_events": float(pct_laund),
    "edge_feature_cols": edge_feature_cols,
    "node_feature_cols": node_feature_cols,
    "trans_file": TRANS_FILE,
    "accounts_file": ACCOUNTS_FILE,
    "format": "TGAT_event_stream",
    "label_leakage_fixed": True,
    "temporal_features_added": True,
    "file_names": "standard_tgat_format",
}

with open(os.path.join(OUT_DIR, "graph_stats.json"), "w", encoding="utf-8") as f:
    json.dump(graph_stats, f, indent=2)
print("  - graph_stats.json")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print(" TGAT EVENT STREAM COMPLETE")
print("=" * 70)
print(f"\nAll files saved to: {OUT_DIR}")
print(f"\nEdge features:  {edge_attr.shape[1]}")
print(f"Node features:  {node_features.shape[1]}")
print(f"Total events:   {len(df):,}")
print(f"Total nodes:    {num_nodes:,}")
print("\n Safe for TGAT training - no label leakage!")
print(" Standard TGAT file names (src_nodes.pt, dst_nodes.pt, etc.)")
print("=" * 70)