"""
graph_builder_baseline_static_ENHANCED.py
-----------------------------------------
Enhanced baseline static graph builder with temporal features.

ENHANCEMENTS:
- Timestamp normalization (0-1 scale)
- Time since last transaction (per account)
- Log-transformed time gaps
- Hour of day, day of week, is_weekend

Uses:
  - HI-Small_Trans.csv
  - HI-Small_accounts.csv

Outputs (to graphs/HI-Small_Trans/):
  - edge_index.pt      [2, E]
  - edge_attr.pt       [E, F_e]   (~20 features with temporal)
  - timestamps.pt      [E]
  - x.pt               [N, F_n]   (degrees + entity type)
  - y_edge.pt          [E]
  - y_node.pt          [N]
  - node_mapping.json
  - edge_attr_cols.json
  - feature_stats.json
  - graph_stats.json
"""

import os
import json
import numpy as np
import pandas as pd
import torch

# ======================================================================
# CONFIG – EDIT THESE PATHS FOR YOUR MACHINE
# ======================================================================

BASE_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets"
OUT_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\graphs"

TRANS_FILE = "HI-Small_Trans.csv"
ACCOUNTS_FILE = "HI-Small_accounts.csv"

DATASET_NAME = "HI-Small_Trans"
OUT_DIR = os.path.join(OUT_DIR, DATASET_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

# Column names
TS_COL = "Timestamp"
FROM_BANK_COL = "From Bank"
SRC_ACCT_COL = "Account"
TO_BANK_COL = "To Bank"
DST_ACCT_COL = "Account.1"
AMT_REC_COL = "Amount Received"
RCURR_COL = "Receiving Currency"
AMT_PAID_COL = "Amount Paid"
PCURR_COL = "Payment Currency"
PFORMAT_COL = "Payment Format"
LABEL_COL = "Is Laundering"

ACC_ACCT_COL = "Account Number"
ACC_ENTITY_NAME_COL = "Entity Name"

# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def log1p_safe(x):
    """Safe log1p on numpy array."""
    x = np.asarray(x, dtype=np.float64)
    x = np.where(x < 0, 0.0, x)
    return np.log1p(x)

def parse_entity_type(entity_name: str) -> str:
    """Extract entity type from 'Entity Name'."""
    if not isinstance(entity_name, str):
        return "Unknown"
    if "#" in entity_name:
        left = entity_name.split("#")[0].strip()
        return left
    parts = entity_name.split()
    if len(parts) <= 1:
        return entity_name.strip()
    return " ".join(parts[:-1]).strip()

# ======================================================================
# LOAD CSVs
# ======================================================================

print("=" * 70)
print("ENHANCED BASELINE STATIC GRAPH BUILDER")
print("(with temporal features: normalized ts, time since last)")
print("=" * 70)

trans_path = os.path.join(BASE_DIR, TRANS_FILE)
acct_path = os.path.join(BASE_DIR, ACCOUNTS_FILE)

print(f"\nLoading transactions from: {trans_path}")
df = pd.read_csv(trans_path, low_memory=False)
print(f"  Loaded {len(df):,} transaction rows")

print(f"\nLoading accounts from: {acct_path}")
df_acct = pd.read_csv(acct_path, low_memory=False)
print(f"  Loaded {len(df_acct):,} account rows")

df[TS_COL] = pd.to_datetime(df[TS_COL], errors="raise")
df = df.sort_values(TS_COL).reset_index(drop=True)
print(f"\nTimestamp range: {df[TS_COL].min()} -> {df[TS_COL].max()}")

# ======================================================================
# NODE MAPPING
# ======================================================================

print("\n" + "=" * 70)
print("NODE MAPPING")
print("=" * 70)

df[SRC_ACCT_COL] = df[SRC_ACCT_COL].astype(str)
df[DST_ACCT_COL] = df[DST_ACCT_COL].astype(str)

all_accts = pd.concat([df[SRC_ACCT_COL], df[DST_ACCT_COL]], axis=0).unique()
acct2idx = {acct: i for i, acct in enumerate(all_accts)}
num_nodes = len(acct2idx)
print(f"Unique accounts (nodes): {num_nodes:,}")

src_idx = df[SRC_ACCT_COL].map(acct2idx).values
dst_idx = df[DST_ACCT_COL].map(acct2idx).values

edge_index = np.stack([src_idx, dst_idx], axis=0)
num_edges = edge_index.shape[1]
print(f"Total edges: {num_edges:,}")

# ======================================================================
# GRAPH STATISTICS
# ======================================================================

print("\n" + "=" * 70)
print("GRAPH STRUCTURE STATS")
print("=" * 70)

avg_degree = num_edges / num_nodes if num_nodes > 0 else 0.0
density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0

print(f"Nodes:          {num_nodes:,}")
print(f"Edges:          {num_edges:,}")
print(f"Avg. degree:    {avg_degree:.2f}")
print(f"Graph density:  {density:.8f}")

self_mask = src_idx == dst_idx
num_self_loops = int(self_mask.sum())
print(f"Self-loops:     {num_self_loops:,} ({100.0 * num_self_loops / num_edges:.3f}%)")

connected_nodes = set(src_idx) | set(dst_idx)
num_isolated = num_nodes - len(connected_nodes)
print(f"Isolated nodes: {num_isolated:,} ({100.0 * num_isolated / num_nodes:.3f}%)")

# ======================================================================
# TEMPORAL INFO
# ======================================================================

print("\n" + "=" * 70)
print("TEMPORAL INFO")
print("=" * 70)

timestamps = (df[TS_COL].astype("int64") // 10**9).values
time_span_days = (timestamps.max() - timestamps.min()) / (3600 * 24)

print(f"Timestamp range (unix): {timestamps.min()} -> {timestamps.max()}")
print(f"Time span:              {time_span_days:.2f} days")

time_diffs = np.diff(timestamps)
neg_gaps = int((time_diffs < 0).sum())
print(f"Min time gap:           {time_diffs.min()} seconds")
print(f"Max time gap:           {time_diffs.max()} seconds")
print(f"Temporal violations:    {neg_gaps}")

# ======================================================================
# LABELS
# ======================================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION (TARGETS)")
print("=" * 70)

df[LABEL_COL] = df[LABEL_COL].astype(int)
y_edge = df[LABEL_COL].values

num_laund_edges = int(y_edge.sum())
pct_laund_edges = 100.0 * y_edge.mean()

print(f"Laundering edges: {num_laund_edges:,} ({pct_laund_edges:.4f}%)")
print(f"Normal edges:     {num_edges - num_laund_edges:,} ({100.0 - pct_laund_edges:.4f}%)")

y_node = np.zeros(num_nodes, dtype=np.int64)
laund_src = df.loc[df[LABEL_COL] == 1, SRC_ACCT_COL].astype(str)
laund_dst = df.loc[df[LABEL_COL] == 1, DST_ACCT_COL].astype(str)
laund_accts = set(laund_src) | set(laund_dst)
for acct in laund_accts:
    y_node[acct2idx[acct]] = 1

num_laund_nodes = int(y_node.sum())
pct_laund_nodes = 100.0 * y_node.mean()
print(f"Laundering nodes: {num_laund_nodes:,} ({pct_laund_nodes:.4f}%)")
print(f"Normal nodes:     {num_nodes - num_laund_nodes:,} ({100.0 - pct_laund_nodes:.4f}%)")

# ======================================================================
# EDGE FEATURES (ENHANCED WITH TEMPORAL)
# ======================================================================

print("\n" + "=" * 70)
print("EDGE FEATURES (ENHANCED)")
print("=" * 70)

# Basic numeric features
amt_rec = df[AMT_REC_COL].astype(float).values
amt_paid = df[AMT_PAID_COL].astype(float).values

log_amt_rec = log1p_safe(amt_rec)
log_amt_paid = log1p_safe(amt_paid)

# Simple boolean features
same_bank = (df[FROM_BANK_COL].astype(str).values == df[TO_BANK_COL].astype(str).values).astype(np.float32)
same_currency = (df[RCURR_COL].astype(str).values == df[PCURR_COL].astype(str).values).astype(np.float32)

# Temporal features
hour_of_day = df[TS_COL].dt.hour.values.astype(np.float32)
day_of_week = df[TS_COL].dt.dayofweek.values.astype(np.float32)
is_weekend = (day_of_week >= 5).astype(np.float32)

# Normalized timestamps (0 to 1 scale)
ts_normalized = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-10)
ts_normalized = ts_normalized.astype(np.float32)

# Time since last transaction (per account)
print("Computing time since last transaction features...")
time_since_last_src = np.zeros(len(df), dtype=np.float32)
last_src_time = {}
for i, (src, ts) in enumerate(zip(src_idx, timestamps)):
    if src in last_src_time:
        time_since_last_src[i] = float(ts - last_src_time[src])
    else:
        time_since_last_src[i] = 0.0
    last_src_time[src] = ts

time_since_last_dst = np.zeros(len(df), dtype=np.float32)
last_dst_time = {}
for i, (dst, ts) in enumerate(zip(dst_idx, timestamps)):
    if dst in last_dst_time:
        time_since_last_dst[i] = float(ts - last_dst_time[dst])
    else:
        time_since_last_dst[i] = 0.0
    last_dst_time[dst] = ts

# Log-transform time gaps
log_time_since_src = np.log1p(time_since_last_src).astype(np.float32)
log_time_since_dst = np.log1p(time_since_last_dst).astype(np.float32)

print("Time-based features computed")

# Categorical features
pf_dummies = pd.get_dummies(df[PFORMAT_COL].astype(str), prefix="pf")
rc_dummies = pd.get_dummies(df[RCURR_COL].astype(str), prefix="rc")

print(f"Payment formats:  {len(pf_dummies.columns)} distinct")
print(f"Receiving currs:  {len(rc_dummies.columns)} distinct")

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

edge_feat_df = pd.concat([edge_feat_df, pf_dummies, rc_dummies], axis=1)

edge_feature_cols = list(edge_feat_df.columns)
print(f"Total edge feature dims: {len(edge_feature_cols)}")

edge_attr_df = edge_feat_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
edge_attr = edge_attr_df.values.astype(np.float32)

print(f"edge_attr shape: {edge_attr.shape}")

nan_edge = np.isnan(edge_attr).sum()
inf_edge = np.isinf(edge_attr).sum()
if nan_edge > 0 or inf_edge > 0:
    print(f"WARNING: edge_attr still has NaN={nan_edge}, Inf={inf_edge}")
else:
    print(" No NaN/Inf in edge_attr")

edge_attr_means = edge_attr.mean(axis=0)
edge_attr_stds = edge_attr.std(axis=0)
edge_attr_mins = edge_attr.min(axis=0)
edge_attr_maxs = edge_attr.max(axis=0)

# ======================================================================
# NODE FEATURES
# ======================================================================

print("\n" + "=" * 70)
print("NODE FEATURES (SAFE)")
print("=" * 70)

node_df = pd.DataFrame({"acct": list(acct2idx.keys())})
node_df["node_id"] = node_df["acct"].map(acct2idx)

out_deg_series = df.groupby(SRC_ACCT_COL).size()
in_deg_series = df.groupby(DST_ACCT_COL).size()

node_df["out_degree"] = node_df["acct"].map(out_deg_series).fillna(0).astype(float)
node_df["in_degree"] = node_df["acct"].map(in_deg_series).fillna(0).astype(float)
node_df["total_degree"] = node_df["out_degree"] + node_df["in_degree"]

# Log-normalized degrees
node_df["log_out_degree"] = np.log1p(node_df["out_degree"])
node_df["log_in_degree"] = np.log1p(node_df["in_degree"])
node_df["log_total_degree"] = np.log1p(node_df["total_degree"])

# Entity type from accounts
df_acct[ACC_ACCT_COL] = df_acct[ACC_ACCT_COL].astype(str)
df_acct["entity_type"] = df_acct[ACC_ENTITY_NAME_COL].apply(parse_entity_type)

acct_meta = df_acct.set_index(ACC_ACCT_COL)[["entity_type"]]
node_df = node_df.join(acct_meta, on="acct")

node_df["entity_type"] = node_df["entity_type"].fillna("Unknown")
ent_dummies = pd.get_dummies(node_df["entity_type"], prefix="ent")

print(f"Entity types: {list(ent_dummies.columns)}")

node_feat_df = pd.concat([
    node_df[["out_degree", "in_degree", "total_degree",
             "log_out_degree", "log_in_degree", "log_total_degree"]],
    ent_dummies
], axis=1)

x = node_feat_df.replace([np.inf, -np.inf], 0.0).fillna(0.0).values
x = x.astype(np.float32)

print(f"x (node features) shape: {x.shape}")

nan_node = np.isnan(x).sum()
inf_node = np.isinf(x).sum()
if nan_node > 0 or inf_node > 0:
    print(f"WARNING: x still has NaN={nan_node}, Inf={inf_node}")
else:
    print(" No NaN/Inf in x")

# ======================================================================
# SAVE
# ======================================================================

print("\n" + "=" * 70)
print("SAVING TENSORS")
print("=" * 70)
print(f"Output directory: {OUT_DIR}\n")

edge_index_t = torch.tensor(edge_index, dtype=torch.long)
edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)
x_t = torch.tensor(x, dtype=torch.float32)
timestamps_t = torch.tensor(timestamps, dtype=torch.long)
y_edge_t = torch.tensor(y_edge, dtype=torch.long)
y_node_t = torch.tensor(y_node, dtype=torch.long)

torch.save(edge_index_t, os.path.join(OUT_DIR, "edge_index.pt"))
print("  - edge_index.pt")

torch.save(edge_attr_t, os.path.join(OUT_DIR, "edge_attr.pt"))
print("  - edge_attr.pt")

torch.save(x_t, os.path.join(OUT_DIR, "x.pt"))
print("  - x.pt")

torch.save(timestamps_t, os.path.join(OUT_DIR, "timestamps.pt"))
print("  - timestamps.pt")

torch.save(y_edge_t, os.path.join(OUT_DIR, "y_edge.pt"))
print("  - y_edge.pt")

torch.save(y_node_t, os.path.join(OUT_DIR, "y_node.pt"))
print("  - y_node.pt")

with open(os.path.join(OUT_DIR, "node_mapping.json"), "w", encoding="utf-8") as f:
    json.dump(acct2idx, f)
print("  - node_mapping.json")

with open(os.path.join(OUT_DIR, "edge_attr_cols.json"), "w", encoding="utf-8") as f:
    json.dump(edge_feature_cols, f, indent=2)
print("  - edge_attr_cols.json")

feature_stats = {
    "edge_attr_means": edge_attr_means.tolist(),
    "edge_attr_stds": edge_attr_stds.tolist(),
    "edge_attr_mins": edge_attr_mins.tolist(),
    "edge_attr_maxs": edge_attr_maxs.tolist(),
    "edge_attr_cols": edge_feature_cols,
}

with open(os.path.join(OUT_DIR, "feature_stats.json"), "w", encoding="utf-8") as f:
    json.dump(feature_stats, f, indent=2)
print("  - feature_stats.json")

graph_stats = {
    "dataset_type": "Baseline_Enhanced",
    "num_nodes": int(num_nodes),
    "num_edges": int(num_edges),
    "num_self_loops": int(num_self_loops),
    "num_isolated_nodes": int(num_isolated),
    "avg_degree": float(avg_degree),
    "graph_density": float(density),
    "time_span_days": float(time_span_days),
    "num_laundering_edges": int(num_laund_edges),
    "pct_laundering_edges": float(pct_laund_edges),
    "num_laundering_nodes": int(num_laund_nodes),
    "pct_laundering_nodes": float(pct_laund_nodes),
    "num_edge_features": int(edge_attr.shape[1]),
    "num_node_features": int(x.shape[1]),
    "temporal_violations": int(neg_gaps),
    "trans_file": TRANS_FILE,
    "accounts_file": ACCOUNTS_FILE,
    "temporal_features_added": True,
    "time_since_last_added": True,
    "timestamp_normalization_added": True,
}

with open(os.path.join(OUT_DIR, "graph_stats.json"), "w", encoding="utf-8") as f:
    json.dump(graph_stats, f, indent=2)
print("  - graph_stats.json")

print("\n" + "=" * 70)
print(" ENHANCED BASELINE GRAPH READY")
print("=" * 70)
print(f"Saved to: {OUT_DIR}")
print(f"\nEdge features: {edge_attr.shape[1]} (with temporal enhancements)")
print(f"Node features: {x.shape[1]} (with log degrees)")
print("\n Includes: normalized timestamps, time since last transaction")