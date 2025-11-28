"""
dyrep_graph_builder_ibm_baseline.py
-----------------------------------
Build a DyRep-compatible event stream from the IBM HI-Small baseline dataset:

  - Baseline: HI-Small_Trans.csv + HI-Small_accounts.csv

Design goals:
  ✓ Use EXACT same edge features as graph_builder_baseline_static_ENHANCED.py
  ✓ Keep all original edges (no synthetic events, no row changes)
  ✓ Temporal train/val/test split (60/20/20 by time)
  ✓ Event-based representation for DyRep

Outputs (in graphs_dyrep/HI-Small_Trans/):
  - src.pt              [E]
  - dst.pt              [E]
  - ts.pt               [E]   (int64 UNIX seconds)
  - event_type.pt       [E]   (int, derived from Payment Format)
  - edge_attr.pt        [E, F_e]
  - node_features.pt    [N, F_n]
  - labels.pt           [E]   (edge labels)
  - y_node.pt           [N]   (node labels: laundering involvement)
  - train_mask.pt       [E]
  - val_mask.pt         [E]
  - test_mask.pt        [E]
  - edge_attr_cols.json
  - node_mapping.json
  - graph_stats.json
"""

import os
import json
import numpy as np
import pandas as pd
import torch

# ============================================================
# CONFIG
# ============================================================

# Project root directory (one level above ibm_transcations_datasets)
BASE_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection"
DATA_DIR = os.path.join(BASE_DIR, "ibm_transcations_datasets")

TRANS_FILE = "HI-Small_Trans.csv"
ACCOUNTS_FILE = "HI-Small_accounts.csv"

INPUT_TRANS_PATH = os.path.join(DATA_DIR, TRANS_FILE)
INPUT_ACCT_PATH = os.path.join(DATA_DIR, ACCOUNTS_FILE)

DATASET_NAME = "HI-Small_Trans"
OUT_DIR = os.path.join(BASE_DIR, "graphs_dyrep", DATASET_NAME)
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

# Temporal split ratios (by time-ordered edges)
TRAIN_RATIO = 0.6
VAL_RATIO = 0.20  # test will be 1 - TRAIN_RATIO - VAL_RATIO

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def log1p_safe(x):
    """Safe log1p on numpy array."""
    x = np.asarray(x, dtype=np.float64)
    x = np.where(x < 0, 0.0, x)
    return np.log1p(x)

def parse_entity_type(entity_name: str) -> str:
    """Extract entity type from 'Entity Name' (same logic as baseline builder)."""
    if not isinstance(entity_name, str):
        return "Unknown"
    if "#" in entity_name:
        left = entity_name.split("#")[0].strip()
        return left
    parts = entity_name.split()
    if len(parts) <= 1:
        return entity_name.strip()
    return " ".join(parts[:-1]).strip()

# ============================================================
# LOAD CSVs
# ============================================================

print("=" * 70)
print("DYREP BASELINE EVENT-BASED GRAPH BUILDER")
print("(matching features of baseline static builder)")
print("=" * 70)

print(f"\nLoading transactions from: {INPUT_TRANS_PATH}")
df = pd.read_csv(INPUT_TRANS_PATH, low_memory=False)
print(f"  Loaded {len(df):,} transaction rows")

print(f"\nLoading accounts from: {INPUT_ACCT_PATH}")
df_acct = pd.read_csv(INPUT_ACCT_PATH, low_memory=False)
print(f"  Loaded {len(df_acct):,} account rows")

# Parse & sort timestamps
df[TS_COL] = pd.to_datetime(df[TS_COL], errors="raise")
df = df.sort_values(TS_COL).reset_index(drop=True)
print(f"\nTimestamp range: {df[TS_COL].min()} -> {df[TS_COL].max()}")

# ============================================================
# NODE MAPPING
# ============================================================

print("\n" + "=" * 70)
print("NODE MAPPING")
print("=" * 70)

df[SRC_ACCT_COL] = df[SRC_ACCT_COL].astype(str)
df[DST_ACCT_COL] = df[DST_ACCT_COL].astype(str)

all_accts = pd.concat([df[SRC_ACCT_COL], df[DST_ACCT_COL]], axis=0).unique()
acct2idx = {acct: i for i, acct in enumerate(all_accts)}
num_nodes = len(acct2idx)
print(f"Unique accounts (nodes): {num_nodes:,}")

src = df[SRC_ACCT_COL].map(acct2idx).values.astype(np.int64)
dst = df[DST_ACCT_COL].map(acct2idx).values.astype(np.int64)
num_edges = len(src)
print(f"Total edges: {num_edges:,}")

# ============================================================
# TEMPORAL INFO
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL INFO")
print("=" * 70)

timestamps = (df[TS_COL].astype("int64") // 10**9).values.astype(np.int64)
time_span_days = (timestamps.max() - timestamps.min()) / (3600 * 24)

print(f"Timestamp range (unix): {timestamps.min()} -> {timestamps.max()}")
print(f"Time span:              {time_span_days:.2f} days")

time_diffs = np.diff(timestamps)
neg_gaps = int((time_diffs < 0).sum())
print(f"Min time gap:           {time_diffs.min()} seconds" if len(time_diffs) > 0 else "Single timestamp")
print(f"Max time gap:           {time_diffs.max()} seconds" if len(time_diffs) > 0 else "Single timestamp")
print(f"Temporal violations:    {neg_gaps}")

# ============================================================
# LABELS
# ============================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION (TARGETS)")
print("=" * 70)

df[LABEL_COL] = df[LABEL_COL].astype(int)
y_edge = df[LABEL_COL].values

num_laund_edges = int(y_edge.sum())
pct_laund_edges = 100.0 * y_edge.mean()

print(f"Laundering edges: {num_laund_edges:,} ({pct_laund_edges:.4f}%)")
print(f"Normal edges:     {num_edges - num_laund_edges:,} ({100.0 - pct_laund_edges:.4f}%)")

# Node labels: any account involved in laundering becomes 1
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

# ============================================================
# EDGE FEATURES (MATCHING ENHANCED BASELINE)
# ============================================================

print("\n" + "=" * 70)
print("EDGE FEATURES (BASELINE ENHANCED)")
print("=" * 70)

amt_rec = df[AMT_REC_COL].astype(float).values
amt_paid = df[AMT_PAID_COL].astype(float).values

log_amt_rec = log1p_safe(amt_rec)
log_amt_paid = log1p_safe(amt_paid)

same_bank = (df[FROM_BANK_COL].astype(str).values == df[TO_BANK_COL].astype(str).values).astype(np.float32)
same_currency = (df[RCURR_COL].astype(str).values == df[PCURR_COL].astype(str).values).astype(np.float32)

hour_of_day = df[TS_COL].dt.hour.values.astype(np.float32)
day_of_week = df[TS_COL].dt.dayofweek.values.astype(np.float32)
is_weekend = (day_of_week >= 5).astype(np.float32)

ts_normalized = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-10)
ts_normalized = ts_normalized.astype(np.float32)

# Time since last transaction (per src)
print("Computing time since last transaction features...")
time_since_last_src = np.zeros(num_edges, dtype=np.float32)
last_src_time = {}
for i, (s, ts) in enumerate(zip(src, timestamps)):
    if s in last_src_time:
        time_since_last_src[i] = float(ts - last_src_time[s])
    else:
        time_since_last_src[i] = 0.0
    last_src_time[s] = ts

# Time since last transaction (per dst)
time_since_last_dst = np.zeros(num_edges, dtype=np.float32)
last_dst_time = {}
for i, (d, ts) in enumerate(zip(dst, timestamps)):
    if d in last_dst_time:
        time_since_last_dst[i] = float(ts - last_dst_time[d])
    else:
        time_since_last_dst[i] = 0.0
    last_dst_time[d] = ts

log_time_since_src = np.log1p(time_since_last_src).astype(np.float32)
log_time_since_dst = np.log1p(time_since_last_dst).astype(np.float32)

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

edge_attr_cols = list(edge_feat_df.columns)
print(f"Total edge feature dims: {len(edge_attr_cols)}")

edge_attr_df = edge_feat_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
edge_attr = edge_attr_df.values.astype(np.float32)

print(f"edge_attr shape: {edge_attr.shape}")

nan_edge = np.isnan(edge_attr).sum()
inf_edge = np.isinf(edge_attr).sum()
if nan_edge > 0 or inf_edge > 0:
    print(f"WARNING: edge_attr still has NaN={nan_edge}, Inf={inf_edge}")
else:
    print(" No NaN/Inf in edge_attr")

# ============================================================
# NODE FEATURES (MATCHING BASELINE BUILDER)
# ============================================================

print("\n" + "=" * 70)
print("NODE FEATURES")
print("=" * 70)

node_df = pd.DataFrame({"acct": list(acct2idx.keys())})
node_df["node_id"] = node_df["acct"].map(acct2idx)

out_deg_series = df.groupby(SRC_ACCT_COL).size()
in_deg_series = df.groupby(DST_ACCT_COL).size()

node_df["out_degree"] = node_df["acct"].map(out_deg_series).fillna(0).astype(float)
node_df["in_degree"] = node_df["acct"].map(in_deg_series).fillna(0).astype(float)
node_df["total_degree"] = node_df["out_degree"] + node_df["in_degree"]

node_df["log_out_degree"] = np.log1p(node_df["out_degree"])
node_df["log_in_degree"] = np.log1p(node_df["in_degree"])
node_df["log_total_degree"] = np.log1p(node_df["total_degree"])

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

node_features = node_feat_df.replace([np.inf, -np.inf], 0.0).fillna(0.0).values.astype(np.float32)

print(f"node_features shape: {node_features.shape}")

nan_node = np.isnan(node_features).sum()
inf_node = np.isinf(node_features).sum()
if nan_node > 0 or inf_node > 0:
    print(f"WARNING: node_features still has NaN={nan_node}, Inf={inf_node}")
else:
    print(" No NaN/Inf in node_features")

# ============================================================
# EVENT TYPES (DyRep)
# ============================================================

# Use Payment Format as event types (same design as motif DyRep builder)
event_type_map = {pf: i for i, pf in enumerate(df[PFORMAT_COL].astype(str).unique())}
event_type = df[PFORMAT_COL].astype(str).map(event_type_map).values.astype(np.int64)

print(f"\nEvent types: {len(event_type_map)} unique types")
print(f"  Types: {list(event_type_map.keys())}")

# ============================================================
# TEMPORAL TRAIN/VAL/TEST SPLIT (by time order)
# ============================================================

train_end = int(num_edges * TRAIN_RATIO)
val_end = int(num_edges * (TRAIN_RATIO + VAL_RATIO))
test_end = num_edges  # sanity

train_mask = np.zeros(num_edges, dtype=bool)
val_mask = np.zeros(num_edges, dtype=bool)
test_mask = np.zeros(num_edges, dtype=bool)

train_mask[:train_end] = True
val_mask[train_end:val_end] = True
test_mask[val_end:test_end] = True

print("\nSplit edges (by time):")
print(f"  Train: {train_mask.sum():,}")
print(f"  Val:   {val_mask.sum():,}")
print(f"  Test:  {test_mask.sum():,}")

# ============================================================
# VALIDATE TEMPORAL ORDERING (critical for DyRep)
# ============================================================

assert np.all(np.diff(timestamps) >= 0), "Timestamps must be non-decreasing for DyRep!"

train_times = timestamps[train_mask]
val_times = timestamps[val_mask]
test_times = timestamps[test_mask]

print("\nTemporal split validation:")
print(f"  Train time: {train_times.min()} → {train_times.max()}")
print(f"  Val time:   {val_times.min()} → {val_times.max()}")
print(f"  Test time:  {test_times.min()} → {test_times.max()}")

assert train_times.max() <= val_times.min(), "Train/Val temporal overlap!"
assert val_times.max() <= test_times.min(), "Val/Test temporal overlap!"
print("  ✓ No temporal leakage")

# ============================================================
# SAVE EVERYTHING (DyRep format)
# ============================================================

torch.save(torch.tensor(src, dtype=torch.long), os.path.join(OUT_DIR, "src.pt"))
torch.save(torch.tensor(dst, dtype=torch.long), os.path.join(OUT_DIR, "dst.pt"))
torch.save(torch.tensor(timestamps, dtype=torch.long), os.path.join(OUT_DIR, "ts.pt"))
torch.save(torch.tensor(event_type, dtype=torch.long), os.path.join(OUT_DIR, "event_type.pt"))

torch.save(torch.tensor(edge_attr, dtype=torch.float32), os.path.join(OUT_DIR, "edge_attr.pt"))
torch.save(torch.tensor(node_features, dtype=torch.float32), os.path.join(OUT_DIR, "node_features.pt"))

torch.save(torch.tensor(y_edge, dtype=torch.long), os.path.join(OUT_DIR, "labels.pt"))
torch.save(torch.tensor(y_node, dtype=torch.long), os.path.join(OUT_DIR, "y_node.pt"))

torch.save(torch.tensor(train_mask, dtype=torch.bool), os.path.join(OUT_DIR, "train_mask.pt"))
torch.save(torch.tensor(val_mask, dtype=torch.bool), os.path.join(OUT_DIR, "val_mask.pt"))
torch.save(torch.tensor(test_mask, dtype=torch.bool), os.path.join(OUT_DIR, "test_mask.pt"))

with open(os.path.join(OUT_DIR, "edge_attr_cols.json"), "w", encoding="utf-8") as f:
    json.dump(edge_attr_cols, f, indent=2)

with open(os.path.join(OUT_DIR, "node_mapping.json"), "w", encoding="utf-8") as f:
    json.dump(acct2idx, f)

graph_stats = {
    "dataset_name": DATASET_NAME,
    "dataset_type": "Baseline_Enhanced_DyRep",
    "num_nodes": int(num_nodes),
    "num_edges": int(num_edges),
    "time_span_days": float(time_span_days),
    "temporal_violations": int(neg_gaps),
    "num_laundering_edges": int(num_laund_edges),
    "pct_laundering_edges": float(pct_laund_edges),
    "num_laundering_nodes": int(num_laund_nodes),
    "pct_laundering_nodes": float(pct_laund_nodes),
    "num_edge_features": int(edge_attr.shape[1]),
    "num_node_features": int(node_features.shape[1]),
    "train_ratio": TRAIN_RATIO,
    "val_ratio": VAL_RATIO,
    "test_ratio": 1.0 - TRAIN_RATIO - VAL_RATIO,
    "trans_file": TRANS_FILE,
    "accounts_file": ACCOUNTS_FILE,
}

with open(os.path.join(OUT_DIR, "graph_stats.json"), "w", encoding="utf-8") as f:
    json.dump(graph_stats, f, indent=2)

print("\n" + "=" * 70)
print("✓ DYREP BASELINE EVENT GRAPH BUILT")
print("=" * 70)
print(f"Saved to: {OUT_DIR}")
print(f"\nEdge features: {edge_attr.shape[1]} (baseline enhanced)")
print(f"Node features: {node_features.shape[1]} (degrees + entity type)")
