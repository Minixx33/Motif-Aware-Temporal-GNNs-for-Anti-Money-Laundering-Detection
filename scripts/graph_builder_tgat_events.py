"""
graph_builder_tgat_events.py (Final Universal Version)
------------------------------------------------------
Build a temporal event stream for TGAT from IBM HI-Small datasets.

Supports:
  - Baseline: HI-Small_Trans.csv
  - RAT:     HI-Small_Trans_RAT_low/medium/high.csv
  - SLT:     HI-Small_Trans_SLT_low/medium/high.csv
  - STRAIN:  HI-Small_Trans_STRAIN_low/medium/high.csv

Outputs (tgat_graphs/{dataset_name}/):
  - src_nodes.pt
  - dst_nodes.pt
  - timestamps.pt
  - edge_attr.pt
  - y_edge.pt
  - x_node.pt
  - y_node.pt
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

# ============================================================
# CONFIG - SELECT DATASET HERE
# ============================================================

BASE_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets"

# Baseline
# DATASET = "HI-Small_Trans.csv"

# RAT examples:
# DATASET = os.path.join("RAT", "HI-Small_Trans_RAT_low.csv")
# DATASET = os.path.join("RAT", "HI-Small_Trans_RAT_medium.csv")
DATASET = os.path.join("RAT", "HI-Small_Trans_RAT_high.csv")

# SLT examples:
# DATASET = os.path.join("SLT", "HI-Small_Trans_SLT_low.csv")
# DATASET = os.path.join("SLT", "HI-Small_Trans_SLT_medium.csv")
# DATASET = os.path.join("SLT", "HI-Small_Trans_SLT_high.csv")

# STRAIN examples:
# DATASET = os.path.join("STRAIN", "HI-Small_Trans_STRAIN_low.csv")
# DATASET = os.path.join("STRAIN", "HI-Small_Trans_STRAIN_medium.csv")
# DATASET = os.path.join("STRAIN", "HI-Small_Trans_STRAIN_high.csv")

INPUT_PATH = os.path.join(BASE_DIR, DATASET)
dataset_name = os.path.splitext(os.path.basename(DATASET))[0]

OUT_DIR = os.path.join(BASE_DIR, "tgat_graphs", dataset_name)
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# COLUMN DEFINITIONS
# ============================================================

SRC_COL = "Account"
DST_COL = "Account.1"
TS_COL = "Timestamp"
LABEL_COL = "Is Laundering"

EXCLUDE_COLS = {
    SRC_COL, DST_COL, TS_COL, LABEL_COL,
    "From Bank", "To Bank",
    "Receiving Currency", "Payment Currency",
    "Payment Format",
    "date_only", "hour", "weekday",
    "RAT_injected", "RAT_intensity_level",
    "SLT_injected", "SLT_intensity_level",
    "STRAIN_injected", "STRAIN_intensity_level",
}

WHITELIST_STRUCT_COLS = {
    "src_out_degree", "dst_in_degree",
    "src_amt_mean", "src_amt_std",
    "dst_amt_mean", "dst_amt_std",
    "src_age_days", "dst_age_days",
    "src_day_tx_count", "dst_day_tx_count",
    "dst_out_degree", "dst_out_deg_norm",
}

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("UNIVERSAL TGAT EVENT STREAM BUILDER")
print("=" * 70)
print(f"Dataset: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Loaded {len(df):,} rows")

df[SRC_COL] = df[SRC_COL].astype(str)
df[DST_COL] = df[DST_COL].astype(str)

df[TS_COL] = pd.to_datetime(df[TS_COL], errors="raise")
df = df.sort_values(TS_COL).reset_index(drop=True)

print(f"Timestamp range: {df[TS_COL].min()} -> {df[TS_COL].max()}")

# ============================================================
# THEORY DETECTION
# ============================================================

has_rat = any(c.startswith("RAT_") for c in df.columns)
has_motif = any(c.startswith("motif_") for c in df.columns)
has_slt = any(c.startswith("SLT_") for c in df.columns)
has_strain = any(c.startswith("STRAIN_") for c in df.columns)

if has_rat or has_motif:
    dataset_type = "RAT-injected"
    theory_prefix = ["RAT_", "motif_"]
elif has_slt:
    dataset_type = "SLT-injected"
    theory_prefix = ["SLT_"]
elif has_strain:
    dataset_type = "STRAIN-injected"
    theory_prefix = ["STRAIN_"]
else:
    dataset_type = "Baseline"
    theory_prefix = []

print(f"Detected dataset type: {dataset_type}")

# ============================================================
# NODE MAPPING
# ============================================================

all_accounts = pd.concat([df[SRC_COL], df[DST_COL]]).unique()
acct2idx = {acct: i for i, acct in enumerate(all_accounts)}

num_nodes = len(acct2idx)
print(f"Num nodes: {num_nodes:,}")

src_nodes = df[SRC_COL].map(acct2idx).values
dst_nodes = df[DST_COL].map(acct2idx).values
num_events = len(df)

print(f"Num events: {num_events:,}")

# ============================================================
# TEMPORAL VALIDATION
# ============================================================

timestamps = (df[TS_COL].astype("int64") // 10**9).values
time_diffs = np.diff(timestamps)

if (time_diffs < 0).any():
    raise ValueError("❌ TGAT requires strictly increasing timestamps — dataset not sorted!")

unique_ts = len(np.unique(timestamps))
print(f"Unique timestamps: {unique_ts:,} / {num_events:,}")

# ============================================================
# EDGE LABELS
# ============================================================

y_edge = df[LABEL_COL].astype(int).values

# ============================================================
# EDGE FEATURES
# ============================================================

edge_feature_cols = []
for col in df.columns:
    if col in EXCLUDE_COLS:
        continue
    if not np.issubdtype(df[col].dtype, np.number):
        continue
    if col in WHITELIST_STRUCT_COLS:
        edge_feature_cols.append(col)
    if any(col.startswith(p) for p in theory_prefix):
        edge_feature_cols.append(col)

edge_feature_cols = sorted(edge_feature_cols)

edge_attr_df = df[edge_feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

for col in edge_feature_cols:
    edge_attr_df[col] = edge_attr_df[col].astype(np.float32)

edge_attr = edge_attr_df.values

# Statistics for later (not applied here)
edge_attr_means = edge_attr.mean(axis=0)
edge_attr_stds  = edge_attr.std(axis=0)
edge_attr_mins  = edge_attr.min(axis=0)
edge_attr_maxs  = edge_attr.max(axis=0)

# ============================================================
# NODE FEATURES
# ============================================================

node_df = pd.DataFrame({"acct": list(acct2idx.keys())})
node_df["node_id"] = node_df["acct"].map(acct2idx)

out_deg = df.groupby(SRC_COL).size()
in_deg  = df.groupby(DST_COL).size()

node_df["out_degree"] = node_df["acct"].map(out_deg).fillna(0).astype(float)
node_df["in_degree"]  = node_df["acct"].map(in_deg).fillna(0).astype(float)
node_df["total_degree"] = node_df["out_degree"] + node_df["in_degree"]

laund_src = df[df[LABEL_COL] == 1][SRC_COL]
laund_dst = df[df[LABEL_COL] == 1][DST_COL]

laund_counts = (
    pd.concat([laund_src, laund_dst])
    .value_counts()
    .reindex(node_df["acct"])
    .fillna(0)
    .astype(float)
)

node_df["laundering_count"] = laund_counts.values

x_node = node_df[["out_degree", "in_degree", "total_degree", "laundering_count"]].values.astype(np.float32)

# ============================================================
# NODE LABELS
# ============================================================

y_node = np.zeros(num_nodes, dtype=np.int64)
laund_accts = set(laund_src) | set(laund_dst)

for acct in laund_accts:
    y_node[acct2idx[acct]] = 1

# ============================================================
# SAVE ALL ARTIFACTS
# ============================================================

torch.save(torch.tensor(src_nodes),     os.path.join(OUT_DIR, "src_nodes.pt"))
torch.save(torch.tensor(dst_nodes),     os.path.join(OUT_DIR, "dst_nodes.pt"))
torch.save(torch.tensor(timestamps),    os.path.join(OUT_DIR, "timestamps.pt"))
torch.save(torch.tensor(edge_attr),     os.path.join(OUT_DIR, "edge_attr.pt"))
torch.save(torch.tensor(y_edge),        os.path.join(OUT_DIR, "y_edge.pt"))
torch.save(torch.tensor(x_node),        os.path.join(OUT_DIR, "x_node.pt"))
torch.save(torch.tensor(y_node),        os.path.join(OUT_DIR, "y_node.pt"))

with open(os.path.join(OUT_DIR, "node_mapping.json"), "w") as f:
    json.dump(acct2idx, f)

with open(os.path.join(OUT_DIR, "edge_attr_cols.json"), "w") as f:
    json.dump(edge_feature_cols, f, indent=2)

with open(os.path.join(OUT_DIR, "feature_stats.json"), "w") as f:
    json.dump({
        "edge_attr_means": edge_attr_means.tolist(),
        "edge_attr_stds":  edge_attr_stds.tolist(),
        "edge_attr_mins":  edge_attr_mins.tolist(),
        "edge_attr_maxs":  edge_attr_maxs.tolist(),
        "edge_attr_cols":  edge_feature_cols,
    }, f, indent=2)

with open(os.path.join(OUT_DIR, "graph_stats.json"), "w") as f:
    json.dump({
        "dataset_type": dataset_type,
        "num_nodes": int(num_nodes),
        "num_events": int(num_events),
        "unique_timestamps": int(unique_ts),
        "num_edge_features": len(edge_feature_cols),
        "num_node_features": x_node.shape[1],
        "pct_laundering_edges": float(y_edge.mean() * 100),
        "dataset_name": dataset_name,
        "format": "TGAT_event_stream"
    }, f, indent=2)

print("\nTGAT EVENT STREAM CONSTRUCTION COMPLETE")
print(f"Saved to: {OUT_DIR}")