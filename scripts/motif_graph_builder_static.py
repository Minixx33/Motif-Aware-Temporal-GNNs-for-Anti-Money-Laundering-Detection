"""
graph_builder_static.py (Final Universal Version)
-------------------------------------------------
Build a static graph for GraphSAGE / GraphSAGE-T from IBM HI-Small datasets.

Supports ALL criminology theory datasets:
    - HI-Small_Trans_RAT_low/medium/high.csv (RAT)
    - HI-Small_Trans_SLT_low/medium/high.csv (SLT)
    - HI-Small_Trans_STRAIN_low/medium/high.csv (Strain)

Features:
  ✓ Automatic dataset type detection
  ✓ Theory-specific feature extraction (RAT_, SLT_, STRAIN_, motif_)
  ✓ Separate output directories per dataset
  ✓ Comprehensive validation and statistics
  ✓ NaN/Inf handling and memory optimization
  ✓ Motif feature normalization for GraphSAGE-T stability

Outputs:
    edge_index.pt, edge_attr.pt, timestamps.pt,
    x.pt, y_edge.pt, y_node.pt, metadata JSONs
"""

import os
import json
import numpy as np
import pandas as pd
import torch

# ============================================================
# CONFIG - CHANGE DATASET HERE
# ============================================================
BASE_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection"
DATA_DIR = os.path.join(BASE_DIR, "ibm_transcations_datasets")
# Choose dataset here:
# DATASET = os.path.join("RAT", "HI-Small_Trans_RAT_low.csv")
# DATASET = os.path.join("RAT", "HI-Small_Trans_RAT_medium.csv")
DATASET = os.path.join("RAT", "HI-Small_Trans_RAT_high.csv")

INPUT_PATH = os.path.join(DATA_DIR, DATASET)
dataset_name = os.path.splitext(os.path.basename(DATASET))[0]
OUT_DIR = os.path.join(BASE_DIR, "graphs", dataset_name)
os.makedirs(OUT_DIR, exist_ok=True)

SRC_COL = "Account"
DST_COL = "Account.1"
TS_COL = "Timestamp"
LABEL_COL = "Is Laundering"

EXCLUDE_COLS = {
    SRC_COL, DST_COL, TS_COL, LABEL_COL,
    "From Bank", "To Bank",
    "Receiving Currency", "Payment Currency", "Payment Format",
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

REMOVE_SELF_LOOPS = False

# ============================================================
# LOAD DATA
# ============================================================

print("="*70)
print("UNIVERSAL GRAPH BUILDER FOR GRAPHSAGE / GRAPHSAGE-T")
print("="*70)

print(f"\nDataset: {DATASET}")
df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"✓ Loaded {len(df):,} rows")

df[SRC_COL] = df[SRC_COL].astype(str)
df[DST_COL] = df[DST_COL].astype(str)
df[TS_COL] = pd.to_datetime(df[TS_COL], errors="raise")

df = df.sort_values(TS_COL).reset_index(drop=True)
print(f"✓ Timestamp range: {df[TS_COL].min()} → {df[TS_COL].max()}")

# ============================================================
# DATASET TYPE DETECTION
# ============================================================

has_rat = any(c.startswith("RAT_") for c in df.columns)
has_motif = any(c.startswith("motif_") for c in df.columns)
has_slt = any(c.startswith("SLT_") for c in df.columns)
has_str = any(c.startswith("STRAIN_") for c in df.columns)

if has_rat or has_motif:
    dataset_type = "RAT-injected"
    theory_prefix = ["RAT_", "motif_"]
elif has_slt:
    dataset_type = "SLT-injected"
    theory_prefix = ["SLT_"]
elif has_str:
    dataset_type = "Strain-injected"
    theory_prefix = ["STRAIN_"]
else:
    dataset_type = "Baseline"
    theory_prefix = []

print(f"\nDetected: {dataset_type}")

# ============================================================
# NODE MAPPING
# ============================================================

all_accounts = pd.concat([df[SRC_COL], df[DST_COL]]).unique()
acct2idx = {acct: i for i, acct in enumerate(all_accounts)}
num_nodes = len(acct2idx)

src_idx = df[SRC_COL].map(acct2idx).values
dst_idx = df[DST_COL].map(acct2idx).values
edge_index = np.stack([src_idx, dst_idx], axis=0)
num_edges = edge_index.shape[1]

print(f"\nNodes: {num_nodes:,}")
print(f"Edges: {num_edges:,}")

# ============================================================
# TEMPORAL
# ============================================================

timestamps = (df[TS_COL].astype("int64") // 1_000_000_000).values
time_diffs = np.diff(timestamps)
num_negative_gaps = (time_diffs < 0).sum()

# ============================================================
# LABELS
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

    is_struct = col in WHITELIST_STRUCT_COLS
    is_theory = any(col.startswith(p) for p in theory_prefix)

    if is_struct or is_theory:
        edge_feature_cols.append(col)

edge_feature_cols = sorted(edge_feature_cols)
edge_attr_df = df[edge_feature_cols].copy()

# ============================================================
# 1. CLEAN FIRST (NaN/Inf)
# ============================================================

edge_attr_df = edge_attr_df.fillna(0).replace([np.inf, -np.inf], 0)

# ============================================================
# 2. NORMALIZE MOTIF COLUMNS (critical)
# ============================================================

motif_cols = [c for c in edge_feature_cols if c.startswith("motif_")]

if len(motif_cols) > 0:
    print(f"\nNormalizing {len(motif_cols)} motif feature columns...")

    for col in motif_cols:
        arr = edge_attr_df[col].values.astype(np.float32)
        mean, std = arr.mean(), arr.std()

        if std < 1e-6:
            std = 1.0

        arr = (arr - mean) / std
        arr = np.clip(arr, -10, 10)

        edge_attr_df[col] = arr

    print("✓ Motif features normalized + clipped")
else:
    print("\nNo motif features detected — skipping normalization.")

# ============================================================
# Convert to numpy & validate
# ============================================================

edge_attr = edge_attr_df.values.astype(np.float32)

nan_after = np.isnan(edge_attr).sum()
inf_after = np.isinf(edge_attr).sum()

if nan_after > 0 or inf_after > 0:
    print(f"✗ ERROR: Found NaNs={nan_after}, Infs={inf_after}")
    raise ValueError("Normalization failed")
else:
    print("✓ Edge features are clean")

edge_attr_means = edge_attr.mean(axis=0)
edge_attr_stds = edge_attr.std(axis=0)
edge_attr_mins = edge_attr.min(axis=0)
edge_attr_maxs = edge_attr.max(axis=0)

# ============================================================
# NODE FEATURES
# ============================================================

node_df = pd.DataFrame({"acct": list(acct2idx.keys())})
node_df["node_id"] = node_df["acct"].map(acct2idx)

out_deg = df.groupby(SRC_COL).size()
in_deg = df.groupby(DST_COL).size()

node_df["out_degree"] = node_df["acct"].map(out_deg).fillna(0)
node_df["in_degree"] = node_df["acct"].map(in_deg).fillna(0)
node_df["total_degree"] = node_df["out_degree"] + node_df["in_degree"]

laund_src = df[df[LABEL_COL] == 1][SRC_COL]
laund_dst = df[df[LABEL_COL] == 1][DST_COL]
laund_counts = pd.concat([laund_src, laund_dst]).value_counts()

node_df["laundering_count"] = node_df["acct"].map(laund_counts).fillna(0)

x = node_df[["out_degree", "in_degree", "total_degree", "laundering_count"]].values.astype(np.float32)

# ============================================================
# NODE LABELS
# ============================================================

y_node = np.zeros(num_nodes, dtype=np.int64)
for acct in laund_counts.index:
    y_node[acct2idx[acct]] = 1

# ============================================================
# SAVE TENSORS
# ============================================================

torch.save(torch.tensor(edge_index, dtype=torch.long), os.path.join(OUT_DIR, "edge_index.pt"))
torch.save(torch.tensor(edge_attr, dtype=torch.float32), os.path.join(OUT_DIR, "edge_attr.pt"))
torch.save(torch.tensor(x, dtype=torch.float32), os.path.join(OUT_DIR, "x.pt"))
torch.save(torch.tensor(timestamps, dtype=torch.long), os.path.join(OUT_DIR, "timestamps.pt"))
torch.save(torch.tensor(y_edge, dtype=torch.long), os.path.join(OUT_DIR, "y_edge.pt"))
torch.save(torch.tensor(y_node, dtype=torch.long), os.path.join(OUT_DIR, "y_node.pt"))

with open(os.path.join(OUT_DIR, "node_mapping.json"), "w") as f:
    json.dump(acct2idx, f)

with open(os.path.join(OUT_DIR, "edge_attr_cols.json"), "w") as f:
    json.dump(edge_feature_cols, f, indent=2)

feature_stats = {
    "edge_attr_means": edge_attr_means.tolist(),
    "edge_attr_stds": edge_attr_stds.tolist(),
    "edge_attr_mins": edge_attr_mins.tolist(),
    "edge_attr_maxs": edge_attr_maxs.tolist(),
}
with open(os.path.join(OUT_DIR, "feature_stats.json"), "w") as f:
    json.dump(feature_stats, f, indent=2)

graph_stats = {
    "dataset_type": dataset_type,
    "num_nodes": int(num_nodes),
    "num_edges": int(num_edges),
    "num_edge_features": len(edge_feature_cols),
    "num_node_features": x.shape[1],
    "temporal_violations": int(num_negative_gaps),
}
with open(os.path.join(OUT_DIR, "graph_stats.json"), "w") as f:
    json.dump(graph_stats, f, indent=2)

print("\n✓ GRAPH CONSTRUCTION COMPLETE")
print("="*70)
print(f"Saved to: {OUT_DIR}")
