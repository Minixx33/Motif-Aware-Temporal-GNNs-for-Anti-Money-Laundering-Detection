"""
graph_builder_static.py (Final Universal Version)
-------------------------------------------------
Build a static graph for GraphSAGE / GraphSAGE-T from IBM HI-Small datasets.

Supports ALL criminology theory datasets:
    - HI-Small_Trans.csv                     (Baseline)
    - HI-Small_Trans_RAT_low/medium/high.csv (Routine Activity Theory)
    - HI-Small_Trans_SLT_low/medium/high.csv (Social Learning Theory)
    - HI-Small_Trans_STRAIN_low/medium/high.csv (Strain Theory)

Features:
  ✓ Automatic dataset type detection
  ✓ Theory-specific feature extraction (RAT_, SLT_, STRAIN_, motif_)
  ✓ Separate output directories per dataset
  ✓ Comprehensive validation and statistics
  ✓ NaN/Inf handling and memory optimization
  ✓ GraphSAGE-T temporal validation

Outputs (in graphs/{dataset_name}/):
  - edge_index.pt      [2, E]
  - edge_attr.pt       [E, F_e]
  - timestamps.pt      [E]
  - x.pt               [N, F_n]
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

# ============================================================
# CONFIG - CHANGE DATASET HERE
# ============================================================

BASE_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets"

# ========== SELECT ONE DATASET ==========
# Baseline:
# DATASET = "HI-Small_Trans.csv"

# RAT (Routine Activity Theory):
DATASET = os.path.join("RAT", "HI-Small_Trans_RAT_low.csv")
# DATASET = os.path.join("RAT", "HI-Small_Trans_RAT_medium.csv")
# DATASET = os.path.join("RAT", "HI-Small_Trans_RAT_high.csv")

# SLT (Social Learning Theory):
# DATASET = os.path.join("SLT", "HI-Small_Trans_SLT_low.csv")
# DATASET = os.path.join("SLT", "HI-Small_Trans_SLT_medium.csv")
# DATASET = os.path.join("SLT", "HI-Small_Trans_SLT_high.csv")

# STRAIN (Strain Theory):
# DATASET = os.path.join("STRAIN", "HI-Small_Trans_STRAIN_low.csv")
# DATASET = os.path.join("STRAIN", "HI-Small_Trans_STRAIN_medium.csv")
# DATASET = os.path.join("STRAIN", "HI-Small_Trans_STRAIN_high.csv")

# ========== AUTO-GENERATE OUTPUT PATH ==========
INPUT_PATH = os.path.join(BASE_DIR, DATASET)
dataset_name = os.path.splitext(os.path.basename(DATASET))[0]
OUT_DIR = os.path.join(BASE_DIR, "graphs", dataset_name)
os.makedirs(OUT_DIR, exist_ok=True)

# ========== COLUMN NAMES ==========
SRC_COL = "Account"
DST_COL = "Account.1"
TS_COL = "Timestamp"
LABEL_COL = "Is Laundering"

# Columns to EXCLUDE from edge features
EXCLUDE_COLS = {
    SRC_COL, DST_COL, TS_COL, LABEL_COL,
    "From Bank", "To Bank",
    "Receiving Currency", "Payment Currency", "Payment Format",
    "date_only", "hour", "weekday",
    "RAT_injected", "RAT_intensity_level",
    "SLT_injected", "SLT_intensity_level",
    "STRAIN_injected", "STRAIN_intensity_level",
}

# Structural columns to INCLUDE (if present)
WHITELIST_STRUCT_COLS = {
    "src_out_degree", "dst_in_degree",
    "src_amt_mean", "src_amt_std",
    "dst_amt_mean", "dst_amt_std",
    "src_age_days", "dst_age_days",
    "src_day_tx_count", "dst_day_tx_count",
    "dst_out_degree", "dst_out_deg_norm",
}

# Self-loop handling
REMOVE_SELF_LOOPS = False

# ============================================================
# LOAD DATA
# ============================================================

print("="*70)
print("UNIVERSAL GRAPH BUILDER FOR GRAPHSAGE / GRAPHSAGE-T")
print("="*70)
print(f"\nDataset: {DATASET}")
print(f"Loading from: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"✓ Loaded {len(df):,} transaction rows")

# Ensure correct types
df[SRC_COL] = df[SRC_COL].astype(str)
df[DST_COL] = df[DST_COL].astype(str)
df[TS_COL] = pd.to_datetime(df[TS_COL], errors="raise")

# Sort by time (critical for GraphSAGE-T)
df = df.sort_values(TS_COL).reset_index(drop=True)
print(f"✓ Timestamp range: {df[TS_COL].min()} → {df[TS_COL].max()}")

# ============================================================
# DATASET TYPE DETECTION
# ============================================================

print("\n" + "="*70)
print("DATASET TYPE DETECTION")
print("="*70)

has_rat_features = any(col.startswith("RAT_") for col in df.columns)
has_motif_features = any(col.startswith("motif_") for col in df.columns)
has_slt_features = any(col.startswith("SLT_") for col in df.columns)
has_strain_features = any(col.startswith("STRAIN_") for col in df.columns)

if has_rat_features or has_motif_features:
    dataset_type = "RAT-injected"
    theory_prefix = ["RAT_", "motif_"]
elif has_slt_features:
    dataset_type = "SLT-injected"
    theory_prefix = ["SLT_"]
elif has_strain_features:
    dataset_type = "Strain-injected"
    theory_prefix = ["STRAIN_"]
else:
    dataset_type = "Baseline"
    theory_prefix = []

print(f"Detected: {dataset_type}")
if theory_prefix:
    print(f"Theory feature prefixes: {', '.join(theory_prefix)}")

# ============================================================
# NODE MAPPING
# ============================================================

print("\n" + "="*70)
print("BUILDING NODE MAPPING")
print("="*70)

all_accounts = pd.concat([df[SRC_COL], df[DST_COL]]).unique()
acct2idx = {acct: i for i, acct in enumerate(all_accounts)}
num_nodes = len(acct2idx)
print(f"✓ Unique accounts (nodes): {num_nodes:,}")

src_idx = df[SRC_COL].map(acct2idx).values
dst_idx = df[DST_COL].map(acct2idx).values

edge_index = np.stack([src_idx, dst_idx], axis=0)  # [2, E]
num_edges = edge_index.shape[1]
print(f"✓ Total edges: {num_edges:,}")

# ============================================================
# GRAPH VALIDATION & STATISTICS
# ============================================================

print("\n" + "="*70)
print("GRAPH STATISTICS")
print("="*70)

# Basic statistics
avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0

print(f"Nodes: {num_nodes:,}")
print(f"Edges: {num_edges:,}")
print(f"Average degree: {avg_degree:.2f}")
print(f"Graph density: {density:.6f}")

# Self-loops check
self_loop_mask = src_idx == dst_idx
num_self_loops = self_loop_mask.sum()
print(f"Self-loops: {num_self_loops:,} ({100*num_self_loops/num_edges:.2f}%)")

if REMOVE_SELF_LOOPS and num_self_loops > 0:
    print(f"  → Removing {num_self_loops:,} self-loops...")
    keep_mask = ~self_loop_mask
    edge_index = edge_index[:, keep_mask]
    df = df[keep_mask].reset_index(drop=True)
    src_idx = src_idx[keep_mask]
    dst_idx = dst_idx[keep_mask]
    num_edges = edge_index.shape[1]
    print(f"  → Remaining edges: {num_edges:,}")

# Isolated nodes check
unique_src = set(src_idx)
unique_dst = set(dst_idx)
connected_nodes = unique_src | unique_dst
num_isolated = num_nodes - len(connected_nodes)
print(f"Isolated nodes: {num_isolated:,} ({100*num_isolated/num_nodes:.2f}%)")

if num_isolated > 0:
    print("  ⚠ Warning: Graph contains isolated nodes")

# ============================================================
# TIMESTAMPS
# ============================================================

print("\n" + "="*70)
print("TEMPORAL FEATURES")
print("="*70)

timestamps = (df[TS_COL].astype("int64") // 10**9).values  # UNIX seconds

time_span_days = (timestamps.max() - timestamps.min()) / (3600 * 24)
print(f"Timestamp range: {timestamps.min()} → {timestamps.max()}")
print(f"Time span: {time_span_days:.1f} days")

# Temporal ordering validation (critical for GraphSAGE-T)
time_diffs = np.diff(timestamps)
num_negative_gaps = (time_diffs < 0).sum()
print(f"Min time gap: {time_diffs.min()} seconds")
print(f"Max time gap: {time_diffs.max()} seconds")
print(f"Temporal violations: {num_negative_gaps}")

if num_negative_gaps > 0:
    print("  ⚠ WARNING: Dataset has temporal ordering violations!")
    print("  → GraphSAGE-T requires strictly increasing timestamps")

# ============================================================
# EDGE LABELS
# ============================================================

y_edge = df[LABEL_COL].astype(int).values

print("\n" + "="*70)
print("LABEL DISTRIBUTION")
print("="*70)

num_laundering_edges = y_edge.sum()
pct_laundering_edges = 100 * y_edge.mean()
print(f"Laundering edges: {num_laundering_edges:,} ({pct_laundering_edges:.2f}%)")
print(f"Normal edges: {num_edges - num_laundering_edges:,} ({100 - pct_laundering_edges:.2f}%)")

# ============================================================
# EDGE FEATURES
# ============================================================

print("\n" + "="*70)
print("EDGE FEATURES")
print("="*70)

edge_feature_cols = []

for col in df.columns:
    if col in EXCLUDE_COLS:
        continue
    if not np.issubdtype(df[col].dtype, np.number):
        continue
    
    # Include theory-specific features + structural features
    is_theory_feature = any(col.startswith(prefix) for prefix in theory_prefix)
    is_structural = col in WHITELIST_STRUCT_COLS
    
    if is_theory_feature or is_structural:
        edge_feature_cols.append(col)

edge_feature_cols = sorted(edge_feature_cols)

print(f"Selected {len(edge_feature_cols)} edge feature columns:")
for i, c in enumerate(edge_feature_cols, 1):
    print(f"  {i:2d}. {c}")

# Extract and clean edge features
print("\nCleaning edge features (NaN/Inf handling)...")
edge_attr_df = df[edge_feature_cols].copy()

# Check for issues before cleaning
nan_before = edge_attr_df.isna().sum().sum()
inf_before = np.isinf(edge_attr_df.select_dtypes(include=[np.number]).values).sum()

if nan_before > 0:
    print(f"  Found {nan_before:,} NaN values → filling with 0")
if inf_before > 0:
    print(f"  Found {inf_before:,} Inf values → replacing with 0")

# Clean
edge_attr_df = edge_attr_df.fillna(0).replace([np.inf, -np.inf], 0)

# Memory optimization: use float32 when possible
for col in edge_feature_cols:
    max_val = edge_attr_df[col].abs().max()
    if max_val < 1e7:  # Safe range for float32
        edge_attr_df[col] = edge_attr_df[col].astype(np.float32)
    else:
        edge_attr_df[col] = edge_attr_df[col].astype(np.float64)

edge_attr = edge_attr_df.values
print(f"✓ edge_attr shape: {edge_attr.shape}")

# Final validation
nan_after = np.isnan(edge_attr).sum()
inf_after = np.isinf(edge_attr).sum()

if nan_after > 0 or inf_after > 0:
    print(f"  ✗ ERROR: Still have {nan_after} NaNs and {inf_after} Infs!")
else:
    print("  ✓ No NaN/Inf values in edge features")

# Compute feature statistics for normalization
edge_attr_means = edge_attr.mean(axis=0)
edge_attr_stds = edge_attr.std(axis=0)
edge_attr_mins = edge_attr.min(axis=0)
edge_attr_maxs = edge_attr.max(axis=0)

# ============================================================
# NODE FEATURES
# ============================================================

print("\n" + "="*70)
print("NODE FEATURES")
print("="*70)

node_df = pd.DataFrame({"acct": list(acct2idx.keys())})
node_df["node_id"] = node_df["acct"].map(acct2idx)

# Degrees
out_deg_series = df.groupby(SRC_COL).size()
in_deg_series = df.groupby(DST_COL).size()

node_df["out_degree"] = node_df["acct"].map(out_deg_series).fillna(0).astype(float)
node_df["in_degree"] = node_df["acct"].map(in_deg_series).fillna(0).astype(float)
node_df["total_degree"] = node_df["out_degree"] + node_df["in_degree"]

# Laundering involvement count
laund_src = df[df[LABEL_COL] == 1][SRC_COL]
laund_dst = df[df[LABEL_COL] == 1][DST_COL]
laund_acct_counts = (
    pd.concat([laund_src, laund_dst], axis=0)
    .value_counts()
    .reindex(node_df["acct"])
    .fillna(0)
    .astype(float)
)

node_df["laundering_count"] = laund_acct_counts.values

# Extract and validate
x = node_df[["out_degree", "in_degree", "total_degree", "laundering_count"]].fillna(0).values
print(f"✓ x (node features) shape: {x.shape}")

# Check for issues
nan_count = np.isnan(x).sum()
if nan_count > 0:
    print(f"  ⚠ WARNING: {nan_count} NaN values in node features")
else:
    print("  ✓ No NaN values in node features")

# ============================================================
# NODE LABELS
# ============================================================

y_node = np.zeros(num_nodes, dtype=np.int64)
laund_accts = set(laund_src.astype(str)) | set(laund_dst.astype(str))
for acct in laund_accts:
    y_node[acct2idx[acct]] = 1

num_laundering_nodes = y_node.sum()
pct_laundering_nodes = 100 * y_node.mean()
print(f"Laundering nodes: {num_laundering_nodes:,} ({pct_laundering_nodes:.2f}%)")
print(f"Normal nodes: {num_nodes - num_laundering_nodes:,} ({100 - pct_laundering_nodes:.2f}%)")

# ============================================================
# CONVERT TO TORCH & SAVE
# ============================================================

print("\n" + "="*70)
print("SAVING TENSORS")
print("="*70)

edge_index_t = torch.tensor(edge_index, dtype=torch.long)
edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)
x_t = torch.tensor(x, dtype=torch.float32)
timestamps_t = torch.tensor(timestamps, dtype=torch.long)
y_edge_t = torch.tensor(y_edge, dtype=torch.long)
y_node_t = torch.tensor(y_node, dtype=torch.long)

print(f"Output directory: {OUT_DIR}\n")

torch.save(edge_index_t, os.path.join(OUT_DIR, "edge_index.pt"))
print("  ✓ edge_index.pt")

torch.save(edge_attr_t, os.path.join(OUT_DIR, "edge_attr.pt"))
print("  ✓ edge_attr.pt")

torch.save(x_t, os.path.join(OUT_DIR, "x.pt"))
print("  ✓ x.pt")

torch.save(timestamps_t, os.path.join(OUT_DIR, "timestamps.pt"))
print("  ✓ timestamps.pt")

torch.save(y_edge_t, os.path.join(OUT_DIR, "y_edge.pt"))
print("  ✓ y_edge.pt")

torch.save(y_node_t, os.path.join(OUT_DIR, "y_node.pt"))
print("  ✓ y_node.pt")

# Save metadata
with open(os.path.join(OUT_DIR, "node_mapping.json"), "w", encoding="utf-8") as f:
    json.dump(acct2idx, f)
print("  ✓ node_mapping.json")

with open(os.path.join(OUT_DIR, "edge_attr_cols.json"), "w", encoding="utf-8") as f:
    json.dump(edge_feature_cols, f, indent=2)
print("  ✓ edge_attr_cols.json")

# Save feature statistics for normalization
feature_stats = {
    "edge_attr_means": edge_attr_means.tolist(),
    "edge_attr_stds": edge_attr_stds.tolist(),
    "edge_attr_mins": edge_attr_mins.tolist(),
    "edge_attr_maxs": edge_attr_maxs.tolist(),
    "edge_attr_cols": edge_feature_cols,
}

with open(os.path.join(OUT_DIR, "feature_stats.json"), "w", encoding="utf-8") as f:
    json.dump(feature_stats, f, indent=2)
print("  ✓ feature_stats.json")

# Save comprehensive graph statistics
graph_stats = {
    "dataset_type": dataset_type,
    "theory_prefix": theory_prefix,
    "has_rat_features": has_rat_features,
    "has_motif_features": has_motif_features,
    "has_slt_features": has_slt_features,
    "has_strain_features": has_strain_features,
    "num_nodes": int(num_nodes),
    "num_edges": int(num_edges),
    "num_self_loops": int(num_self_loops),
    "num_isolated_nodes": int(num_isolated),
    "avg_degree": float(avg_degree),
    "graph_density": float(density),
    "time_span_days": float(time_span_days),
    "num_laundering_edges": int(num_laundering_edges),
    "pct_laundering_edges": float(pct_laundering_edges),
    "num_laundering_nodes": int(num_laundering_nodes),
    "pct_laundering_nodes": float(pct_laundering_nodes),
    "num_edge_features": len(edge_feature_cols),
    "num_node_features": x.shape[1],
    "temporal_violations": int(num_negative_gaps),
    "dataset_path": INPUT_PATH,
    "dataset_name": dataset_name,
    "self_loops_removed": REMOVE_SELF_LOOPS,
}

with open(os.path.join(OUT_DIR, "graph_stats.json"), "w", encoding="utf-8") as f:
    json.dump(graph_stats, f, indent=2)
print("  ✓ graph_stats.json")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"Dataset Type:   {dataset_type}")
print(f"  edge_index:   {edge_index_t.shape}")
print(f"  edge_attr:    {edge_attr_t.shape}")
print(f"  x:            {x_t.shape}")
print(f"  timestamps:   {timestamps_t.shape}")
print(f"  y_edge:       {y_edge_t.shape}")
print(f"  y_node:       {y_node_t.shape}")

print("\n" + "="*70)
print("✓ GRAPH CONSTRUCTION COMPLETE")
print("="*70)
print(f"\nAll files saved to:")
print(f"  {OUT_DIR}")
print(f"\nReady for GraphSAGE / GraphSAGE-T training!")
print("="*70)