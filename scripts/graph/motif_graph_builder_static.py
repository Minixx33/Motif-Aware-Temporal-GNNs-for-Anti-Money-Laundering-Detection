"""
graph_builder_theory_aligned.py
---------------------------------------------------
Baseline-aligned static graph builder for:
  - HI-Small_Trans_RAT_low/medium/high.csv
  - HI-Small_Trans_SLT_low/medium/high.csv
  - HI-Small_Trans_STRAIN_low/medium/high.csv

This version EXACTLY replicates the baseline builder behavior:
  ✓ baseline temporal features
  ✓ baseline engineered numeric features
  ✓ baseline categorical one-hot encodings
  ✓ baseline structural node features
  ✓ z-score normalization on ALL numeric theory features
  ✓ NO identity/leaky columns
  ✓ Guaranteed compatibility with GraphSAGE + GraphSAGE-T

Everything baseline → preserved.
Only difference → added criminology theory + motif features.
"""

import os
import json
import numpy as np
import pandas as pd
import torch

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection"
DATA_DIR = os.path.join(BASE_DIR, "ibm_transcations_datasets")

# RAT
# DATASET = os.path.join("RAT", "HI-Small_Trans_RAT_low.csv")
# DATASET = os.path.join("RAT", "HI-Small_Trans_RAT_medium.csv")
DATASET = os.path.join("RAT", "HI-Small_Trans_RAT_high.csv")

# SLT
# DATASET = os.path.join("SLT", "HI-Small_Trans_SLT_low.csv")
# DATASET = os.path.join("SLT", "HI-Small_Trans_SLT_low.csv")
# DATASET = os.path.join("SLT", "HI-Small_Trans_SLT_low.csv")

# STRAIN
# DATASET = os.path.join("STRAIN", "HI-Small_Trans_STRAIN_low.csv")
# DATASET = os.path.join("STRAIN", "HI-Small_Trans_STRAIN_medium.csv")
# DATASET = os.path.join("STRAIN", "HI-Small_Trans_STRAIN_high.csv")


INPUT_PATH = os.path.join(DATA_DIR, DATASET)
dataset_name = os.path.splitext(os.path.basename(DATASET))[0]

OUT_DIR = os.path.join(BASE_DIR, "graphs", dataset_name)
os.makedirs(OUT_DIR, exist_ok=True)

# Column names
TS_COL = "Timestamp"
SRC_COL = "Account"
DST_COL = "Account.1"
FROM_BANK = "From Bank"
TO_BANK = "To Bank"
RCURR = "Receiving Currency"
PCURR = "Payment Currency"
PFORMAT = "Payment Format"
LABEL_COL = "Is Laundering"

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BASELINE-ALIGNED THEORY GRAPH BUILDER")
print("=" * 70)

df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Loaded {len(df):,} rows")

df[TS_COL] = pd.to_datetime(df[TS_COL], errors="raise")
df = df.sort_values(TS_COL).reset_index(drop=True)

df[SRC_COL] = df[SRC_COL].astype(str)
df[DST_COL] = df[DST_COL].astype(str)

# ============================================================
# DETECT THEORY TYPE
# ============================================================

has_rat = any(col.startswith("RAT_") for col in df.columns)
has_slt = any(col.startswith("SLT_") for col in df.columns)
has_strain = any(col.startswith("STRAIN_") for col in df.columns)
has_motif = any(col.startswith("motif_") for col in df.columns)

if has_rat:
    theory_prefix = ["RAT_", "motif_"]
    dataset_type = "RAT-injected"
elif has_slt:
    theory_prefix = ["SLT_", "motif_"]
    dataset_type = "SLT-injected"
elif has_strain:
    theory_prefix = ["STRAIN_", "motif_"]
    dataset_type = "STRAIN-injected"
else:
    raise ValueError("ERROR: This builder is ONLY for theory-injected datasets")

print(f"Detected: {dataset_type}")

# ============================================================
# NODE MAPPING
# ============================================================

all_nodes = pd.concat([df[SRC_COL], df[DST_COL]]).unique()
acct2idx = {acct: i for i, acct in enumerate(all_nodes)}
num_nodes = len(acct2idx)

src = df[SRC_COL].map(acct2idx).values
dst = df[DST_COL].map(acct2idx).values
edge_index = np.stack([src, dst], axis=0)
num_edges = len(src)

# ============================================================
# LABELS
# ============================================================

y_edge = df[LABEL_COL].astype(int).values

# Node labels: any account involved in laundering becomes 1
y_node = np.zeros(num_nodes, dtype=np.int64)
laund_src = df.loc[df[LABEL_COL] == 1, SRC_COL].astype(str)
laund_dst = df.loc[df[LABEL_COL] == 1, DST_COL].astype(str)
for acct in set(laund_src) | set(laund_dst):
    y_node[acct2idx[acct]] = 1

# ============================================================
# BASELINE FEATURES
# (replicate EXACT baseline design)
# ============================================================

def log1p_safe(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.where(x < 0, 0, x)
    return np.log1p(x)

amt_rec = df["Amount Received"].astype(float).values
amt_paid = df["Amount Paid"].astype(float).values

log_amt_rec = log1p_safe(amt_rec)
log_amt_paid = log1p_safe(amt_paid)

same_bank = (df[FROM_BANK].astype(str) == df[TO_BANK].astype(str)).astype(float)
same_curr = (df[RCURR].astype(str) == df[PCURR].astype(str)).astype(float)

hour = df[TS_COL].dt.hour.values.astype(float)
weekday = df[TS_COL].dt.dayofweek.values.astype(float)
is_weekend = (weekday >= 5).astype(float)

# normalized timestamps
timestamps = (df[TS_COL].astype("int64") // 10**9).values
ts_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-9)

# time since last src
last_src = {}
tsls = np.zeros(num_edges)
for i, (s, ts) in enumerate(zip(src, timestamps)):
    tsls[i] = ts - last_src.get(s, ts)
    last_src[s] = ts
tsls = np.log1p(tsls)

# time since last dst
last_dst = {}
tsld = np.zeros(num_edges)
for i, (d, ts) in enumerate(zip(dst, timestamps)):
    tsld[i] = ts - last_dst.get(d, ts)
    last_dst[d] = ts
tsld = np.log1p(tsld)

pf = pd.get_dummies(df[PFORMAT].astype(str), prefix="pf")
rc = pd.get_dummies(df[RCURR].astype(str), prefix="rc")

baseline_df = pd.DataFrame({
    "log_amt_rec": log_amt_rec,
    "log_amt_paid": log_amt_paid,
    "same_bank": same_bank,
    "same_currency": same_curr,
    "hour_of_day": hour,
    "day_of_week": weekday,
    "is_weekend": is_weekend,
    "ts_normalized": ts_norm.astype(np.float32),
    "log_time_since_src": tsls.astype(np.float32),
    "log_time_since_dst": tsld.astype(np.float32),
})

baseline_df = pd.concat([baseline_df, pf, rc], axis=1)

baseline_cols = list(baseline_df.columns)

# ============================================================
# THEORY + MOTIF FEATURES
# ============================================================

theory_cols = [
    col for col in df.columns
    if any(col.startswith(p) for p in theory_prefix)
]

# Remove theory metadata flags (NOT real features)
METADATA_COLS = {
    "RAT_injected", "RAT_intensity_level",
    "SLT_injected", "SLT_intensity_level",
    "STRAIN_injected", "STRAIN_intensity_level"
}

theory_cols = [c for c in theory_cols if c not in METADATA_COLS]

print(f"Detected {len(theory_cols)} theory/motif features.")

# sanitize theory df
theory_df = df[theory_cols].copy()
theory_df = theory_df.replace([np.inf, -np.inf], 0).fillna(0)

# normalize theory/motif features
for col in theory_cols:
    v = theory_df[col].astype(np.float32).values
    std = v.std() + 1e-6
    v = (v - v.mean()) / std
    theory_df[col] = np.clip(v, -10, 10)

# ============================================================
# FINAL EDGE FEATURES
# ============================================================

edge_feat_df = pd.concat([baseline_df, theory_df], axis=1)
edge_attr_cols = list(edge_feat_df.columns)
edge_attr = edge_feat_df.values.astype(np.float32)

# ============================================================
# NODE FEATURES (same as baseline)
# ============================================================

node_df = pd.DataFrame({"acct": list(acct2idx.keys())})
node_df["node_id"] = node_df["acct"].map(acct2idx)

out_deg = df.groupby(SRC_COL).size().reindex(node_df["acct"]).fillna(0).values
in_deg  = df.groupby(DST_COL).size().reindex(node_df["acct"]).fillna(0).values

node_df["out_degree"] = out_deg
node_df["in_degree"] = in_deg
node_df["total_degree"] = out_deg + in_deg

node_df["log_out_degree"] = np.log1p(out_deg)
node_df["log_in_degree"]  = np.log1p(in_deg)
node_df["log_total_degree"] = np.log1p(node_df["total_degree"])

x = node_df[[
    "out_degree", "in_degree", "total_degree",
    "log_out_degree", "log_in_degree", "log_total_degree"
]].values.astype(np.float32)

# ============================================================
# SAVE EVERYTHING
# ============================================================

torch.save(torch.tensor(edge_index, dtype=torch.long), os.path.join(OUT_DIR, "edge_index.pt"))
torch.save(torch.tensor(edge_attr, dtype=torch.float32), os.path.join(OUT_DIR, "edge_attr.pt"))
torch.save(torch.tensor(x, dtype=torch.float32), os.path.join(OUT_DIR, "x.pt"))
torch.save(torch.tensor(timestamps, dtype=torch.long), os.path.join(OUT_DIR, "timestamps.pt"))
torch.save(torch.tensor(y_edge, dtype=torch.long), os.path.join(OUT_DIR, "y_edge.pt"))
torch.save(torch.tensor(y_node, dtype=torch.long), os.path.join(OUT_DIR, "y_node.pt"))

with open(os.path.join(OUT_DIR, "edge_attr_cols.json"), "w") as f:
    json.dump(edge_attr_cols, f, indent=2)

with open(os.path.join(OUT_DIR, "node_mapping.json"), "w") as f:
    json.dump(acct2idx, f)

graph_stats = {
    "dataset_type": dataset_type,
    "num_nodes": int(num_nodes),
    "num_edges": int(num_edges),
    "num_edge_features": len(edge_attr_cols),
    "num_node_features": x.shape[1],
    "has_motif": has_motif,
    "has_rat": has_rat,
    "has_slt": has_slt,
    "has_strain": has_strain,
}
with open(os.path.join(OUT_DIR, "graph_stats.json"), "w") as f:
    json.dump(graph_stats, f, indent=2)

print("="*70)
print("✓ BASELINE-ALIGNED THEORY GRAPH BUILT")
print(f"Saved to: {OUT_DIR}")
print("="*70)
