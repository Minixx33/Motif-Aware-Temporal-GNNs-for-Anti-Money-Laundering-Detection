"""
dyrep_graph_builder_ibm_dyrep.py
--------------------------------
Build a DyRep-compatible event stream from IBM HI-Small theory-injected datasets:

  - RAT:    HI-Small_Trans_RAT_low/medium/high.csv
  - SLT:    HI-Small_Trans_SLT_low/medium/high.csv
  - STRAIN: HI-Small_Trans_STRAIN_low/medium/high.csv

Design goals:
  ✓ Use EXACT same edge features as GraphSAGE / GraphSAGE-T
  ✓ Keep all original edges (no synthetic events, no row changes)
  ✓ NO train/val/test splits here (handled by a separate script)
  ✓ Event-based representation for DyRep

Outputs (in graphs_dyrep/{dataset_name}/):
  - src.pt
  - dst.pt
  - ts.pt
  - event_type.pt
  - edge_attr.pt
  - node_features.pt
  - labels.pt
  - y_node.pt
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

BASE_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection"
DATA_DIR = os.path.join(BASE_DIR, "ibm_transcations_datasets")

# --- SELECT DATASET HERE ---

# RAT
# DATASET = os.path.join("RAT", "HI-Small_Trans_RAT_low.csv")
# DATASET = os.path.join("RAT", "HI-Small_Trans_RAT_medium.csv")
DATASET = os.path.join("RAT", "HI-Small_Trans_RAT_high.csv")

INPUT_PATH = os.path.join(DATA_DIR, DATASET)
dataset_name = os.path.splitext(os.path.basename(DATASET))[0]

OUT_DIR = os.path.join(BASE_DIR, "graphs_dyrep", dataset_name)
os.makedirs(OUT_DIR, exist_ok=True)

# Columns
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
# LOAD + SORT DATA
# ============================================================

print("=" * 70)
print("DYREP EVENT-BASED GRAPH BUILDER (THEORY/MOTIF, NO SPLITS)")
print("=" * 70)
print(f"Loading: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Loaded {len(df):,} rows")

df[TS_COL] = pd.to_datetime(df[TS_COL], errors="raise")
df = df.sort_values(TS_COL).reset_index(drop=True)

df[SRC_COL] = df[SRC_COL].astype(str)
df[DST_COL] = df[DST_COL].astype(str)

# ============================================================
# DETECT THEORY TYPE (for metadata only)
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
    theory_prefix = []
    dataset_type = "baseline-or-other"

print(f"Detected dataset_type = {dataset_type}")

# ============================================================
# NODE MAPPING
# ============================================================

all_nodes = pd.concat([df[SRC_COL], df[DST_COL]], axis=0).unique()
acct2idx = {acct: i for i, acct in enumerate(all_nodes)}
num_nodes = len(acct2idx)

src = df[SRC_COL].map(acct2idx).values.astype(np.int64)
dst = df[DST_COL].map(acct2idx).values.astype(np.int64)
num_edges = len(src)

print(f"Num nodes: {num_nodes:,}")
print(f"Num edges: {num_edges:,}")

# ============================================================
# LABELS
# ============================================================

df[LABEL_COL] = df[LABEL_COL].astype(int)
y_edge = df[LABEL_COL].values

y_node = np.zeros(num_nodes, dtype=np.int64)
laund_src = df.loc[df[LABEL_COL] == 1, SRC_COL]
laund_dst = df.loc[df[LABEL_COL] == 1, DST_COL]
for acct in set(laund_src) | set(laund_dst):
    y_node[acct2idx[acct]] = 1

# ============================================================
# BASELINE EDGE FEATURES
# ============================================================

def log1p_safe(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.where(x < 0, 0.0, x)
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

timestamps = (df[TS_COL].astype("int64") // 10**9).values.astype(np.int64)
ts_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-9)

# time since src
last_src = {}
tsls = np.zeros(num_edges)
for i, (s, ts) in enumerate(zip(src, timestamps)):
    tsls[i] = ts - last_src.get(s, ts)
    last_src[s] = ts
tsls = np.log1p(tsls)

# time since dst
last_dst = {}
tsld = np.zeros(num_edges)
for i, (d, ts) in enumerate(zip(dst, timestamps)):
    tsld[i] = ts - last_dst.get(d, ts)
    last_dst[d] = ts
tsld = np.log1p(tsld)

pf = pd.get_dummies(df[PFORMAT].astype(str), prefix="pf")
rc = pd.get_dummies(df[RCURR].astype(str),  prefix="rc")

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

# ============================================================
# THEORY + MOTIF FEATURES (z-score normalized)
# ============================================================

theory_cols = [
    col for col in df.columns
    if any(col.startswith(p) for p in theory_prefix)
]

METADATA_COLS = {
    "RAT_injected", "RAT_intensity_level",
    "SLT_injected", "SLT_intensity_level",
    "STRAIN_injected", "STRAIN_intensity_level",
}
theory_cols = [c for c in theory_cols if c not in METADATA_COLS]

print(f"Detected {len(theory_cols)} theory/motif features.")

theory_df = df[theory_cols].copy() if theory_cols else pd.DataFrame(index=df.index)
if not theory_df.empty:
    theory_df = theory_df.replace([np.inf, -np.inf], 0).fillna(0)
    for col in theory_cols:
        v = theory_df[col].astype(np.float32).values
        std = v.std() + 1e-6
        v = (v - v.mean()) / std
        theory_df[col] = np.clip(v, -10, 10)

# ============================================================
# FINAL EDGE FEATURES = baseline + theory
# ============================================================

edge_feat_df = pd.concat([baseline_df, theory_df], axis=1)
edge_attr_cols = list(edge_feat_df.columns)
edge_attr = edge_feat_df.values.astype(np.float32)

print(f"Num edge features: {edge_attr.shape[1]}")

# ============================================================
# NODE FEATURES (degrees + entity types, vectorized)
# ============================================================

node_df = pd.DataFrame({"acct": list(acct2idx.keys())})
node_df["node_id"] = node_df["acct"].map(acct2idx)

out_deg = df.groupby(SRC_COL).size().reindex(node_df["acct"]).fillna(0).values
in_deg  = df.groupby(DST_COL).size().reindex(node_df["acct"]).fillna(0).values

node_df["out_degree"] = out_deg
node_df["in_degree"]  = in_deg
node_df["total_degree"] = out_deg + in_deg
node_df["log_out_degree"] = np.log1p(out_deg)
node_df["log_in_degree"]  = np.log1p(in_deg)
node_df["log_total_degree"] = np.log1p(node_df["total_degree"])

# -------- ENTITY TYPES FROM RAT COLUMNS (vectorized) --------

def parse_entity_type(entity_name: str) -> str:
    """Extract entity type from 'Entity Name' (same logic as baseline builder)."""
    if not isinstance(entity_name, str) or pd.isna(entity_name):
        return "Unknown"
    if "#" in entity_name:
        return entity_name.split("#")[0].strip()
    parts = entity_name.split()
    if len(parts) <= 1:
        return entity_name.strip()
    return " ".join(parts[:-1]).strip()

# Mode entity name per src account
src_entity_map = (
    df.groupby(SRC_COL)["srcacct_Entity Name"]
    .apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown")
    .to_dict()
)

# Mode entity name per dst account
dst_entity_map = (
    df.groupby(DST_COL)["dstacct_Entity Name"]
    .apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown")
    .to_dict()
)

def get_entity_name(acct):
    return src_entity_map.get(acct, dst_entity_map.get(acct, "Unknown"))

node_df["entity_name"] = node_df["acct"].apply(get_entity_name)
node_df["entity_type"] = node_df["entity_name"].apply(parse_entity_type)

ent_dummies = pd.get_dummies(node_df["entity_type"], prefix="ent")
print(f"Entity types found: {list(ent_dummies.columns)}")

node_feat_df = pd.concat([
    node_df[[
        "out_degree", "in_degree", "total_degree",
        "log_out_degree", "log_in_degree", "log_total_degree"
    ]],
    ent_dummies
], axis=1)

node_features = node_feat_df.values.astype(np.float32)
print(f"Num node features (with entity types): {node_features.shape[1]}")

# ============================================================
# EVENT TYPES (DyRep)
# ============================================================

event_type_map = {pf: i for i, pf in enumerate(df[PFORMAT].astype(str).unique())}
event_type = df[PFORMAT].astype(str).map(event_type_map).values.astype(np.int64)

print(f"Event types: {event_type_map}")

# ============================================================
# SANITY: TEMPORAL ORDER
# ============================================================

assert np.all(np.diff(timestamps) >= 0), "Timestamps must be non-decreasing for DyRep!"

# ============================================================
# SAVE (NO SPLITS)
# ============================================================

torch.save(torch.tensor(src, dtype=torch.long), os.path.join(OUT_DIR, "src.pt"))
torch.save(torch.tensor(dst, dtype=torch.long), os.path.join(OUT_DIR, "dst.pt"))
torch.save(torch.tensor(timestamps, dtype=torch.long), os.path.join(OUT_DIR, "ts.pt"))
torch.save(torch.tensor(event_type, dtype=torch.long), os.path.join(OUT_DIR, "event_type.pt"))

torch.save(torch.tensor(edge_attr, dtype=torch.float32), os.path.join(OUT_DIR, "edge_attr.pt"))
torch.save(torch.tensor(node_features, dtype=torch.float32), os.path.join(OUT_DIR, "node_features.pt"))

torch.save(torch.tensor(y_edge, dtype=torch.long), os.path.join(OUT_DIR, "labels.pt"))
torch.save(torch.tensor(y_node, dtype=torch.long), os.path.join(OUT_DIR, "y_node.pt"))

with open(os.path.join(OUT_DIR, "edge_attr_cols.json"), "w", encoding="utf-8") as f:
    json.dump(edge_attr_cols, f, indent=2)

with open(os.path.join(OUT_DIR, "node_mapping.json"), "w", encoding="utf-8") as f:
    json.dump(acct2idx, f)

graph_stats = {
    "dataset_name": dataset_name,
    "dataset_type": dataset_type,
    "num_nodes": int(num_nodes),
    "num_edges": int(num_edges),
    "num_edge_features": int(edge_attr.shape[1]),
    "num_node_features": int(node_features.shape[1]),
    "has_motif": bool(has_motif),
    "has_rat": bool(has_rat),
    "has_slt": bool(has_slt),
    "has_strain": bool(has_strain),
}

with open(os.path.join(OUT_DIR, "graph_stats.json"), "w", encoding="utf-8") as f:
    json.dump(graph_stats, f, indent=2)

print("=" * 70)
print("✓ DYREP EVENT GRAPH BUILT (NO SPLITS)")
print(f"Saved to: {OUT_DIR}")
print("=" * 70)
