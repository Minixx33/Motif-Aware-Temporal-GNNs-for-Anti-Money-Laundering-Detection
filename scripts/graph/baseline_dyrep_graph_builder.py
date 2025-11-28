"""
dyrep_graph_builder_ibm_baseline.py
-----------------------------------
Build a DyRep-compatible event stream from the *baseline* dataset:
    - HI-Small_Trans.csv
    - HI-Small_accounts.csv

Design:
  ✓ EXACT same edge features as baseline static GraphSAGE builder
  ✓ EXACT same node features (degrees + entity type one-hot)
  ✓ No motif/theory features
  ✓ NO train/val/test splits here (handled by a separate script)
  ✓ Event stream format for DyRep

Outputs (in graphs_dyrep/HI-Small_Trans/):
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

TRANS_FILE = "HI-Small_Trans.csv"
ACCT_FILE  = "HI-Small_accounts.csv"

INPUT_TRANS = os.path.join(DATA_DIR, TRANS_FILE)
INPUT_ACCT  = os.path.join(DATA_DIR, ACCT_FILE)

DATASET_NAME = "HI-Small_Trans"
OUT_DIR = os.path.join(BASE_DIR, "graphs_dyrep", DATASET_NAME)
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
# LOAD & SORT
# ============================================================

print("=" * 70)
print("BASELINE DYREP GRAPH BUILDER (NO SPLITS)")
print("=" * 70)

print(f"Loading transactions: {INPUT_TRANS}")
df = pd.read_csv(INPUT_TRANS, low_memory=False)
print(f"  Loaded {len(df):,} transactions")

print(f"Loading accounts: {INPUT_ACCT}")
df_acct = pd.read_csv(INPUT_ACCT, low_memory=False)
print(f"  Loaded {len(df_acct):,} accounts")

df[TS_COL] = pd.to_datetime(df[TS_COL], errors="raise")
df = df.sort_values(TS_COL).reset_index(drop=True)

df[SRC_COL] = df[SRC_COL].astype(str)
df[DST_COL] = df[DST_COL].astype(str)

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
# EDGE FEATURES (baseline enhanced)
# ============================================================

def log1p_safe(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.where(x < 0, 0.0, x)
    return np.log1p(x)

amt_rec = df["Amount Received"].astype(float).values
amt_paid = df["Amount Paid"].astype(float).values

log_amt_rec = log1p_safe(amt_rec)
log_amt_paid = log1p_safe(amt_paid)

same_bank = (df[FROM_BANK].astype(str).values == df[TO_BANK].astype(str).values).astype(np.float32)
same_curr = (df[RCURR].astype(str).values == df[PCURR].astype(str).values).astype(np.float32)

hour_of_day = df[TS_COL].dt.hour.values.astype(np.float32)
day_of_week = df[TS_COL].dt.dayofweek.values.astype(np.float32)
is_weekend = (day_of_week >= 5).astype(np.float32)

timestamps = (df[TS_COL].astype("int64") // 10**9).values.astype(np.int64)
ts_normalized = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-10)
ts_normalized = ts_normalized.astype(np.float32)

# time since last src/dst
time_since_last_src = np.zeros(num_edges, dtype=np.float32)
last_src_time = {}
for i, (s, ts) in enumerate(zip(src, timestamps)):
    if s in last_src_time:
        time_since_last_src[i] = float(ts - last_src_time[s])
    else:
        time_since_last_src[i] = 0.0
    last_src_time[s] = ts

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

pf_dummies = pd.get_dummies(df[PFORMAT].astype(str), prefix="pf")
rc_dummies = pd.get_dummies(df[RCURR].astype(str),  prefix="rc")

edge_feat_df = pd.DataFrame({
    "log_amt_rec": log_amt_rec,
    "log_amt_paid": log_amt_paid,
    "same_bank": same_bank,
    "same_currency": same_curr,
    "hour_of_day": hour_of_day,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "ts_normalized": ts_normalized,
    "log_time_since_src": log_time_since_src,
    "log_time_since_dst": log_time_since_dst,
})

edge_feat_df = pd.concat([edge_feat_df, pf_dummies, rc_dummies], axis=1)

edge_attr_cols = list(edge_feat_df.columns)
edge_attr = edge_feat_df.replace([np.inf, -np.inf], 0.0).fillna(0.0).values.astype(np.float32)

print(f"Num edge features: {edge_attr.shape[1]}")

# ============================================================
# NODE FEATURES (degrees + entity types)
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

# Entity types from accounts CSV
df_acct["Account Number"] = df_acct["Account Number"].astype(str)

def parse_entity_type(entity_name: str) -> str:
    if not isinstance(entity_name, str) or pd.isna(entity_name):
        return "Unknown"
    if "#" in entity_name:
        return entity_name.split("#")[0].strip()
    parts = entity_name.split()
    if len(parts) <= 1:
        return entity_name.strip()
    return " ".join(parts[:-1]).strip()

df_acct["entity_type"] = df_acct["Entity Name"].apply(parse_entity_type)
acct_meta = df_acct.set_index("Account Number")[["entity_type"]]

node_df = node_df.join(acct_meta, on="acct")
node_df["entity_type"] = node_df["entity_type"].fillna("Unknown")

ent_dummies = pd.get_dummies(node_df["entity_type"], prefix="ent")

node_feat_df = pd.concat([
    node_df[[
        "out_degree", "in_degree", "total_degree",
        "log_out_degree", "log_in_degree", "log_total_degree"
    ]],
    ent_dummies
], axis=1)

node_features = node_feat_df.replace([np.inf, -np.inf], 0.0).fillna(0.0).values.astype(np.float32)

print(f"Num node features (with entity types): {node_features.shape[1]}")

# ============================================================
# EVENT TYPES (DyRep)
# ============================================================

event_type_map = {pf: i for i, pf in enumerate(df[PFORMAT].astype(str).unique())}
event_type = df[PFORMAT].astype(str).map(event_type_map).values.astype(np.int64)

print(f"Event types: {event_type_map}")

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
    "dataset_name": DATASET_NAME,
    "dataset_type": "Baseline",
    "num_nodes": int(num_nodes),
    "num_edges": int(num_edges),
    "num_edge_features": int(edge_attr.shape[1]),
    "num_node_features": int(node_features.shape[1]),
    "has_motif": False,
    "has_rat": False,
    "has_slt": False,
    "has_strain": False,
}

with open(os.path.join(OUT_DIR, "graph_stats.json"), "w", encoding="utf-8") as f:
    json.dump(graph_stats, f, indent=2)

print("=" * 70)
print("✓ BASELINE DYREP GRAPH BUILT (NO SPLITS)")
print(f"Saved to: {OUT_DIR}")
print("=" * 70)
