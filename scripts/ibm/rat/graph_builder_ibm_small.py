"""
graph_builder_ibm_small.py
--------------------------
Prepare IBM HI-Small_Trans_RAT as a temporal graph dataset
for PyTorch Geometric / PyTorch Geometric Temporal.

Outputs (in OUT_DIR):
  - x.pt           : [num_nodes, num_node_features] node feature matrix
  - edge_index.pt  : [2, num_edges] (src_idx, dst_idx)
  - edge_attr.pt   : [num_edges, num_edge_features]
  - timestamps.pt  : [num_edges] (int64 UNIX seconds)
  - y_edge.pt      : [num_edges] edge labels (Is Laundering)
  - y_node.pt      : [num_nodes] node labels (account laundering involvement)
  - node_mapping.json : {acct_id -> node_idx}
  - motif_mapping.json: {motif_name -> motif_idx}
"""

import os
import json
import numpy as np
import pandas as pd
import torch

# ===================== CONFIG =====================

BASE_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets\RAT"

TX_FILE = "HI-Small_Trans_RAT.csv"
NODE_FEATS_FILE = "HI-Small_motif_features.csv"

OUT_DIR = os.path.join(BASE_DIR, "pyg_graph_hismall")
os.makedirs(OUT_DIR, exist_ok=True)

SRC_COL = "Account"
DST_COL = "Account.1"
TS_COL = "Timestamp"
AMT_PAID_COL = "Amount Paid"
LABEL_COL = "Is Laundering"
MOTIF_COL = "injected_motif"

# ===================== LOAD DATA =====================

tx_path = os.path.join(BASE_DIR, TX_FILE)
nf_path = os.path.join(BASE_DIR, NODE_FEATS_FILE)

print(f"Loading transactions from:\n  {tx_path}")
tx = pd.read_csv(tx_path, low_memory=False)

print(f"Loading node features from:\n  {nf_path}")
nodes_df = pd.read_csv(nf_path)

print(f"Transactions: {len(tx):,}")
print(f"Node features rows: {len(nodes_df):,}")

# Ensure account IDs are strings
tx[SRC_COL] = tx[SRC_COL].astype(str)
tx[DST_COL] = tx[DST_COL].astype(str)
nodes_df["acct_id"] = nodes_df["acct_id"].astype(str)

# Parse timestamps and sort
tx[TS_COL] = pd.to_datetime(tx[TS_COL], errors="raise")
tx = tx.sort_values(TS_COL).reset_index(drop=True)

print(f"Timestamp range: {tx[TS_COL].min()} -> {tx[TS_COL].max()}")

# ===================== BUILD NODE INDEX MAPPING =====================

print("Building account -> node index mapping from node features...")

acct_ids = nodes_df["acct_id"].tolist()
acct2idx = {acct: i for i, acct in enumerate(acct_ids)}
num_nodes = len(acct2idx)

print(f"Unique nodes (accounts): {num_nodes:,}")

# Sanity check: all accounts in tx should be in node features
missing_src = set(tx[SRC_COL].unique()) - set(acct2idx.keys())
missing_dst = set(tx[DST_COL].unique()) - set(acct2idx.keys())

if missing_src or missing_dst:
    print(f"WARNING: Missing accounts in node features:"
          f" src_missing={len(missing_src)}, dst_missing={len(missing_dst)}")

# ===================== BUILD EDGE INDEX & TIMESTAMPS =====================

print("Building edge_index and timestamps...")

src_idx = tx[SRC_COL].map(acct2idx).values
dst_idx = tx[DST_COL].map(acct2idx).values

edge_index = np.stack([src_idx, dst_idx], axis=0)  # [2, num_edges]

# Timestamps as UNIX seconds (int64)
ts = tx[TS_COL]
timestamps = (ts.astype("int64") // 10**9).values  # seconds since epoch

num_edges = edge_index.shape[1]
print(f"Edges: {num_edges:,}")

# ===================== EDGE LABELS (y_edge) =====================

print("Building edge labels (Is Laundering)...")
y_edge = tx[LABEL_COL].astype(int).values  # 0/1 per transaction

# ===================== EDGE FEATURES (edge_attr) =====================

print("Building edge features (amount + motif info)...")

# Amount: log-normalized Amount Paid
amount = tx[AMT_PAID_COL].astype(float).values
amt_log = np.log1p(np.maximum(amount, 0.0))  # avoid negatives just in case
amt_log_mean = amt_log.mean()
amt_log_std = amt_log.std() if amt_log.std() > 0 else 1.0
amt_norm = (amt_log - amt_log_mean) / amt_log_std

# Motif: categorical -> index + "is motif edge" flag
motif_series = tx[MOTIF_COL].fillna("none").astype(str)
motif_types = sorted(motif_series.unique())
motif2idx = {m: i for i, m in enumerate(motif_types)}
motif_idx = motif_series.map(motif2idx).astype(int).values
is_motif_edge = (motif_series != "none").astype(int).values

print("Motif types and indices:")
for m, i in motif2idx.items():
    print(f"  {i}: {m}")

# edge_attr = [amt_norm, is_motif_edge, motif_idx]
edge_attr = np.stack([amt_norm,
                      is_motif_edge.astype(float),
                      motif_idx.astype(float)], axis=1)

# ===================== NODE FEATURES (x) =====================

print("Building node feature matrix x...")

# Drop non-numeric / ID columns
non_feature_cols = ["acct_id", "first_ts", "last_ts"]
feature_cols = [c for c in nodes_df.columns if c not in non_feature_cols]

x = nodes_df[feature_cols].astype(float).values  # [num_nodes, num_features]

print(f"Node feature matrix shape: {x.shape}")  # (num_nodes, num_features)

# ===================== NODE LABELS (y_node) =====================

print("Building node labels (accounts involved in laundering)...")

laund_tx = tx[tx[LABEL_COL] == 1]
laund_accts = set(laund_tx[SRC_COL].astype(str)) | set(laund_tx[DST_COL].astype(str))

y_node = np.zeros(num_nodes, dtype=np.int64)
for acct in laund_accts:
    idx = acct2idx.get(acct, None)
    if idx is not None:
        y_node[idx] = 1

print(f"Accounts with laundering involvement: {y_node.sum():,}")

# ===================== CONVERT TO TORCH TENSORS =====================

print("Converting to torch tensors...")

edge_index_t = torch.tensor(edge_index, dtype=torch.long)
x_t = torch.tensor(x, dtype=torch.float32)
edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)
timestamps_t = torch.tensor(timestamps, dtype=torch.long)
y_edge_t = torch.tensor(y_edge, dtype=torch.long)
y_node_t = torch.tensor(y_node, dtype=torch.long)

# ===================== SAVE ALL ARTIFACTS =====================

print(f"Saving tensors to: {OUT_DIR}")

torch.save(edge_index_t, os.path.join(OUT_DIR, "edge_index.pt"))
torch.save(x_t, os.path.join(OUT_DIR, "x.pt"))
torch.save(edge_attr_t, os.path.join(OUT_DIR, "edge_attr.pt"))
torch.save(timestamps_t, os.path.join(OUT_DIR, "timestamps.pt"))
torch.save(y_edge_t, os.path.join(OUT_DIR, "y_edge.pt"))
torch.save(y_node_t, os.path.join(OUT_DIR, "y_node.pt"))

with open(os.path.join(OUT_DIR, "node_mapping.json"), "w") as f:
    json.dump(acct2idx, f)

with open(os.path.join(OUT_DIR, "motif_mapping.json"), "w") as f:
    json.dump(motif2idx, f)

print("Done.")
print(f"  edge_index.pt shape: {edge_index_t.shape}")
print(f"  x.pt shape:          {x_t.shape}")
print(f"  edge_attr.pt shape:  {edge_attr_t.shape}")
print(f"  timestamps.pt shape: {timestamps_t.shape}")
print(f"  y_edge.pt shape:     {y_edge_t.shape}")
print(f"  y_node.pt shape:     {y_node_t.shape}")