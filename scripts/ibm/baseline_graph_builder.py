"""
graph_builder_ibm_small_base.py
-------------------------------
Prepare IBM HI-Small_Trans as a temporal graph dataset
for PyTorch Geometric / temporal GNNs, **without any motif features**.

Inputs (in BASE_DIR):
  - HI-Small_Trans.csv
  - HI-Small_accounts.csv

Outputs (in OUT_DIR):
  - x.pt           : [num_nodes, num_node_features] node feature matrix
  - edge_index.pt  : [2, num_edges] (src_idx, dst_idx)
  - edge_attr.pt   : [num_edges, 1] edge feature matrix (normalized amount)
  - timestamps.pt  : [num_edges] (int64 UNIX seconds)
  - y_edge.pt      : [num_edges] edge labels (Is Laundering)
  - y_node.pt      : [num_nodes] node labels (account laundering involvement)
  - node_mapping.json : {acct_id -> node_idx}
"""

import os
import json
import numpy as np
import pandas as pd
import torch

# ===================== CONFIG =====================

# Update this to your actual base directory if needed
BASE_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets"

TX_FILE = "HI-Small_Trans.csv"
ACCOUNTS_FILE = "HI-Small_accounts.csv"

# Use a separate folder for the baseline (no motifs)
OUT_DIR = os.path.join(BASE_DIR, "pyg_graph_hismall_base")
os.makedirs(OUT_DIR, exist_ok=True)

# Column names in HI-Small_Trans.csv
TS_COL = "Timestamp"
SRC_COL = "Account"
DST_COL = "Account.1"
AMT_REC_COL = "Amount Received"
AMT_PAID_COL = "Amount Paid"
LABEL_COL = "Is Laundering"

# Column name in HI-Small_accounts.csv that holds account IDs
ACCT_ID_COL = "Account Number"

# ===================== LOAD DATA =====================

tx_path = os.path.join(BASE_DIR, TX_FILE)
acc_path = os.path.join(BASE_DIR, ACCOUNTS_FILE)

print(f"Loading transactions from:\n  {tx_path}")
tx = pd.read_csv(tx_path, low_memory=False)

print(f"Loading accounts from:\n  {acc_path}")
accounts_df = pd.read_csv(acc_path, low_memory=False)

print(f"Transactions: {len(tx):,}")
print(f"Accounts rows: {len(accounts_df):,}")

# Ensure account IDs are strings
tx[SRC_COL] = tx[SRC_COL].astype(str)
tx[DST_COL] = tx[DST_COL].astype(str)
accounts_df[ACCT_ID_COL] = accounts_df[ACCT_ID_COL].astype(str)

# Parse timestamps and sort
print("Parsing and sorting by timestamp...")
tx[TS_COL] = pd.to_datetime(tx[TS_COL], errors="raise")
tx = tx.sort_values(TS_COL).reset_index(drop=True)

print(f"Timestamp range: {tx[TS_COL].min()} -> {tx[TS_COL].max()}")

# ===================== BUILD NODE INDEX MAPPING =====================

print("Building account -> node index mapping from HI-Small_accounts...")

acct_ids = accounts_df[ACCT_ID_COL].tolist()
acct2idx = {acct: i for i, acct in enumerate(acct_ids)}
num_nodes = len(acct2idx)

print(f"Unique accounts in accounts file: {num_nodes:,}")

# Sanity check: all accounts in tx should be in accounts_df
tx_accounts = set(tx[SRC_COL].unique()) | set(tx[DST_COL].unique())
missing = tx_accounts - set(acct2idx.keys())
print(f"Accounts in transactions: {len(tx_accounts):,}")
print(f"Missing accounts in accounts file: {len(missing)}")
if missing:
    print("WARNING: Some transaction accounts are missing in HI-Small_accounts.csv")

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
print(f"Positive (laundering) edges: {y_edge.sum():,} ({y_edge.mean()*100:.4f}%)")

# ===================== EDGE FEATURES (edge_attr) =====================

print("Building edge features (amount only, no motifs)...")

amount = tx[AMT_PAID_COL].astype(float).values
amt_log = np.log1p(np.maximum(amount, 0.0))  # log(1 + amount), avoid negatives
amt_log_mean = amt_log.mean()
amt_log_std = amt_log.std() if amt_log.std() > 0 else 1.0
amt_norm = (amt_log - amt_log_mean) / amt_log_std

# edge_attr = [normalized_amount]
edge_attr = amt_norm.reshape(-1, 1)
print(f"Edge feature matrix shape: {edge_attr.shape}  (dim=1)")

# ===================== NODE FEATURES (x) =====================
# Baseline structural features derived from transactions:
#   - in_degree
#   - out_degree
#   - total_in_amount
#   - total_out_amount
# All log1p-transformed and standardized.

print("Building node feature matrix x (degree + amount stats, no motifs)...")

# Initialize arrays
in_deg = np.zeros(num_nodes, dtype=np.float32)
out_deg = np.zeros(num_nodes, dtype=np.float32)
in_amt = np.zeros(num_nodes, dtype=np.float32)
out_amt = np.zeros(num_nodes, dtype=np.float32)

# Groupby for degrees and sums
print("  Computing in/out degree and in/out amount per account...")
# Out-degree and total out amount
out_deg_series = tx.groupby(SRC_COL).size()
out_amt_series = tx.groupby(SRC_COL)[AMT_PAID_COL].sum()

# In-degree and total in amount
in_deg_series = tx.groupby(DST_COL).size()
in_amt_series = tx.groupby(DST_COL)[AMT_REC_COL].sum()

# Fill arrays
for acct, val in out_deg_series.items():
    idx = acct2idx.get(acct, None)
    if idx is not None:
        out_deg[idx] = float(val)

for acct, val in in_deg_series.items():
    idx = acct2idx.get(acct, None)
    if idx is not None:
        in_deg[idx] = float(val)

for acct, val in out_amt_series.items():
    idx = acct2idx.get(acct, None)
    if idx is not None:
        out_amt[idx] = float(val)

for acct, val in in_amt_series.items():
    idx = acct2idx.get(acct, None)
    if idx is not None:
        in_amt[idx] = float(val)

def log_standardize(arr: np.ndarray) -> np.ndarray:
    """Apply log1p then standardize to mean 0, std 1."""
    logv = np.log1p(arr.astype(np.float64))
    mean = logv.mean()
    std = logv.std()
    if std == 0:
        std = 1.0
    return ((logv - mean) / std).astype(np.float32)

in_deg_n = log_standardize(in_deg)
out_deg_n = log_standardize(out_deg)
in_amt_n = log_standardize(in_amt)
out_amt_n = log_standardize(out_amt)

x = np.stack([in_deg_n, out_deg_n, in_amt_n, out_amt_n], axis=1)  # [num_nodes, 4]

print(f"Node feature matrix shape: {x.shape}  (features: in_deg, out_deg, in_amt, out_amt)")

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

print("Done.")
print(f"  edge_index.pt shape: {edge_index_t.shape}")
print(f"  x.pt shape:          {x_t.shape}")
print(f"  edge_attr.pt shape:  {edge_attr_t.shape}")
print(f"  timestamps.pt shape: {timestamps_t.shape}")
print(f"  y_edge.pt shape:     {y_edge_t.shape}")
print(f"  y_node.pt shape:     {y_node_t.shape}")
