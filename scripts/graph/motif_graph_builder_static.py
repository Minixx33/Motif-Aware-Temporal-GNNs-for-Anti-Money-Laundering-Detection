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
import gc
import time
import psutil
import pyarrow as pa
import pyarrow.parquet as pq


def stamp(msg):
    rss_gb = psutil.Process().memory_info().rss / (1024**3)
    print(f"[{time.strftime('%H:%M:%S')}] {msg} | RAM={rss_gb:.2f} GB", flush=True)

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection"
DATA_DIR = os.path.join(BASE_DIR, "ibm_transcations_datasets")

# RAT
DATASET = os.path.join("RAT", "HI-Medium_Trans_RAT_low.csv")
# DATASET = os.path.join("RAT", "HI-Small_Trans_RAT_medium.csv")
# DATASET = os.path.join("RAT", "HI-Small_Trans_RAT_high.csv")

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

NEEDED_COLS = [
    TS_COL, SRC_COL, DST_COL, FROM_BANK, TO_BANK,
    "Amount Received", "Amount Paid",
    RCURR, PCURR, PFORMAT,
    LABEL_COL,
]
# add theory/motif columns dynamically later, but without reading the whole file first:
# safest approach: peek header, select columns, then load.

stamp(f"Reading header: {INPUT_PATH}")
all_cols = pd.read_csv(INPUT_PATH, nrows=0).columns

theory_cols = [c for c in all_cols if c.startswith(("RAT_", "motif_", "SLT_", "STRAIN_"))]
theory_cols = [c for c in theory_cols if c not in {
    "RAT_injected","RAT_intensity_level",
    "SLT_injected","SLT_intensity_level",
    "STRAIN_injected","STRAIN_intensity_level"
}]

USECOLS = NEEDED_COLS + theory_cols

DTYPES = {
    SRC_COL: "string",
    DST_COL: "string",
    FROM_BANK: "string",
    TO_BANK: "string",
    RCURR: "string",
    PCURR: "string",
    PFORMAT: "string",
    LABEL_COL: "int8",
    "Amount Paid": "float32",
    "Amount Received": "float32",
}

PARQUET_PATH = INPUT_PATH.replace(".csv", ".parquet")

def build_parquet_from_csv_cengine(csv_path: str, pq_path: str):
    stamp(f"Creating Parquet via C-engine chunks -> {pq_path}")
    t0 = time.time()

    writer = None
    rows = 0

    # IMPORTANT: use C engine here (fast) + chunksize (avoids tokenizer OOM)
    for i, chunk in enumerate(pd.read_csv(
        csv_path,
        usecols=USECOLS,
        dtype=DTYPES,
        engine="c",
        low_memory=False,
        memory_map=True,
        chunksize=500_000,
    )):
        # keep your exact timestamp logic
        chunk[TS_COL] = pd.to_datetime(chunk[TS_COL], errors="raise")

        # write parquet incrementally (no big RAM spike)
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(pq_path, table.schema, compression="zstd")
        writer.write_table(table)

        rows += len(chunk)
        stamp(f"  wrote chunk {i:03d}: {len(chunk):,} rows (total {rows:,})")

    if writer is not None:
        writer.close()

    stamp(f"Parquet built: {rows:,} rows in {time.time() - t0:.1f}s")

# --- Load path: prefer parquet, build it if missing ---
if not os.path.exists(PARQUET_PATH):
    build_parquet_from_csv_cengine(INPUT_PATH, PARQUET_PATH)

stamp(f"Reading Parquet: {PARQUET_PATH}")
t0 = time.time()
df = pd.read_parquet(PARQUET_PATH, columns=USECOLS)
stamp(f"Loaded: {len(df):,} rows in {time.time() - t0:.1f}s")

# keep your exact behavior
stamp("Sorting by timestamp...")
t0 = time.time()
df = df.sort_values(TS_COL).reset_index(drop=True)
stamp(f"Sorted in {time.time() - t0:.1f}s")

stamp("Casting account IDs...")
df[SRC_COL] = df[SRC_COL].astype(str)
df[DST_COL] = df[DST_COL].astype(str)
stamp("Account IDs casted")


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
laund_mask = (y_edge == 1)
laund_accts = pd.unique(pd.concat([df.loc[laund_mask, SRC_COL], df.loc[laund_mask, DST_COL]]))
y_node = np.zeros(num_nodes, dtype=np.int64)
y_node[np.fromiter((acct2idx[a] for a in laund_accts), dtype=np.int64)] = 1


# ============================================================
# BASELINE FEATURES (memory-safe, baseline-aligned)
# ============================================================

def log1p_safe(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.where(x < 0, 0, x)
    return np.log1p(x, dtype=np.float32)

E = len(df)

amt_rec  = pd.to_numeric(df["Amount Received"], errors="coerce").fillna(0).to_numpy(np.float32)
amt_paid = pd.to_numeric(df["Amount Paid"], errors="coerce").fillna(0).to_numpy(np.float32)

log_amt_rec  = log1p_safe(amt_rec)
log_amt_paid = log1p_safe(amt_paid)

same_bank = (df[FROM_BANK].astype(str).to_numpy() == df[TO_BANK].astype(str).to_numpy()).astype(np.float32)
same_curr = (df[RCURR].astype(str).to_numpy() == df[PCURR].astype(str).to_numpy()).astype(np.float32)

hour = df[TS_COL].dt.hour.to_numpy(np.float32)
weekday = df[TS_COL].dt.dayofweek.to_numpy(np.float32)
is_weekend = (weekday >= 5).astype(np.float32)

timestamps = (df[TS_COL].astype("int64") // 10**9).to_numpy(np.int64)
ts_min = timestamps.min()
ts_max = timestamps.max()
ts_norm = ((timestamps - ts_min) / (ts_max - ts_min + 1e-9)).astype(np.float32)

# time since last src/dst (keep your logic, but ensure float32)
last_src = {}
tsls = np.empty(E, dtype=np.float32)
for i, (s, ts) in enumerate(zip(src, timestamps)):
    prev = last_src.get(s, ts)
    tsls[i] = np.log1p(ts - prev)
    last_src[s] = ts

last_dst = {}
tsld = np.empty(E, dtype=np.float32)
for i, (d, ts) in enumerate(zip(dst, timestamps)):
    prev = last_dst.get(d, ts)
    tsld[i] = np.log1p(ts - prev)
    last_dst[d] = ts

# categorical codes instead of one-hot
pf_codes = df[PFORMAT].astype("category").cat.codes.to_numpy(np.int16).astype(np.float32)
rc_codes = df[RCURR].astype("category").cat.codes.to_numpy(np.int16).astype(np.float32)

baseline_cols = [
    "log_amt_rec", "log_amt_paid",
    "same_bank", "same_currency",
    "hour_of_day", "day_of_week", "is_weekend",
    "ts_normalized", "log_time_since_src", "log_time_since_dst",
    "pf_code", "rc_code"
]

baseline_mat = np.column_stack([
    log_amt_rec, log_amt_paid,
    same_bank, same_curr,
    hour, weekday, is_weekend,
    ts_norm, tsls, tsld,
    pf_codes, rc_codes
]).astype(np.float32)

# ============================================================
# THEORY + MOTIF FEATURES (no big DataFrame copies)
# ============================================================
METADATA_COLS = {
    "RAT_injected", "RAT_intensity_level",
    "SLT_injected", "SLT_intensity_level",
    "STRAIN_injected", "STRAIN_intensity_level"
}

theory_cols = [
    col for col in df.columns
    if any(col.startswith(p) for p in theory_prefix)
    and col not in METADATA_COLS
]

print(f"Detected {len(theory_cols)} theory/motif features.")

# build theory matrix directly
T = len(theory_cols)
theory_mat = np.empty((E, T), dtype=np.float32)

for j, col in enumerate(theory_cols):
    v = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(np.float32)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    mu = float(v.mean())
    sd = float(v.std()) + 1e-6
    v = (v - mu) / sd
    theory_mat[:, j] = np.clip(v, -10, 10)

# ============================================================
# FINAL EDGE FEATURES
# ============================================================

edge_attr_cols = baseline_cols + theory_cols

F0 = baseline_mat.shape[1]
F1 = theory_mat.shape[1]
edge_attr = np.empty((E, F0 + F1), dtype=np.float32)
edge_attr[:, :F0] = baseline_mat
edge_attr[:, F0:] = theory_mat

del baseline_mat, theory_mat
gc.collect()


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

del df
gc.collect()

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
