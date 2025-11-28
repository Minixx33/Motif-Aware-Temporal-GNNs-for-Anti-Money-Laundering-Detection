"""
graph_builder_tgat_events_baseline_aligned.py
---------------------------------------------
Baseline-aligned TGAT builder with added theory + motif features.

This is the correct TGAT builder for:
  - Baseline (HI-Small_Trans.csv)
  - RAT / SLT / STRAIN (low/medium/high)

Baseline features reproduced exactly:
  log_amt_rec, log_amt_paid
  same_bank, same_currency
  hour_of_day, day_of_week, is_weekend
  ts_normalized
  log_time_since_src, log_time_since_dst
  cur_recv_*  one-hots   (Receiving Currency)
  fmt_*       one-hots   (Payment Format)

Theory + Motif features:
  - All RAT_/SLT_/STRAIN_/motif_* features
  - z-score normalized + clipped
  - identity columns removed

This version is NaN-safe and Inf-safe for TGAT.
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

# SELECT ONE DATASET
DATASET = os.path.join("RAT", "HI-Small_Trans_RAT_low.csv")
# DATASET = os.path.join("RAT", "HI-Small_Trans_RAT_medium.csv")
# DATASET = os.path.join("RAT", "HI-Small_Trans_RAT_high.csv")
# DATASET = os.path.join("SLT", "HI-Small_Trans_SLT_low.csv")
# DATASET = os.path.join("SLT", "HI-Small_Trans_SLT_medium.csv")
# DATASET = os.path.join("SLT", "HI-Small_Trans_SLT_high.csv")
# DATASET = os.path.join("STRAIN", "HI-Small_Trans_STRAIN_low.csv")
# DATASET = os.path.join("STRAIN", "HI-Small_Trans_STRAIN_medium.csv")
# DATASET = os.path.join("STRAIN", "HI-Small_Trans_STRAIN_high.csv")

INPUT_PATH = os.path.join(DATA_DIR, DATASET)
dataset_name = os.path.splitext(os.path.basename(DATASET))[0]

OUT_DIR = os.path.join(BASE_DIR, "tgat_graphs", dataset_name)
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# COLUMN NAMES
# ============================================================

SRC_COL = "Account"
DST_COL = "Account.1"
TS_COL = "Timestamp"
LABEL_COL = "Is Laundering"

FROM_BANK = "From Bank"
TO_BANK = "To Bank"
RCURR = "Receiving Currency"
PCURR = "Payment Currency"
PFORMAT = "Payment Format"

IDENTITY_COLS = {
    "RAT_injected", "RAT_intensity_level",
    "SLT_injected", "SLT_intensity_level",
    "STRAIN_injected", "STRAIN_intensity_level",
}

# ============================================================
# BASELINE CATEGORIES (must match baseline TGAT exactly)
# ============================================================

BASE_CURR = [
    "cur_recv_Australian Dollar",
    "cur_recv_Bitcoin",
    "cur_recv_Brazil Real",
    "cur_recv_Canadian Dollar",
    "cur_recv_Euro",
    "cur_recv_Mexican Peso",
    "cur_recv_Ruble",
    "cur_recv_Rupee",
    "cur_recv_Saudi Riyal",
    "cur_recv_Shekel",
    "cur_recv_Swiss Franc",
    "cur_recv_UK Pound",
    "cur_recv_US Dollar",
    "cur_recv_Yen",
    "cur_recv_Yuan",
]

BASE_FMT = [
    "fmt_ACH",
    "fmt_Bitcoin",
    "fmt_Cash",
    "fmt_Cheque",
    "fmt_Credit Card",
    "fmt_Reinvestment",
    "fmt_Wire",
]

BASELINE_EDGE_FEATURES = [
    "log_amt_rec",
    "log_amt_paid",
    "same_bank",
    "same_currency",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "ts_normalized",
    "log_time_since_src",
    "log_time_since_dst",
] + BASE_CURR + BASE_FMT


# ============================================================
# HELPERS
# ============================================================

def log1p_safe(x):
    """Safe log1p that clips negatives to zero before log1p."""
    x = np.asarray(x, dtype=np.float64)
    x = np.where(x < 0, 0.0, x)
    return np.log1p(x).astype(np.float32)


# ============================================================
# LOAD CSV
# ============================================================

print("=" * 70)
print("TGAT BUILDER (BASELINE-ALIGNED + THEORY ADD-ON, NAN-SAFE)")
print("=" * 70)

print("Input CSV:", INPUT_PATH)
df = pd.read_csv(INPUT_PATH, low_memory=False)
print("Loaded rows:", len(df))

df[SRC_COL] = df[SRC_COL].astype(str)
df[DST_COL] = df[DST_COL].astype(str)
df[TS_COL] = pd.to_datetime(df[TS_COL], errors="raise")

df = df.sort_values(TS_COL).reset_index(drop=True)

# ============================================================
# DETECT THEORY TYPE
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

print("Detected dataset type:", dataset_type)
print("Theory prefixes:", theory_prefix)

# ============================================================
# NODE MAPPING
# ============================================================

all_nodes = pd.concat([df[SRC_COL], df[DST_COL]]).unique()
acct2idx = {acct: i for i, acct in enumerate(all_nodes)}

src = df[SRC_COL].map(acct2idx).values
dst = df[DST_COL].map(acct2idx).values

num_nodes = len(acct2idx)
num_events = len(src)

print("Number of nodes:", num_nodes)
print("Number of events:", num_events)

# ============================================================
# TEMPORAL CHECK
# ============================================================

timestamps = (df[TS_COL].astype("int64") // 10**9).values
if np.any(np.diff(timestamps) < 0):
    raise RuntimeError("TGAT requires strictly increasing timestamps!")

# ============================================================
# BUILD BASELINE EDGE FEATURES
# ============================================================

print("\nBuilding baseline TGAT features...")

amount_received = df["Amount Received"].astype(float).values
amount_paid = df["Amount Paid"].astype(float).values

log_amt_rec = log1p_safe(amount_received)
log_amt_paid = log1p_safe(amount_paid)

same_bank = (df[FROM_BANK].astype(str).values == df[TO_BANK].astype(str).values).astype(np.float32)
same_currency = (df[RCURR].astype(str).values == df[PCURR].astype(str).values).astype(np.float32)

hour_of_day = df[TS_COL].dt.hour.values.astype(np.float32)
day_of_week = df[TS_COL].dt.dayofweek.values.astype(np.float32)
is_weekend = (day_of_week >= 5).astype(np.float32)

ts_min = timestamps.min()
ts_max = timestamps.max()
ts_norm = (timestamps - ts_min) / float(ts_max - ts_min + 1e-9)
ts_norm = ts_norm.astype(np.float32)

# time since last src
last_src_time = {}
tsls = np.zeros(num_events, dtype=np.float32)
for i, (s, ts) in enumerate(zip(src, timestamps)):
    prev = last_src_time.get(s, ts)
    delta = ts - prev
    if delta < 0:
        delta = 0
    tsls[i] = float(delta)
    last_src_time[s] = ts
log_time_since_src = log1p_safe(tsls)

# time since last dst
last_dst_time = {}
tsld = np.zeros(num_events, dtype=np.float32)
for i, (d, ts) in enumerate(zip(dst, timestamps)):
    prev = last_dst_time.get(d, ts)
    delta = ts - prev
    if delta < 0:
        delta = 0
    tsld[i] = float(delta)
    last_dst_time[d] = ts
log_time_since_dst = log1p_safe(tsld)

# categorical encodings
curr_df = pd.get_dummies(df[RCURR].astype(str), prefix="cur_recv")
fmt_df = pd.get_dummies(df[PFORMAT].astype(str), prefix="fmt")

# ensure baseline columns exist with consistent order
for col in BASE_CURR:
    if col not in curr_df:
        curr_df[col] = 0.0

for col in BASE_FMT:
    if col not in fmt_df:
        fmt_df[col] = 0.0

curr_df = curr_df[BASE_CURR]
fmt_df = fmt_df[BASE_FMT]

baseline_df = pd.DataFrame({
    "log_amt_rec": log_amt_rec,
    "log_amt_paid": log_amt_paid,
    "same_bank": same_bank,
    "same_currency": same_currency,
    "hour_of_day": hour_of_day,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "ts_normalized": ts_norm,
    "log_time_since_src": log_time_since_src,
    "log_time_since_dst": log_time_since_dst,
})

baseline_df = pd.concat([baseline_df, curr_df, fmt_df], axis=1)
baseline_df = baseline_df[BASELINE_EDGE_FEATURES]

# Replace any accidental NaNs/Infs in baseline features
baseline_df = baseline_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

# ============================================================
# THEORY / MOTIF FEATURES (NAN-SAFE)
# ============================================================

print("\nAdding theory/motif features (if present)...")

theory_cols = [
    c for c in df.columns
    if theory_prefix and any(c.startswith(p) for p in theory_prefix)
    and c not in IDENTITY_COLS
]

if theory_cols:
    theory_df = df[theory_cols].copy()

    # Replace obvious Inf/NaN first
    theory_df = theory_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Normalize each theory/motif feature safely
    for col in theory_cols:
        v = theory_df[col].astype(np.float32).values
        # Replace non-finite with 0 before stats
        v[~np.isfinite(v)] = 0.0

        m = np.nanmean(v)
        s = np.nanstd(v)

        if not np.isfinite(m):
            m = 0.0
        if not np.isfinite(s) or s < 1e-6:
            s = 1.0

        v = (v - m) / s
        v = np.clip(v, -10.0, 10.0)
        v = np.nan_to_num(v, nan=0.0, posinf=10.0, neginf=-10.0)

        theory_df[col] = v.astype(np.float32)

    theory_df = theory_df.astype(np.float32)
else:
    theory_df = pd.DataFrame(index=df.index)
    print("No theory/motif features detected; using baseline TGAT features only.")

# ============================================================
# FINAL EDGE FEATURE MATRIX (WITH GLOBAL CLEANUP)
# ============================================================

edge_feat_df = pd.concat([baseline_df, theory_df], axis=1)
edge_attr_cols = list(edge_feat_df.columns)

# Convert to numpy and clean globally for TGAT
edge_attr = edge_feat_df.values.astype(np.float32)
edge_attr = np.nan_to_num(edge_attr, nan=0.0, posinf=1e6, neginf=-1e6)
edge_attr = np.clip(edge_attr, -10.0, 10.0)

print("Final edge_attr shape:", edge_attr.shape)

# ============================================================
# NODE FEATURES (same structure as before)
# ============================================================

node_df = pd.DataFrame({"acct": list(acct2idx.keys())})
node_df["node_id"] = node_df["acct"].map(acct2idx)

out_deg_series = df.groupby(SRC_COL).size()
in_deg_series = df.groupby(DST_COL).size()

node_df["out_degree"] = node_df["acct"].map(out_deg_series).fillna(0).astype(float)
node_df["in_degree"] = node_df["acct"].map(in_deg_series).fillna(0).astype(float)
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
x_node = np.nan_to_num(x_node, nan=0.0, posinf=1e6, neginf=-1e6)

print("Node feature shape:", x_node.shape)

# ============================================================
# NODE LABELS
# ============================================================

y_node = np.zeros(num_nodes, dtype=np.int64)
laundering_accounts = set(laund_src) | set(laund_dst)
for acct in laundering_accounts:
    y_node[acct2idx[acct]] = 1

# ============================================================
# EDGE LABELS
# ============================================================

y_edge = df[LABEL_COL].astype(int).values

# ============================================================
# SAVE TENSORS
# ============================================================

print("\nSaving tensors to:", OUT_DIR)

torch.save(torch.tensor(src, dtype=torch.long), os.path.join(OUT_DIR, "src_nodes.pt"))
torch.save(torch.tensor(dst, dtype=torch.long), os.path.join(OUT_DIR, "dst_nodes.pt"))
torch.save(torch.tensor(timestamps, dtype=torch.long), os.path.join(OUT_DIR, "timestamps.pt"))
torch.save(torch.tensor(edge_attr, dtype=torch.float32), os.path.join(OUT_DIR, "edge_attr.pt"))
torch.save(torch.tensor(y_edge, dtype=torch.long), os.path.join(OUT_DIR, "y_edge.pt"))
torch.save(torch.tensor(x_node, dtype=torch.float32), os.path.join(OUT_DIR, "x_node.pt"))
torch.save(torch.tensor(y_node, dtype=torch.long), os.path.join(OUT_DIR, "y_node.pt"))

with open(os.path.join(OUT_DIR, "edge_attr_cols.json"), "w") as f:
    json.dump(edge_attr_cols, f, indent=2)

with open(os.path.join(OUT_DIR, "node_mapping.json"), "w") as f:
    json.dump(acct2idx, f)

# ============================================================
# GRAPH STATS (MATCHES STATIC STYLE)
# ============================================================

node_df["out_degree"] = node_df["out_degree"].fillna(0)
node_df["in_degree"] = node_df["in_degree"].fillna(0)
node_df["total_degree"] = node_df["total_degree"].fillna(0)

avg_out_degree = float(num_events) / float(num_nodes) if num_nodes > 0 else 0.0
avg_in_degree = avg_out_degree
avg_total_degree = 2.0 * avg_out_degree

graph_stats = {
    "dataset_name": dataset_name,
    "dataset_type": dataset_type,
    "theory_prefix": theory_prefix,

    "num_nodes": int(num_nodes),
    "num_events": int(num_events),
    "num_edge_features": int(edge_attr.shape[1]),
    "num_node_features": int(x_node.shape[1]),

    "num_laundering_events": int(y_edge.sum()),
    "pct_laundering_events": float(y_edge.mean() * 100.0),

    "num_laundering_nodes": int(y_node.sum()),
    "pct_laundering_nodes": float(y_node.mean() * 100.0),

    "timestamp_min": int(timestamps.min()),
    "timestamp_max": int(timestamps.max()),
    "time_span_days": float((timestamps.max() - timestamps.min()) / (3600.0 * 24.0)),
    "temporal_violations": int(np.sum(np.diff(timestamps) < 0)),

    "avg_out_degree": float(avg_out_degree),
    "avg_in_degree": float(avg_in_degree),
    "avg_total_degree": float(avg_total_degree),
    "graph_density_estimate": float(num_events / (num_nodes * max(1, num_nodes - 1))),

    "has_motif": bool(has_motif),
    "has_rat": bool(has_rat),
    "has_slt": bool(has_slt),
    "has_strain": bool(has_strain),
    "num_motif_features": int(len([c for c in edge_attr_cols if c.startswith("motif_")])),
    "num_theory_features": int(len([c for c in edge_attr_cols if c.startswith(("RAT_", "SLT_", "STRAIN_"))])),

    "edge_feature_cols": edge_attr_cols,
    "node_feature_cols": ["out_degree", "in_degree", "total_degree", "laundering_count"],

    "dataset_path": INPUT_PATH,
    "output_dir": OUT_DIR,
}

with open(os.path.join(OUT_DIR, "graph_stats.json"), "w") as f:
    json.dump(graph_stats, f, indent=2)

print("\nTGAT baseline-aligned event graph built successfully.")
print("Saved to:", OUT_DIR)
print("=" * 70)
