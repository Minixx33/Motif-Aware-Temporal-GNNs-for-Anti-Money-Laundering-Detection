"""
graph_builder_tgat_events_baseline_aligned.py
---------------------------------------------
Baseline-aligned TGAT builder with added theory + motif features.

This is the *correct* TGAT builder for:
  - Baseline (HI-Small_Trans.csv)
  - RAT / SLT / STRAIN (low/medium/high)

Baseline features reproduced EXACTLY:
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
# DATASET = os.path.join("STRAIN", "HI-Small_Trans_STRAIN_low.csv")

INPUT_PATH = os.path.join(DATA_DIR, DATASET)
dataset_name = os.path.splitext(os.path.basename(DATASET))[0]

OUT_DIR = os.path.join(BASE_DIR, "tgat_graphs", dataset_name)
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================================
# COLUMN NAMES
# =====================================================================

SRC_COL  = "Account"
DST_COL  = "Account.1"
TS_COL   = "Timestamp"
LABEL_COL = "Is Laundering"

FROM_BANK = "From Bank"
TO_BANK   = "To Bank"
RCURR     = "Receiving Currency"
PCURR     = "Payment Currency"
PFORMAT   = "Payment Format"

IDENTITY_COLS = {
    "RAT_injected", "RAT_intensity_level",
    "SLT_injected", "SLT_intensity_level",
    "STRAIN_injected", "STRAIN_intensity_level",
}

# =====================================================================
# BASELINE CATEGORIES (IMPORTANT: must match baseline TGAT exactly)
# =====================================================================

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

# =====================================================================
# HELPERS
# =====================================================================

def log1p_safe(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.where(x < 0, 0.0, x)
    return np.log1p(x)

# =====================================================================
# LOAD CSV
# =====================================================================

print("=" * 70)
print("TGAT BUILDER (BASELINE-ALIGNED + THEORY ADD-ON)")
print("=" * 70)

df = pd.read_csv(INPUT_PATH, low_memory=False)
print("Loaded rows:", len(df))

df[SRC_COL] = df[SRC_COL].astype(str)
df[DST_COL] = df[DST_COL].astype(str)
df[TS_COL]  = pd.to_datetime(df[TS_COL], errors="raise")

df = df.sort_values(TS_COL).reset_index(drop=True)

# =====================================================================
# DETECT THEORY TYPE
# =====================================================================

has_rat    = any(c.startswith("RAT_") for c in df.columns)
has_slt    = any(c.startswith("SLT_") for c in df.columns)
has_strain = any(c.startswith("STRAIN_") for c in df.columns)
has_motif  = any(c.startswith("motif_") for c in df.columns)

if has_rat:
    theory_prefix = ["RAT_", "motif_"]
elif has_slt:
    theory_prefix = ["SLT_", "motif_"]
elif has_strain:
    theory_prefix = ["STRAIN_", "motif_"]
else:
    theory_prefix = []

print("Theory prefixes:", theory_prefix)

# =====================================================================
# NODE MAPPING
# =====================================================================

all_nodes = pd.concat([df[SRC_COL], df[DST_COL]]).unique()
acct2idx = {acct: i for i, acct in enumerate(all_nodes)}

src = df[SRC_COL].map(acct2idx).values
dst = df[DST_COL].map(acct2idx).values

num_nodes  = len(acct2idx)
num_events = len(src)

# =====================================================================
# TEMPORAL CHECK
# =====================================================================

timestamps = (df[TS_COL].astype("int64") // 10**9).values
if np.any(np.diff(timestamps) < 0):
    raise RuntimeError("TGAT requires strictly increasing timestamps!")

# =====================================================================
# BUILD BASELINE EDGE FEATURES
# =====================================================================

print("\nBuilding baseline TGAT features...")

log_amt_rec  = log1p_safe(df["Amount Received"].astype(float).values)
log_amt_paid = log1p_safe(df["Amount Paid"].astype(float).values)

same_bank    = (df[FROM_BANK].astype(str).values == df[TO_BANK].astype(str).values).astype(np.float32)
same_currency= (df[RCURR].astype(str).values == df[PCURR].astype(str).values).astype(np.float32)

hour_of_day  = df[TS_COL].dt.hour.values.astype(np.float32)
day_of_week  = df[TS_COL].dt.dayofweek.values.astype(np.float32)
is_weekend   = (day_of_week >= 5).astype(np.float32)

ts_min = timestamps.min()
ts_max = timestamps.max()
ts_norm = (timestamps - ts_min) / (ts_max - ts_min + 1e-9)

# time since last src
last_src_time = {}
tsls = np.zeros(num_events, dtype=np.float32)
for i, (s, ts) in enumerate(zip(src, timestamps)):
    prev = last_src_time.get(s, ts)
    tsls[i] = ts - prev
    last_src_time[s] = ts
log_time_since_src = np.log1p(tsls)

# time since last dst
last_dst_time = {}
tsld = np.zeros(num_events, dtype=np.float32)
for i, (d, ts) in enumerate(zip(dst, timestamps)):
    prev = last_dst_time.get(d, ts)
    tsld[i] = ts - prev
    last_dst_time[d] = ts
log_time_since_dst = np.log1p(tsld)

# categorical encodings
curr_df = pd.get_dummies(df[RCURR].astype(str), prefix="cur_recv")
fmt_df  = pd.get_dummies(df[PFORMAT].astype(str), prefix="fmt")

# ensure baseline columns exist
for col in BASE_CURR:
    if col not in curr_df:
        curr_df[col] = 0.0

for col in BASE_FMT:
    if col not in fmt_df:
        fmt_df[col] = 0.0

curr_df = curr_df[BASE_CURR]
fmt_df  = fmt_df[BASE_FMT]

baseline_df = pd.DataFrame({
    "log_amt_rec": log_amt_rec,
    "log_amt_paid": log_amt_paid,
    "same_bank": same_bank,
    "same_currency": same_currency,
    "hour_of_day": hour_of_day,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "ts_normalized": ts_norm.astype(np.float32),
    "log_time_since_src": log_time_since_src,
    "log_time_since_dst": log_time_since_dst,
})

baseline_df = pd.concat([baseline_df, curr_df, fmt_df], axis=1)
baseline_df = baseline_df[BASELINE_EDGE_FEATURES]

# =====================================================================
# THEORY / MOTIF FEATURES
# =====================================================================

print("\nAdding theory/motif features...")

theory_cols = [
    c for c in df.columns
    if any(c.startswith(p) for p in theory_prefix)
    and c not in IDENTITY_COLS
]

theory_df = df[theory_cols].copy()
theory_df = theory_df.replace([np.inf, -np.inf], 0).fillna(0)

# normalize all theory/motif features
for col in theory_cols:
    v = theory_df[col].astype(np.float32).values
    m = v.mean()
    s = v.std() + 1e-6
    v = (v - m) / s
    theory_df[col] = np.clip(v, -10, 10)

# =====================================================================
# FINAL EDGE FEATURE MATRIX
# =====================================================================

edge_feat_df = pd.concat([baseline_df, theory_df], axis=1)
edge_attr_cols = list(edge_feat_df.columns)
edge_attr = edge_feat_df.values.astype(np.float32)

# =====================================================================
# NODE FEATURES (same structure as your existing TGAT)
# =====================================================================

node_df = pd.DataFrame({"acct": list(acct2idx.keys())})
node_df["node_id"] = node_df["acct"].map(acct2idx)

out_deg = df.groupby(SRC_COL).size().reindex(node_df["acct"]).fillna(0).astype(float)
in_deg  = df.groupby(DST_COL).size().reindex(node_df["acct"]).fillna(0).astype(float)

node_df["out_degree"] = out_deg
node_df["in_degree"]  = in_deg
node_df["total_degree"] = out_deg + in_deg

# laundering involvement count
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

x_node = node_df[["out_degree","in_degree","total_degree","laundering_count"]].values.astype(np.float32)

# =====================================================================
# NODE LABELS
# =====================================================================

y_node = np.zeros(num_nodes, dtype=np.int64)
for acct in laund_counts.index:
    y_node[acct2idx[acct]] = 1

# =====================================================================
# SAVE TENSORS
# =====================================================================

print("\nSaving tensors to:", OUT_DIR)

torch.save(torch.tensor(src),       os.path.join(OUT_DIR,"src_nodes.pt"))
torch.save(torch.tensor(dst),       os.path.join(OUT_DIR,"dst_nodes.pt"))
torch.save(torch.tensor(timestamps),os.path.join(OUT_DIR,"timestamps.pt"))
torch.save(torch.tensor(edge_attr), os.path.join(OUT_DIR,"edge_attr.pt"))
torch.save(torch.tensor(y_edge),    os.path.join(OUT_DIR,"y_edge.pt"))
torch.save(torch.tensor(x_node),    os.path.join(OUT_DIR,"x_node.pt"))
torch.save(torch.tensor(y_node),    os.path.join(OUT_DIR,"y_node.pt"))

with open(os.path.join(OUT_DIR,"edge_attr_cols.json"),"w") as f:
    json.dump(edge_attr_cols,f,indent=2)

with open(os.path.join(OUT_DIR,"node_mapping.json"),"w") as f:
    json.dump(acct2idx,f)

print("\n✓ TGAT baseline-aligned event graph built.")
print("Saved to:", OUT_DIR)
print("=" * 70)
