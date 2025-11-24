"""
rat_injector_ibm_small_v2.py
----------------------------
Injects RAT-style laundering motifs (fan-in, chains, cycles)
into the IBM HI-Small_Trans dataset.

Input:
  - HI-Small_Trans.csv  (original IBM transactions)

Output:
  - HI-Small_Trans_RAT.csv  (with extra injected motif edges)
"""

import os
import random
import numpy as np
import pandas as pd

# ===================== CONFIG =====================

INPUT_PATH  = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets\HI-Small_Trans.csv"
OUTPUT_PATH = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets\RAT\HI-Small_Trans_RAT.csv"

# ~0.08% of ~5M rows ≈ 4k injected edges
NUM_FANIN_EDGES = 1800
NUM_CHAIN_EDGES = 1400
NUM_CYCLE_EDGES = 800

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

SRC_COL      = "Account"
DST_COL      = "Account.1"
AMT_REC_COL  = "Amount Received"
AMT_PAID_COL = "Amount Paid"
TS_COL       = "Timestamp"
LABEL_COL    = "Is Laundering"

# ===================== LOAD DATA =====================

print(f"Loading IBM transactions from: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH, low_memory=False)

required_cols = [
    TS_COL, "From Bank", SRC_COL, "To Bank", DST_COL,
    AMT_REC_COL, "Receiving Currency", AMT_PAID_COL,
    "Payment Currency", "Payment Format", LABEL_COL
]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Required column '{c}' not found in dataset.")

print(f"Loaded {len(df):,} transactions")

# parse timestamps (auto-detect format)
df[TS_COL] = pd.to_datetime(df[TS_COL], errors="raise")
min_ts = df[TS_COL].min()
max_ts = df[TS_COL].max()
print(f"Timestamp range: {min_ts}  ->  {max_ts}")

# account universe
accounts_src = df[SRC_COL].astype(str)
accounts_dst = df[DST_COL].astype(str)
all_accounts = pd.Index(accounts_src).append(pd.Index(accounts_dst)).unique()
print(f"Unique accounts: {len(all_accounts):,}")

# amount distribution
amounts_paid = df[AMT_PAID_COL].astype(float).values
median_amt = np.median(amounts_paid)
std_amt    = np.std(amounts_paid)
print(f"Amount Paid median: {median_amt:.2f}, std: {std_amt:.2f}")


def sample_amount(n=1):
    vals = np.random.normal(loc=median_amt, scale=0.5 * std_amt, size=n)
    vals = np.maximum(vals, 1.0)
    return vals


def sample_timestamp(n=1):
    total_seconds = int((max_ts - min_ts).total_seconds())
    offs = np.random.randint(0, total_seconds + 1, size=n)
    return [min_ts + pd.to_timedelta(int(o), unit="s") for o in offs]


injected_rows = []

# ===================== FAN-IN MOTIFS =====================

print(f"Injecting FAN-IN motifs (~{NUM_FANIN_EDGES} edges)...")
fan_in_edges_added = 0

while fan_in_edges_added < NUM_FANIN_EDGES:
    target = str(np.random.choice(all_accounts))
    k = np.random.randint(3, 8)  # 3–7 sources
    sources = np.random.choice(all_accounts, size=k, replace=False)
    timestamps = sample_timestamp(k)
    amts = sample_amount(k)

    for src, ts_, amt_ in zip(sources, timestamps, amts):
        if fan_in_edges_added >= NUM_FANIN_EDGES:
            break

        row = {
            TS_COL: ts_,
            "From Bank": np.nan,
            SRC_COL: str(src),
            "To Bank": np.nan,
            DST_COL: target,
            AMT_REC_COL: float(amt_),
            "Receiving Currency": "US Dollar",
            AMT_PAID_COL: float(amt_),
            "Payment Currency": "US Dollar",
            "Payment Format": "ACH",
            LABEL_COL: 1,
            "injected_motif": "fanin",
        }
        injected_rows.append(row)
        fan_in_edges_added += 1

print(f"  -> Added {fan_in_edges_added} FAN-IN edges.")

# ===================== CHAIN MOTIFS =====================

print(f"Injecting CHAIN motifs (~{NUM_CHAIN_EDGES} edges)...")
chain_edges_added = 0

while chain_edges_added < NUM_CHAIN_EDGES:
    L = np.random.randint(3, 7)  # chain length
    chain_accounts = np.random.choice(all_accounts, size=L + 1, replace=False)
    timestamps = sorted(sample_timestamp(L))
    amts = sample_amount(L)
    motif_name = f"chain_L{L}"

    for i in range(L):
        if chain_edges_added >= NUM_CHAIN_EDGES:
            break

        src = str(chain_accounts[i])
        dst = str(chain_accounts[i + 1])
        ts_ = timestamps[i]
        amt_ = amts[i]

        row = {
            TS_COL: ts_,
            "From Bank": np.nan,
            SRC_COL: src,
            "To Bank": np.nan,
            DST_COL: dst,
            AMT_REC_COL: float(amt_),
            "Receiving Currency": "US Dollar",
            AMT_PAID_COL: float(amt_),
            "Payment Currency": "US Dollar",
            "Payment Format": "ACH",
            LABEL_COL: 1,
            "injected_motif": motif_name,
        }
        injected_rows.append(row)
        chain_edges_added += 1

print(f"  -> Added {chain_edges_added} CHAIN edges.")

# ===================== CYCLE MOTIFS =====================

print(f"Injecting CYCLE motifs (~{NUM_CYCLE_EDGES} edges)...")
cycle_edges_added = 0

while cycle_edges_added < NUM_CYCLE_EDGES:
    L = np.random.randint(3, 6)  # cycle length (3–5 nodes)
    cycle_accounts = np.random.choice(all_accounts, size=L, replace=False)
    timestamps = sorted(sample_timestamp(L))
    amts = sample_amount(L)
    motif_name = f"cycle_L{L}"

    for i in range(L):
        if cycle_edges_added >= NUM_CYCLE_EDGES:
            break

        src = str(cycle_accounts[i])
        dst = str(cycle_accounts[(i + 1) % L])  # wrap
        ts_ = timestamps[i]
        amt_ = amts[i]

        row = {
            TS_COL: ts_,
            "From Bank": np.nan,
            SRC_COL: src,
            "To Bank": np.nan,
            DST_COL: dst,
            AMT_REC_COL: float(amt_),
            "Receiving Currency": "US Dollar",
            AMT_PAID_COL: float(amt_),
            "Payment Currency": "US Dollar",
            "Payment Format": "ACH",
            LABEL_COL: 1,
            "injected_motif": motif_name,
        }
        injected_rows.append(row)
        cycle_edges_added += 1

print(f"  -> Added {cycle_edges_added} CYCLE edges.")

# ===================== CONCAT & SAVE =====================

print(f"Total injected edges created: {len(injected_rows)}")
if len(injected_rows) == 0:
    raise RuntimeError("No injected rows were created – aborting save.")

inj_df = pd.DataFrame(injected_rows)

# ensure original has injected_motif column
if "injected_motif" not in df.columns:
    df["injected_motif"] = np.nan

combined = pd.concat([df, inj_df], ignore_index=True, sort=False)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
combined.to_csv(OUTPUT_PATH, index=False)
print(f"\Saved RAT-injected IBM dataset to: {OUTPUT_PATH}")
print(f"Original rows: {len(df):,}  |  New rows: {len(combined):,}")
