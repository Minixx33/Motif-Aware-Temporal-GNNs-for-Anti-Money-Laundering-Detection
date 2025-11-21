import pandas as pd
import numpy as np
import random
from datetime import timedelta
import os
import shutil

# ---------- CONFIG ----------
INPUT_DIR = r"Datasets\baseline"   # <-- adjust if needed
OUTPUT_DIR = r"Datasets\baseline_RAT"
TX_FILE = "transactions.csv"
ACCT_FILE = "accounts.csv"

NUM_OFFENDERS = 200
FANIN_P = 0.3
CHAIN_P = 0.4
CYCLE_P = 0.3
FANIN_SIZE = (3, 8)
CHAIN_LEN = (3, 6)
CYCLE_LEN = (3, 5)
TIME_SPAN_SECONDS = 300
AMOUNT_MEAN = 3000
AMOUNT_STD = 400
# ----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === COLUMN NAMES for your dataset ===
SRC_COL = "orig_acct"
DST_COL = "bene_acct"
AMT_COL = "base_amt"
TS_COL  = "tran_timestamp"
LABEL_COL = "is_sar"

# === Load data ===
tx = pd.read_csv(os.path.join(INPUT_DIR, TX_FILE), low_memory=False)
accounts = pd.read_csv(os.path.join(INPUT_DIR, ACCT_FILE), low_memory=False)

# === Account ID column ===
ACCT_ID_COL = "acct_id"

# === Helper to parse timestamps ===
def parse_ts(val):
    try:
        if isinstance(val, (int, float, np.integer, np.floating)):
            return pd.to_datetime(val, unit="s")
        return pd.to_datetime(val)
    except Exception:
        return pd.to_datetime(val, errors="coerce")

# === Prepare lists ===
offenders = accounts.sample(min(NUM_OFFENDERS, len(accounts)))[ACCT_ID_COL].astype(str).tolist()
all_account_ids = accounts[ACCT_ID_COL].astype(str).tolist()

# === Prepare base timestamps ===
tx_ts = tx[TS_COL].apply(parse_ts).dropna()
base_ts_choices = tx_ts.sample(n=min(1000, len(tx_ts)), replace=False).tolist()

injected_rows = []

for off in offenders:
    r = random.random()
    t0 = random.choice(base_ts_choices)
    if r < FANIN_P:
        # Fan-in motif: many -> off
        k = random.randint(FANIN_SIZE[0], FANIN_SIZE[1])
        sources = random.sample(all_account_ids, k)
        span = timedelta(seconds=random.randint(1, TIME_SPAN_SECONDS))
        for i, src in enumerate(sources):
            ts = t0 + timedelta(seconds=random.randint(0, int(span.total_seconds())))
            amt = round(max(10, random.gauss(AMOUNT_MEAN, AMOUNT_STD)), 2)
            injected_rows.append({
                SRC_COL: src,
                DST_COL: str(off),
                AMT_COL: amt,
                TS_COL: ts.isoformat(),
                LABEL_COL: 1,
                "alert_id": "RAT_FANIN",
                "injected_motif": "fanin"
            })
    elif r < FANIN_P + CHAIN_P:
        # Chain motif: off -> n1 -> n2 -> ...
        L = random.randint(CHAIN_LEN[0], CHAIN_LEN[1])
        chain_nodes = [str(off)] + random.sample(all_account_ids, L)
        for i in range(len(chain_nodes) - 1):
            src = chain_nodes[i]
            dst = chain_nodes[i + 1]
            ts = t0 + timedelta(seconds=int(i * (TIME_SPAN_SECONDS / max(1, L))))
            amt = round(max(10, random.gauss(AMOUNT_MEAN, AMOUNT_STD)), 2)
            injected_rows.append({
                SRC_COL: src,
                DST_COL: dst,
                AMT_COL: amt,
                TS_COL: ts.isoformat(),
                LABEL_COL: 1,
                "alert_id": "RAT_CHAIN",
                "injected_motif": f"chain_L{L}"
            })
    else:
        # Cycle motif: A -> B -> C -> A
        L = random.randint(CYCLE_LEN[0], CYCLE_LEN[1])
        cycle_nodes = random.sample(all_account_ids, L)
        span = timedelta(seconds=random.randint(1, TIME_SPAN_SECONDS))
        for i in range(L):
            src = cycle_nodes[i]
            dst = cycle_nodes[(i + 1) % L]
            ts = t0 + timedelta(seconds=random.randint(0, int(span.total_seconds())))
            amt = round(max(10, random.gauss(AMOUNT_MEAN, AMOUNT_STD)), 2)
            injected_rows.append({
                SRC_COL: src,
                DST_COL: dst,
                AMT_COL: amt,
                TS_COL: ts.isoformat(),
                LABEL_COL: 1,
                "alert_id": "RAT_CYCLE",
                "injected_motif": f"cycle_L{L}"
            })

# === Create DataFrame and merge ===
injected_df = pd.DataFrame(injected_rows)

# Ensure all original columns exist
for c in tx.columns:
    if c not in injected_df.columns:
        injected_df[c] = np.nan

# Keep original order + new columns
final_injected = injected_df[tx.columns.tolist() + [c for c in injected_df.columns if c not in tx.columns]]

tx_with_injected = pd.concat([tx, final_injected], ignore_index=True)

# Save outputs
tx_with_injected.to_csv(os.path.join(OUTPUT_DIR, TX_FILE), index=False)

# Copy other CSVs
for f in os.listdir(INPUT_DIR):
    src_path = os.path.join(INPUT_DIR, f)
    dst_path = os.path.join(OUTPUT_DIR, f)

    # ✅ Skip directories and non-CSV files
    if os.path.isdir(src_path) or not f.lower().endswith(".csv"):
        continue

    if f == TX_FILE:
        continue  # already saved modified transactions.csv

    try:
        pd.read_csv(src_path, low_memory=False).to_csv(dst_path, index=False)
    except Exception:
        shutil.copy2(src_path, dst_path)

print(f"✅ Injected {len(injected_df)} motif edges. Saved to {OUTPUT_DIR}/{TX_FILE}")
