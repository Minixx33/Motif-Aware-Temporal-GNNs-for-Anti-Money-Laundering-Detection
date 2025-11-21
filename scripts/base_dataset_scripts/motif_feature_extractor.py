import pandas as pd
import numpy as np
from collections import defaultdict

# ========= CONFIG =========
TX_PATH = r"Datasets\baseline_RAT\transactions.csv"
OUT_PATH = r"Datasets\baseline_RAT\motif_features.csv"
# ==========================

print("Loading dataset...")
tx = pd.read_csv(TX_PATH, low_memory=False, dtype={"alert_id": str, "injected_motif": str})

SRC_COL = "orig_acct"
DST_COL = "bene_acct"
MOTIF_COL = "injected_motif"
AMT_COL = "base_amt"
LABEL_COL = "is_sar"

print(f"Loaded {len(tx)} transactions.")

# ========= 1. Basic structural features =========
print("Computing fan-in / fan-out degrees...")

fan_in = tx.groupby(DST_COL).size().rename("fan_in")
fan_out = tx.groupby(SRC_COL).size().rename("fan_out")

mean_in_amt = tx.groupby(DST_COL)[AMT_COL].mean().rename("mean_in_amt")
mean_out_amt = tx.groupby(SRC_COL)[AMT_COL].mean().rename("mean_out_amt")

# ========= 2. Motif participation counts =========
print("Computing motif participation counts...")

motif_role_counts = defaultdict(lambda: defaultdict(int))
tx_motif = tx[tx[MOTIF_COL].notna()].copy()

print(f"Counting motif participation for {len(tx_motif)} injected transactions...")

for _, row in tx_motif.iterrows():
    src = int(row[SRC_COL])
    dst = int(row[DST_COL])
    motif = str(row[MOTIF_COL]).strip()
    motif_role_counts[src][f"{motif}_as_src"] += 1
    motif_role_counts[dst][f"{motif}_as_dst"] += 1

# Convert dictionary to dataframe
motif_df = pd.DataFrame.from_dict(motif_role_counts, orient="index").fillna(0)
motif_df.index.name = "acct_id"
motif_df.reset_index(inplace=True)

print("Motif DF shape:", motif_df.shape)
print("Non-zero motif entries:", (motif_df.drop(columns=['acct_id']) > 0).sum().sum())
print("Sample motif rows:\n", motif_df.head())

# ========= 3. Merge all features =========
print("Merging features...")

accounts = pd.concat([fan_in, fan_out, mean_in_amt, mean_out_amt], axis=1).fillna(0)
accounts.index.name = "acct_id"
accounts.reset_index(inplace=True)

# Convert both to string before merging
accounts["acct_id"] = accounts["acct_id"].astype(np.int64)
motif_df["acct_id"] = motif_df["acct_id"].astype(np.int64)
print("Overlap in IDs:", len(set(accounts["acct_id"]) & set(motif_df["acct_id"])))

features = pd.merge(accounts, motif_df, on="acct_id", how="outer").fillna(0)

# ========= 4. Add summary metrics =========
features["total_tx"] = features["fan_in"] + features["fan_out"]
features["motif_participation"] = features[[c for c in features.columns if "_as_" in c]].sum(axis=1)
features["motif_ratio"] = features["motif_participation"] / (features["total_tx"] + 1e-6)

print("Saving features...")
features.to_csv(OUT_PATH, index=False)
print(f"✅ Saved motif-aware features to {OUT_PATH}")

print("\nTop 5 feature columns:")
print(features.head())
