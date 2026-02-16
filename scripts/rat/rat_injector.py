"""
RAT + Motif Feature Injection for IBM HI-Small (Multiplicative Version)
Fixed Version – No NaN Threshold, Correct Injection
"""

import os
import numpy as np
import pandas as pd

# ===================== CONFIG =====================

BASE_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets"

TX_CSV_PATH       = os.path.join(BASE_DIR, "HI-Medium_Trans.csv")
ACCOUNTS_CSV_PATH = os.path.join(BASE_DIR, "HI-Medium_accounts.csv")
PATTERNS_TXT_PATH = os.path.join(BASE_DIR, "HI-Medium_patterns.txt")  # optional

OUTPUT_DIR        = os.path.join(BASE_DIR, "RAT")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TS_COL      = "Timestamp"
SRC_COL     = "Account"
DST_COL     = "Account.1"
FROM_BANK   = "From Bank"
TO_BANK     = "To Bank"
AMT_PAID    = "Amount Paid"
AMT_REC     = "Amount Received"
LABEL_COL   = "Is Laundering"

ACCT_ID_COL      = "Account Number"
ACCT_ENTITY_ID   = "Entity ID"
ACCT_ENTITY_NAME = "Entity Name"

INTENSITIES = {"low": 0.15, "medium": 0.30, "high": 0.60}
EPS = 1e-8

# ===================== HELPERS =====================

def safe_zscore(x, mean, std):
    return (x - mean) / (std.replace(0, np.nan) + EPS)

def norm_by_quantile(series, q=0.95):
    s = series.astype(float)
    qv = s.quantile(q)
    if not np.isfinite(qv) or qv <= 0:
        qv = s.max()
    if not np.isfinite(qv) or qv <= 0:
        return pd.Series(0.0, index=series.index)
    out = (s / qv).replace([np.inf, -np.inf], np.nan).fillna(0)
    return out.clip(0, 1)

def clip_positive(series):
    return series.clip(lower=0.0)

# ===================== LOAD TRANSACTIONS =====================

print(f"Loading transactions from: {TX_CSV_PATH}")
df = pd.read_csv(TX_CSV_PATH, low_memory=False).reset_index(drop=True)
df[TS_COL] = pd.to_datetime(df[TS_COL], errors="raise")

df[AMT_PAID] = pd.to_numeric(df[AMT_PAID], errors="coerce", downcast="float")
df[AMT_REC]  = pd.to_numeric(df[AMT_REC],  errors="coerce", downcast="float")
df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce", downcast="integer").fillna(0).astype(np.int8)

df[SRC_COL] = df[SRC_COL].astype(str)
df[DST_COL] = df[DST_COL].astype(str)


# ===================== LOAD ACCOUNTS =====================

print(f"Loading accounts from: {ACCOUNTS_CSV_PATH}")
df_acct = pd.read_csv(ACCOUNTS_CSV_PATH, low_memory=False)

# Make join key consistent BEFORE setting index
df_acct[ACCT_ID_COL] = df_acct[ACCT_ID_COL].astype(str)
df_acct = df_acct.set_index(ACCT_ID_COL)

# Optional pattern accounts
pattern_accounts = set()
if os.path.exists(PATTERNS_TXT_PATH):
    with open(PATTERNS_TXT_PATH, "r") as f:
        for line in f:
            acc = line.strip()
            if acc:
                pattern_accounts.add(acc)
print(f"Pattern accounts loaded: {len(pattern_accounts)}")

# ===================== PER-ACCOUNT STATS =====================

src_group = df.groupby(SRC_COL)
dst_group = df.groupby(DST_COL)

df["src_out_degree"] = src_group[DST_COL].nunique().reindex(df[SRC_COL]).values
df["dst_in_degree"]  = dst_group[SRC_COL].nunique().reindex(df[DST_COL]).values

df["src_amt_mean"] = src_group[AMT_PAID].mean().reindex(df[SRC_COL]).values
df["src_amt_std"]  = src_group[AMT_PAID].std().reindex(df[SRC_COL]).values
df["dst_amt_mean"] = dst_group[AMT_REC].mean().reindex(df[DST_COL]).values
df["dst_amt_std"]  = dst_group[AMT_REC].std().reindex(df[DST_COL]).values

df["src_first_seen"] = src_group[TS_COL].min().reindex(df[SRC_COL]).values
df["dst_first_seen"] = dst_group[TS_COL].min().reindex(df[DST_COL]).values

df["src_age_days"] = (df[TS_COL] - df["src_first_seen"]).dt.total_seconds() / (3600*24)
df["dst_age_days"] = (df[TS_COL] - df["dst_first_seen"]).dt.total_seconds() / (3600*24)

df["src_age_days"] = df["src_age_days"].fillna(0)
df["dst_age_days"] = df["dst_age_days"].fillna(0)

# ===================== BURSTINESS =====================

df["date_only"] = df[TS_COL].dt.date
df["src_day_tx_count"] = df.groupby([SRC_COL,"date_only"])[AMT_PAID].transform("count")
df["dst_day_tx_count"] = df.groupby([DST_COL,"date_only"])[AMT_REC].transform("count")

# ===================== TIME CONTEXT =====================

df["hour"] = df[TS_COL].dt.hour
df["weekday"] = df[TS_COL].dt.weekday

df["RAT_is_off_hours"]  = ((df["hour"] < 8) | (df["hour"] > 18)).astype(int)
df["RAT_is_weekend"]    = (df["weekday"] >= 5).astype(int)
df["RAT_is_cross_bank"] = (df[FROM_BANK] != df[TO_BANK]).astype(int)

# ===================== AMOUNT Z-SCORES =====================

df["RAT_src_amount_z_pos"] = clip_positive(safe_zscore(df[AMT_PAID], df["src_amt_mean"], df["src_amt_std"]))
df["RAT_dst_amount_z_pos"] = clip_positive(safe_zscore(df[AMT_REC],  df["dst_amt_mean"], df["dst_amt_std"]))

# ===================== NORMALIZE STRUCTURAL =====================

df["RAT_src_out_deg_norm"] = norm_by_quantile(df["src_out_degree"].fillna(0))
df["RAT_dst_in_deg_norm"]  = norm_by_quantile(df["dst_in_degree"].fillna(0))
df["RAT_src_burst_norm"]   = norm_by_quantile(df["src_day_tx_count"].fillna(0))
df["RAT_dst_burst_norm"]   = norm_by_quantile(df["dst_day_tx_count"].fillna(0))
df["RAT_combined_burst"]   = norm_by_quantile(df["src_day_tx_count"].fillna(0) +
                                              df["dst_day_tx_count"].fillna(0)

)

# ===================== MERGE ENTITY INFO =====================

df = df.join(df_acct.add_prefix("srcacct_"), on=SRC_COL)
df = df.join(df_acct.add_prefix("dstacct_"), on=DST_COL)

df["src_entity_id"] = df["srcacct_" + ACCT_ENTITY_ID]
df["dst_entity_id"] = df["dstacct_" + ACCT_ENTITY_ID]

df["RAT_same_entity"] = (df["src_entity_id"].astype(str) == df["dst_entity_id"].astype(str)).astype(int)

entity_acct_count = df_acct.reset_index().groupby(ACCT_ENTITY_ID)[ACCT_ID_COL].nunique()
df["RAT_src_entity_accounts"] = df["src_entity_id"].map(entity_acct_count).fillna(1)
df["RAT_dst_entity_accounts"] = df["dst_entity_id"].map(entity_acct_count).fillna(1)

df["RAT_src_entity_acct_norm"] = norm_by_quantile(df["RAT_src_entity_accounts"])
df["RAT_dst_entity_acct_norm"] = norm_by_quantile(df["RAT_dst_entity_accounts"])

# ===================== PATTERN FLAGS =====================

df["RAT_src_pattern_flag"] = df[SRC_COL].astype(str).isin(pattern_accounts).astype(int)
df["RAT_dst_pattern_flag"] = df[DST_COL].astype(str).isin(pattern_accounts).astype(int)

# ===================== MUTUAL FLOW =====================

edge_counts = df.groupby([SRC_COL, DST_COL]).size().reset_index(name="count")
rev = edge_counts.rename(columns={SRC_COL:"DST_tmp", DST_COL:"SRC_tmp"})

mutual = edge_counts.merge(
    rev,
    left_on=[SRC_COL, DST_COL],
    right_on=["DST_tmp", "SRC_tmp"],
    how="inner"
)[[SRC_COL, DST_COL]].drop_duplicates()

mutual["RAT_mutual_flag"] = 1
df = df.merge(mutual, on=[SRC_COL, DST_COL], how="left")
df["RAT_mutual_flag"] = df["RAT_mutual_flag"].fillna(0)

# ===================== MOTIF FEATURES =====================

src_outdeg_by_acct = df.groupby(SRC_COL)[DST_COL].nunique()
df["dst_out_degree"] = df[DST_COL].map(src_outdeg_by_acct)

df["dst_out_deg_norm"]  = norm_by_quantile(df["dst_out_degree"].fillna(0))

df["motif_fanin"]   = df["RAT_dst_in_deg_norm"]
df["motif_fanout"]  = df["RAT_src_out_deg_norm"]
df["motif_chain"]   = np.sqrt(df["RAT_dst_in_deg_norm"] * df["dst_out_deg_norm"])

df["motif_cycle"] = (
    0.5 * df["RAT_mutual_flag"] +
    0.3 * df["RAT_same_entity"] +
    0.2 * df["RAT_combined_burst"]
)

# ===================== RAT SUB-SCORES =====================

df["RAT_offender_score"] = (
    0.30*df["RAT_src_amount_z_pos"] +
    0.20*df["RAT_src_out_deg_norm"] +
    0.20*df["RAT_src_burst_norm"] +
    0.10*df["RAT_is_off_hours"] +
    0.10*df["RAT_src_pattern_flag"] +
    0.10*df["RAT_src_entity_acct_norm"]
)

df["RAT_target_score"] = (
    0.35*df["RAT_dst_amount_z_pos"] +
    0.25*df["RAT_dst_in_deg_norm"] +
    0.15*(1 - norm_by_quantile(df["dst_age_days"].fillna(0))) +
    0.15*df["RAT_dst_entity_acct_norm"] +
    0.10*df["RAT_dst_pattern_flag"]
)

df["RAT_guardian_weakness_score"] = (
    0.30*df["RAT_is_off_hours"] +
    0.20*df["RAT_is_weekend"] +
    0.20*df["RAT_is_cross_bank"] +
    0.20*df["RAT_combined_burst"] +
    0.10*df["RAT_same_entity"]
)

# ===================== MULTIPLICATIVE RAT SCORE =====================

print("Computing multiplicative RAT score...")

df["RAT_score"] = (
    (df["RAT_offender_score"] + EPS) *
    (df["RAT_target_score"] + EPS) *
    (df["RAT_guardian_weakness_score"] + EPS)
) ** (1/3)

df["RAT_score"] = df["RAT_score"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0,1)

# ===================== CREATE INTENSITY DATASETS =====================

launder_mask = df[LABEL_COL] == 1

# IMPORTANT FIX: DROP NaN RAT SCORES
launder_scores = (
    df.loc[launder_mask, "RAT_score"]
      .replace([np.inf, -np.inf], np.nan)
      .dropna()
      .astype(float)
      .values
)

print(f"Valid laundering RAT_scores: {len(launder_scores)}")

if len(launder_scores) < 10:
    raise RuntimeError("Too few valid laundering RAT scores — check calculations.")

float_cols = [c for c in df.columns if c.startswith(("RAT_", "motif_")) or c.endswith("_norm")]
df[float_cols] = df[float_cols].astype(np.float32)

for name, frac in INTENSITIES.items():
    threshold = float(np.quantile(launder_scores, 1 - frac))
    print(f"{name}: threshold = {threshold:.4f}")

    df["RAT_injected"] = (
        (df[LABEL_COL] == 1) &
        (df["RAT_score"] >= threshold)
    ).astype(np.int8)

    df["RAT_intensity_level"] = (df["RAT_injected"] * {"low": 1, "medium": 2, "high": 3}[name]).astype(np.int8)

    out_path = os.path.join(OUTPUT_DIR, f"HI-Medium_Trans_RAT_{name}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} [{int(df['RAT_injected'].sum())} injected rows]")


print("DONE.")