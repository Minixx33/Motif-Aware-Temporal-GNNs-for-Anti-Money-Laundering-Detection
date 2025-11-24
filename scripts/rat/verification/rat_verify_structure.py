"""
verify_rat_multiplicative.py
Usage:
    python verify_rat_multiplicative.py
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets"
RAT_DIR  = os.path.join(BASE_DIR, "RAT")  # or "RAT" if you kept old name

LABEL_COL = "Is Laundering"

RAT_FILES = {
    "low":    "HI-Small_Trans_RAT_low.csv",
    "medium": "HI-Small_Trans_RAT_medium.csv",
    "high":   "HI-Small_Trans_RAT_high.csv",
}

def verify_file(intensity, filename):
    path = os.path.join(RAT_DIR, filename)
    print(f"\n=== [{intensity}] {path} ===")
    df = pd.read_csv(path, low_memory=False)

    required_cols = [
        "RAT_offender_score",
        "RAT_target_score",
        "RAT_guardian_weakness_score",
        "RAT_score",
        "RAT_injected",
        "RAT_intensity_level",
        LABEL_COL,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return

    n_rows = len(df)
    print(f"Rows: {n_rows:,}")

    # Overall RAT_score validity
    valid_rat = df["RAT_score"].notna().sum()
    nan_rat   = df["RAT_score"].isna().sum()
    print(f"RAT_score valid: {valid_rat:,} | NaN: {nan_rat:,}")

    # Laundering subset
    launder = df[df[LABEL_COL] == 1]
    print(f"Laundering rows: {len(launder):,}")

    if len(launder) > 0:
        ls = launder["RAT_score"].dropna()
        print("\nRAT_score (laundering) stats:")
        print(ls.describe())
    else:
        print("⚠ No laundering rows found in this file.")

    # Injection stats
    injected = df[df["RAT_injected"] == 1]
    print(f"\nInjected rows (RAT_injected=1): {len(injected):,}")
    print("RAT_intensity_level value_counts:\n", df["RAT_intensity_level"].value_counts())

    # Sanity: injected subset should be laundering only
    non_launder_injected = injected[injected[LABEL_COL] == 0]
    if len(non_launder_injected) > 0:
        print(f"⚠ WARNING: {len(non_launder_injected)} injected rows are non-laundering.")
    else:
        print("✔ All injected rows are laundering cases (good).")

def main():
    for name, fname in RAT_FILES.items():
        verify_file(name, fname)

if __name__ == "__main__":
    main()
