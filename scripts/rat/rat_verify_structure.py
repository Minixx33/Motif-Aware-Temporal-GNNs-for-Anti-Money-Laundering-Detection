"""
Usage:
    python verify_structure.py low
    python verify_structure.py medium
    python verify_structure.py high
"""

import pandas as pd
import numpy as np
import sys, os

# --------------------------
# CONFIG
# --------------------------
BASE_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets\RAT"   # <<< EDIT THIS ONCE

if len(sys.argv) < 2:
    raise SystemExit("Please specify: low | medium | high")

intensity = sys.argv[1]
if intensity not in {"low", "medium", "high"}:
    raise SystemExit("Argument must be one of: low, medium, high")

FILE_PATH = os.path.join(BASE_DIR, f"HI-Small_Trans_RAT_{intensity}.csv")

print(f"\n=== STRUCTURAL VERIFICATION: {FILE_PATH} ===\n")

df = pd.read_csv(FILE_PATH, low_memory=False)

EXPECTED_COLUMNS = [...]  # paste same list as before

missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
extra_cols   = set(df.columns) - set(EXPECTED_COLUMNS)

print(f"Rows: {len(df):,}")
print(f"Missing columns: {missing_cols}")
print(f"Extra columns:   {extra_cols}")

# Timestamp validation
try:
    pd.to_datetime(df["Timestamp"], errors="raise")
    print("Timestamp format OK.")
except:
    print("ERROR: invalid timestamps!")

# Injection consistency
launder_rows = df[df["Is Laundering"] == 1]
injected = df[df["RAT_injected"] == 1]

print(f"Laundering rows: {len(launder_rows):,}")
print(f"Injected rows (RAT_injected=1): {len(injected):,}")

print("\nIntensity level counts:")
print(df["RAT_intensity_level"].value_counts())

print("\nDONE.")
