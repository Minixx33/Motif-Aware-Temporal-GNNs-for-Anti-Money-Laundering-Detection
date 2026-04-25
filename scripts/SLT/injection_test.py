from pathlib import Path

import pandas as pd

# scripts/SLT/injection_test.py  →  parents[2] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
path = str(PROJECT_ROOT / "ibm_transcations_datasets" / "SLT" / "HI-Small_Trans_SLT_high.csv")

df = pd.read_csv(path)

launder = df[df["Is Laundering"] == 1][["SLT_score", "SLT_injected"]].copy()

print("Laundering rows:", len(launder))
print("Injected rows:", int(launder["SLT_injected"].sum()))
print(launder["SLT_score"].describe())
print("Unique scores:", sorted(launder["SLT_score"].unique()))