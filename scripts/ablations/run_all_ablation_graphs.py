from rat_ablation_groups import FULL_FEATURES, ABLATED_SETS, TOP20_FEATURES
from run_ablation import run_ablation
import os
from pathlib import Path

# Resolve project root so this script runs on Windows / Linux / macOS.
# scripts/ablations/run_all_ablation_graphs.py  →  parents[2] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FULL = str(PROJECT_ROOT / "graphs_dyrep" / "HI-Small_Trans_RAT_medium")

def main():
    for name, removed in ABLATED_SETS.items():

        print(f"\n=== Building ablation: {name} ===")

        if name == "top20_features":
            # SPECIAL CASE: keep ONLY these features
            keep = TOP20_FEATURES
        else:
            # Default case: remove these features
            keep = [f for f in FULL_FEATURES if f not in removed]

        out_dir = f"{FULL}__{name}"
        run_ablation(FULL, out_dir, keep)

if __name__ == "__main__":
    main()
