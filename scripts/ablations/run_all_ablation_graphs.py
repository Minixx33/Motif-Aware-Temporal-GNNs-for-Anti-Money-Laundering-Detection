import argparse
import os
from pathlib import Path

from rat_ablation_groups import FULL_FEATURES, ABLATED_SETS, TOP20_FEATURES
from run_ablation import run_ablation

# ============================================================
# CLI ARGS
# ============================================================

_parser = argparse.ArgumentParser(description="Run all feature ablation graphs")
_parser.add_argument("--input_graph", type=str, default=None,
                     help="Path to the full DyRep graph directory "
                          "(default: <project_root>/graphs_dyrep/HI-Small_Trans_RAT_medium)")
_args, _ = _parser.parse_known_args()

# ============================================================
# CONFIG
# ============================================================

# Resolve project root so this script runs on Windows / Linux / macOS.
# scripts/ablations/run_all_ablation_graphs.py  →  parents[2] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FULL = _args.input_graph if _args.input_graph else str(PROJECT_ROOT / "graphs_dyrep" / "HI-Small_Trans_RAT_medium")

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
