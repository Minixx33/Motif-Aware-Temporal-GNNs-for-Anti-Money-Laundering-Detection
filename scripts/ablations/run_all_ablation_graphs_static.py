import argparse
import os
from pathlib import Path

from rat_ablation_groups import FULL_FEATURES, ABLATED_SETS, TOP20_FEATURES
from run_ablation_static import run_ablation_static

# ============================================================
# CLI ARGS
# ============================================================

_parser = argparse.ArgumentParser(
    description="Build all RAT feature-ablation graphs for static (GraphSAGE-T) format"
)
_parser.add_argument(
    "--input_graph", type=str, default=None,
    help="Path to the full static graph directory "
         "(default: <project_root>/graphs/HI-Small_Trans_RAT_medium)"
)
_args, _ = _parser.parse_known_args()

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FULL = _args.input_graph if _args.input_graph else str(
    PROJECT_ROOT / "graphs" / "HI-Small_Trans_RAT_medium"
)

# ============================================================
# RUN ALL ABLATIONS
# ============================================================

def main():
    print(f"Source graph : {FULL}")
    print(f"Ablations    : {list(ABLATED_SETS.keys())}\n")

    for name, removed in ABLATED_SETS.items():
        print(f"=== Building ablation: {name} ===")

        if name == "top20_features":
            keep = TOP20_FEATURES
        else:
            keep = [f for f in FULL_FEATURES if f not in removed]

        out_dir = f"{FULL}__{name}"
        run_ablation_static(FULL, out_dir, keep)
        print()

    print("Done. Run create_splits.py on each ablation folder before training.")


if __name__ == "__main__":
    main()
