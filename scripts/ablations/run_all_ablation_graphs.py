from rat_ablation_groups import FULL_FEATURES, ABLATED_SETS, TOP20_FEATURES
from run_ablation import run_ablation
import os

FULL = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\graphs_dyrep\HI-Small_Trans_RAT_medium"

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
