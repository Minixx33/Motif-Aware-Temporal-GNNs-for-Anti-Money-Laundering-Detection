"""
build_structural_only_graph_dyrep.py
---------------------------------------------------------------------------
DyRep-Lite equivalent of build_structural_only_graph.py, for the 4-condition
primary comparison (base / base+structural / base+RAT-natural / base+SLT-natural)
review feedback Aug 4 2026 directive #4: "Build the dyrep equivalent for the
4 condition comparison."

"Structural" = the 12 baseline transaction edge features plus the 4 motif_*
graph-topology features only (fanin/fanout/chain/cycle) -- no RAT or SLT
composite theory scores. Same definition as the static (GraphSAGE/GraphSAGE-T)
condition, just applied to the DyRep event-graph format.

Source graph: graphs_dyrep/HI-Small_Trans_RAT_pristine/ (built by running
motif_dyrep_graph_builder.py --dataset RAT/HI-Small_Trans_RAT_pristine.csv).
That DyRep graph already has baseline edge features + RAT theory_cols
(including motif_fanin/fanout/chain/cycle, computed regardless of intensity).
This script subsets its edge_attr down to baseline_cols + motif_cols via
run_ablation() (scripts/ablations/run_ablation.py), the DyRep-format analog
of run_ablation_static() used by the static version of this script.

Column lists are kept in exact sync with scripts/graph/motif_dyrep_graph_builder.py's
baseline_df columns (BASELINE_COLS here matches it 1:1) and
scripts/analysis/build_structural_only_graph.py's MOTIF_COLS.

DyRep graphs carry no train/val/test splits (motif_dyrep_graph_builder.py
saves "NO SPLITS" by design -- splits live separately under splits_dyrep/).
run_ablation() copies train_idx.pt/val_idx.pt/test_idx.pt if present, but
they never are for DyRep graphs, so this is a harmless no-op. After running
this script, generate splits the normal way:
    python scripts/create_splits.py --graph_folder graphs_dyrep/HI-Small_Trans_RAT_pristine_structural_only

Usage:
    python scripts/analysis/build_structural_only_graph_dyrep.py \\
        --source_graph_dir graphs_dyrep/HI-Small_Trans_RAT_pristine \\
        --output_dir graphs_dyrep/HI-Small_Trans_RAT_pristine_structural_only
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ablations.run_ablation import run_ablation  # noqa: E402

# Must match motif_dyrep_graph_builder.py's `baseline_df` columns exactly
# (which are themselves kept in sync with motif_graph_builder_static.py's
# baseline_cols, per that file's comment).
BASELINE_COLS = [
    "log_amt_rec", "log_amt_paid",
    "same_bank", "same_currency",
    "hour_of_day", "day_of_week", "is_weekend",
    "ts_normalized", "log_time_since_src", "log_time_since_dst",
    "pf_code", "rc_code",
]

# The 4 graph-structural (motif) features. Computed identically regardless of
# theory/intensity, so any injected RAT DyRep graph carries them.
MOTIF_COLS = ["motif_fanin", "motif_fanout", "motif_chain", "motif_cycle"]


def main():
    parser = argparse.ArgumentParser(
        description="Build the DyRep base+structural (baseline + motif_*) "
                     "graph condition for the primary natural-features experiment."
    )
    parser.add_argument("--source_graph_dir", type=str, default=None,
                        help="Default: <root>/graphs_dyrep/HI-Small_Trans_RAT_pristine")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default: <root>/graphs_dyrep/HI-Small_Trans_RAT_pristine_structural_only")
    args = parser.parse_args()

    root = PROJECT_ROOT
    source_graph_dir = args.source_graph_dir or str(root / "graphs_dyrep" / "HI-Small_Trans_RAT_pristine")
    output_dir = args.output_dir or str(root / "graphs_dyrep" / "HI-Small_Trans_RAT_pristine_structural_only")

    keep_features = BASELINE_COLS + MOTIF_COLS

    print(f"Source graph:  {source_graph_dir}")
    print(f"Output graph:  {output_dir}")
    print(f"Keeping {len(keep_features)} features: {keep_features}")

    if not os.path.isdir(source_graph_dir):
        raise FileNotFoundError(
            f"{source_graph_dir} does not exist. Build it first with:\n"
            f"  python scripts/graph/motif_dyrep_graph_builder.py "
            f"--dataset RAT/HI-Small_Trans_RAT_pristine.csv"
        )

    run_ablation(source_graph_dir, output_dir, keep_features)
    print("DONE. Next: generate splits with")
    print(f"  python scripts/create_splits.py --graph_folder {output_dir}")


if __name__ == "__main__":
    main()
