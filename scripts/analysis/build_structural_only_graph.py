"""
build_structural_only_graph.py
---------------------------------------------------------------------------
Builds the "base + structural" condition for the primary natural-features
experiment (review feedback Aug 4 2026): base transaction features vs.
base+structural vs. base+RAT-natural vs. base+SLT-natural.

"Structural" here means the 4 motif_* features (fanin, fanout, chain, cycle)
-- graph-topology signals that don't depend on any criminology theory's
composite score -- layered on top of the same 12 baseline transaction
features used everywhere else in the pipeline.

Source graph: the pristine (never-boosted) RAT graph, built by running
motif_graph_builder_static.py on HI-Small_Trans_RAT_pristine.csv (the
--dump_pristine output of rat_injector.py). That graph already contains
baseline_cols + RAT theory_cols (which includes motif_*, since rat_injector
computes motif_fanin/fanout/chain/cycle regardless of intensity/boosting).
This script subsets its edge_attr down to baseline_cols + motif_cols only,
dropping every RAT_*-prefixed column, via run_ablation_static (same subsetting
utility already used for the paper's other feature ablations).

Column lists are kept in exact sync with scripts/graph/motif_graph_builder_static.py's
baseline_cols list and the motif_* columns rat_injector.py creates.

Usage:
    python scripts/analysis/build_structural_only_graph.py \\
        --source_graph_dir graphs/HI-Small_Trans_RAT_pristine \\
        --output_dir graphs/HI-Small_Trans_RAT_pristine_structural_only
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ablations.run_ablation_static import run_ablation_static  # noqa: E402

# Must match motif_graph_builder_static.py's `baseline_cols` exactly.
BASELINE_COLS = [
    "log_amt_rec", "log_amt_paid",
    "same_bank", "same_currency",
    "hour_of_day", "day_of_week", "is_weekend",
    "ts_normalized", "log_time_since_src", "log_time_since_dst",
    "pf_code", "rc_code",
]

# The 4 graph-structural (motif) features. Computed identically regardless of
# theory/intensity, so any injected RAT graph carries them.
MOTIF_COLS = ["motif_fanin", "motif_fanout", "motif_chain", "motif_cycle"]


def main():
    parser = argparse.ArgumentParser(
        description="Build the base+structural (baseline + motif_*) graph "
                     "condition for the primary natural-features experiment."
    )
    parser.add_argument("--source_graph_dir", type=str, default=None,
                        help="Default: <root>/graphs/HI-Small_Trans_RAT_pristine")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default: <root>/graphs/HI-Small_Trans_RAT_pristine_structural_only")
    args = parser.parse_args()

    root = PROJECT_ROOT
    source_graph_dir = args.source_graph_dir or str(root / "graphs" / "HI-Small_Trans_RAT_pristine")
    output_dir = args.output_dir or str(root / "graphs" / "HI-Small_Trans_RAT_pristine_structural_only")

    keep_features = BASELINE_COLS + MOTIF_COLS

    print(f"Source graph:  {source_graph_dir}")
    print(f"Output graph:  {output_dir}")
    print(f"Keeping {len(keep_features)} features: {keep_features}")

    if not os.path.isdir(source_graph_dir):
        raise FileNotFoundError(
            f"{source_graph_dir} does not exist. Build it first with:\n"
            f"  python scripts/graph/motif_graph_builder_static.py "
            f"--dataset RAT/HI-Small_Trans_RAT_pristine.csv"
        )

    run_ablation_static(source_graph_dir, output_dir, keep_features)
    print("DONE.")


if __name__ == "__main__":
    main()
