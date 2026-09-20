"""
build_slt_plus_structural_graph.py
---------------------------------------------------------------------------
Fixes a real asymmetry found in the primary 4-condition comparison (Sept 21
2026 review): rat_natural's edge_attr is baseline + motif_* + RAT_* (RAT's
own injector computes motif_fanin/fanout/chain/cycle, so the structural
features are already nested inside rat_natural). slt_natural's edge_attr is
baseline + SLT_* only -- it never got the motif_* columns at all. That means
"slt_natural vs structural_only" was never an apples-to-apples "theory vs.
structure" comparison the way "rat_natural vs structural_only" is: SLT's
weak/unstable showing in the primary results could reflect the theory not
helping, OR just SLT missing structural signal RAT and structural_only get
for free. This script closes that gap by building an SLT+structural
condition: baseline + motif_* + SLT_*, so it can be compared against
structural_only on equal footing (the same way rat_natural already can).

WHY SPLICING (NOT RECOMPUTING) IS SAFE
motif_fanin/fanout/chain/cycle are pure graph-topology features (computed
from which accounts transact with which, not from any theory's composite
score). The RAT-pristine and SLT-pristine graphs are built from the SAME
underlying transaction set (HI-Small_Trans), so their motif_* values are
identical for the same edge -- this script copies them from the already-
built RAT-pristine graph instead of re-deriving them inside slt_injector.py.
It verifies edge_index and y_edge are IDENTICAL between the two source
graphs before touching anything, and aborts rather than producing a
silently-misaligned graph if they aren't (same safety pattern as
build_traincorrected_graph.py).

Usage:
    python scripts/analysis/build_slt_plus_structural_graph.py \\
        --slt_source_dir graphs/HI-Small_Trans_SLT_pristine \\
        --rat_source_dir graphs/HI-Small_Trans_RAT_pristine \\
        --output_dir graphs/HI-Small_Trans_SLT_pristine_plus_structural
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Must match motif_graph_builder_static.py / rat_injector.py's motif columns.
MOTIF_COLS = ["motif_fanin", "motif_fanout", "motif_chain", "motif_cycle"]

# Files copied verbatim from the SLT source (node features, structure,
# labels, timestamps are all identical regardless of which theory's
# edge_attr we're building -- only edge_attr.pt / edge_attr_cols.json
# change).
STATIC_COPY_FILES = [
    "edge_index.pt", "x.pt", "timestamps.pt", "y_edge.pt", "y_node.pt",
    "node_mapping.json",
]


def load_cols(d):
    with open(os.path.join(d, "edge_attr_cols.json")) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(
        description="Build the SLT+structural (baseline + motif_* + SLT_*) "
                     "graph condition, splicing motif_* columns from the "
                     "RAT-pristine graph into the SLT-pristine graph."
    )
    ap.add_argument("--slt_source_dir", type=str, default=None,
                     help="Default: <root>/graphs/HI-Small_Trans_SLT_pristine")
    ap.add_argument("--rat_source_dir", type=str, default=None,
                     help="Default: <root>/graphs/HI-Small_Trans_RAT_pristine "
                          "(only used as the motif_* column source)")
    ap.add_argument("--output_dir", type=str, default=None,
                     help="Default: <root>/graphs/HI-Small_Trans_SLT_pristine_plus_structural")
    args = ap.parse_args()

    root = PROJECT_ROOT
    slt_dir = args.slt_source_dir or str(root / "graphs" / "HI-Small_Trans_SLT_pristine")
    rat_dir = args.rat_source_dir or str(root / "graphs" / "HI-Small_Trans_RAT_pristine")
    output_dir = args.output_dir or str(root / "graphs" / "HI-Small_Trans_SLT_pristine_plus_structural")

    for d, label in [(slt_dir, "SLT source"), (rat_dir, "RAT source (motif donor)")]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"{label} not found: {d}")

    print(f"SLT source (base):        {slt_dir}")
    print(f"RAT source (motif donor): {rat_dir}")
    print(f"Output:                   {output_dir}")

    # ------------------------------------------------------------------
    # Alignment checks -- abort rather than silently splicing mismatched
    # rows. Both graphs are built from the same underlying transaction set
    # so edge_index and y_edge must match exactly if row order is intact.
    # ------------------------------------------------------------------
    slt_edge_index = torch.load(os.path.join(slt_dir, "edge_index.pt"))
    rat_edge_index = torch.load(os.path.join(rat_dir, "edge_index.pt"))
    assert torch.equal(slt_edge_index, rat_edge_index), (
        "edge_index differs between the SLT and RAT pristine graphs -- row "
        "order or account mapping does not match. Aborting rather than "
        "producing a silently-misaligned graph. Rebuild both from the SAME "
        "underlying HI-Small_Trans.csv if this trips."
    )

    slt_y_edge = torch.load(os.path.join(slt_dir, "y_edge.pt"))
    rat_y_edge = torch.load(os.path.join(rat_dir, "y_edge.pt"))
    assert torch.equal(slt_y_edge, rat_y_edge), (
        "y_edge (labels) differ between the SLT and RAT pristine graphs -- "
        "row alignment is broken. Aborting."
    )
    print("[OK] SLT and RAT pristine graphs are row-aligned (edge_index, y_edge match).")

    # ------------------------------------------------------------------
    # Build the spliced edge_attr: SLT's own columns + motif_* from RAT.
    # ------------------------------------------------------------------
    slt_edge_attr = torch.load(os.path.join(slt_dir, "edge_attr.pt"))
    slt_cols = load_cols(slt_dir)

    if any(c in slt_cols for c in MOTIF_COLS):
        raise RuntimeError(
            f"SLT source already contains motif columns {MOTIF_COLS} -- "
            f"did you point --slt_source_dir at an already-augmented graph?"
        )

    rat_edge_attr = torch.load(os.path.join(rat_dir, "edge_attr.pt"))
    rat_cols = load_cols(rat_dir)
    for feat in MOTIF_COLS:
        if feat not in rat_cols:
            raise ValueError(f"'{feat}' not found in RAT source's edge_attr_cols.json")
    motif_idxs = [rat_cols.index(feat) for feat in MOTIF_COLS]
    motif_block = rat_edge_attr[:, motif_idxs]

    new_edge_attr = torch.cat([slt_edge_attr, motif_block], dim=1)
    new_cols = slt_cols + MOTIF_COLS

    print(f"Edge features: {len(slt_cols)} (SLT) + {len(MOTIF_COLS)} (motif) "
          f"= {len(new_cols)}")

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    torch.save(new_edge_attr, os.path.join(output_dir, "edge_attr.pt"))
    with open(os.path.join(output_dir, "edge_attr_cols.json"), "w") as f:
        json.dump(new_cols, f, indent=2)

    for f in STATIC_COPY_FILES:
        src = os.path.join(slt_dir, f)
        dst = os.path.join(output_dir, f)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"  [WARN] {f} not found in SLT source -- skipping")

    src_stats_path = os.path.join(slt_dir, "graph_stats.json")
    if os.path.exists(src_stats_path):
        with open(src_stats_path) as f:
            stats = json.load(f)
        stats["num_edge_features"] = len(new_cols)
        stats["has_motif"] = True
        stats["note"] = (
            "motif_fanin/fanout/chain/cycle spliced in from "
            f"{os.path.basename(rat_dir)} (identical underlying transactions) "
            "so SLT can be compared against structural_only on equal footing, "
            "the same way rat_natural already can."
        )
        with open(os.path.join(output_dir, "graph_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
    else:
        print("  [WARN] graph_stats.json not found in SLT source -- skipping")

    print(f"\nDONE. Next: generate splits with")
    print(f"  python scripts/create_splits.py --graph_folder {output_dir}")


if __name__ == "__main__":
    main()
