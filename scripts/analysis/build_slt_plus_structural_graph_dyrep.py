"""
build_slt_plus_structural_graph_dyrep.py
---------------------------------------------------------------------------
DyRep-Lite equivalent of build_slt_plus_structural_graph.py. Same fix, same
splicing approach, applied to the DyRep event-graph format (src.pt/dst.pt
instead of edge_index.pt). See that script's docstring for the full
rationale (RAT's own injector already bakes in motif_fanin/fanout/chain/
cycle so rat_natural is nested over structural_only; SLT never got motif at
all, so slt_natural vs structural_only wasn't apples-to-apples).

DyRep graphs carry no train/val/test splits by design -- generate them
separately with create_splits.py after running this script.

Usage:
    python scripts/analysis/build_slt_plus_structural_graph_dyrep.py \\
        --slt_source_dir graphs_dyrep/HI-Small_Trans_SLT_pristine \\
        --rat_source_dir graphs_dyrep/HI-Small_Trans_RAT_pristine \\
        --output_dir graphs_dyrep/HI-Small_Trans_SLT_pristine_plus_structural
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

MOTIF_COLS = ["motif_fanin", "motif_fanout", "motif_chain", "motif_cycle"]

# Files copied verbatim from the SLT source -- structure, timestamps,
# event types, node features, and labels don't depend on which theory's
# edge_attr we're building.
DYREP_COPY_FILES = [
    "src.pt", "dst.pt", "ts.pt", "event_type.pt", "node_features.pt",
    "labels.pt", "y_node.pt", "node_mapping.json",
]


def load_cols(d):
    with open(os.path.join(d, "edge_attr_cols.json")) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(
        description="Build the DyRep SLT+structural (baseline + motif_* + "
                     "SLT_*) graph condition, splicing motif_* columns from "
                     "the RAT-pristine DyRep graph into the SLT-pristine one."
    )
    ap.add_argument("--slt_source_dir", type=str, default=None,
                     help="Default: <root>/graphs_dyrep/HI-Small_Trans_SLT_pristine")
    ap.add_argument("--rat_source_dir", type=str, default=None,
                     help="Default: <root>/graphs_dyrep/HI-Small_Trans_RAT_pristine")
    ap.add_argument("--output_dir", type=str, default=None,
                     help="Default: <root>/graphs_dyrep/HI-Small_Trans_SLT_pristine_plus_structural")
    args = ap.parse_args()

    root = PROJECT_ROOT
    slt_dir = args.slt_source_dir or str(root / "graphs_dyrep" / "HI-Small_Trans_SLT_pristine")
    rat_dir = args.rat_source_dir or str(root / "graphs_dyrep" / "HI-Small_Trans_RAT_pristine")
    output_dir = args.output_dir or str(root / "graphs_dyrep" / "HI-Small_Trans_SLT_pristine_plus_structural")

    for d, label in [(slt_dir, "SLT source"), (rat_dir, "RAT source (motif donor)")]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"{label} not found: {d}")

    print(f"SLT source (base):        {slt_dir}")
    print(f"RAT source (motif donor): {rat_dir}")
    print(f"Output:                   {output_dir}")

    # ------------------------------------------------------------------
    # Alignment checks -- same reasoning as the static version: abort
    # rather than splice mismatched rows.
    # ------------------------------------------------------------------
    slt_struct = torch.stack([
        torch.load(os.path.join(slt_dir, "src.pt")),
        torch.load(os.path.join(slt_dir, "dst.pt")),
    ])
    rat_struct = torch.stack([
        torch.load(os.path.join(rat_dir, "src.pt")),
        torch.load(os.path.join(rat_dir, "dst.pt")),
    ])
    assert torch.equal(slt_struct, rat_struct), (
        "src/dst differ between the SLT and RAT pristine DyRep graphs -- row "
        "order or account mapping does not match. Aborting rather than "
        "producing a silently-misaligned graph."
    )

    slt_labels = torch.load(os.path.join(slt_dir, "labels.pt"))
    rat_labels = torch.load(os.path.join(rat_dir, "labels.pt"))
    assert torch.equal(slt_labels, rat_labels), (
        "labels differ between the SLT and RAT pristine DyRep graphs -- row "
        "alignment is broken. Aborting."
    )
    print("[OK] SLT and RAT pristine DyRep graphs are row-aligned (src/dst, labels match).")

    # ------------------------------------------------------------------
    # Build the spliced edge_attr
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

    for f in DYREP_COPY_FILES:
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
            "so SLT can be compared against structural_only on equal footing."
        )
        with open(os.path.join(output_dir, "graph_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
    else:
        print("  [WARN] graph_stats.json not found in SLT source -- skipping")

    print(f"\nDONE. Next: generate splits with")
    print(f"  python scripts/create_splits.py --graph_folder {output_dir}")


if __name__ == "__main__":
    main()
