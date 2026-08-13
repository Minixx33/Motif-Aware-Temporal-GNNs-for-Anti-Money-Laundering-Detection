"""
fix_node_degree_leakage.py
---------------------------------------------------------------------------
Design-review finding (Aug 13 2026 code sweep, before running the primary
comparison): every graph builder --
  scripts/graph/motif_graph_builder_static.py
  scripts/graph/baseline_graph_builder.py
  scripts/graph/motif_dyrep_graph_builder.py
  scripts/graph/baseline_dyrep_graph_builder.py
-- computes each account's out_degree / in_degree / total_degree (+ log
versions) via a single groupby over the ENTIRE edge list: all transactions,
all splits, past and future. These become static per-node input features
(x.pt for static graphs, node_features.pt for DyRep), used identically for
every edge regardless of which split that edge is in.

That means a transaction scored during training can "see" its
counterparty's FINAL degree -- computed using transactions from the
val/test period. This is a structural leakage channel distinct from (and
NOT fixed by) train_graphsage.py/train_graphsage_t.py's train-only
message-passing restriction, which only restricts which EDGES are
aggregated over inside encode() -- it doesn't touch the node input feature
matrix itself.

This script closes that channel. It reads the graph's already-built edge
list and the already-computed split assignment (train_edge_idx.pt from
create_splits.py), recomputes out_degree/in_degree/total_degree/log_* using
ONLY the src/dst endpoints of TRAIN-split edges, and overwrites those 6
leading columns of the node feature matrix in place. Val/test nodes get
their train-period degree, not their final degree -- the correct
point-in-time semantics (what you could actually have known at decision
time). The remaining columns (entity-type one-hot dummies) are left
untouched: an account's business-entity classification isn't a temporally
accumulating quantity, so it isn't part of this leakage channel.

Works for both graph formats, auto-detected:
  - Static (GraphSAGE/GraphSAGE-T): edge_index.pt + x.pt
  - DyRep (temporal):                src.pt + dst.pt + node_features.pt

Idempotent: always recomputes degree from the raw edge list + split file,
never from the (possibly already-corrected) existing degree columns, so
running it twice gives the same result as running it once. A one-time
backup of the original node feature file is kept alongside it
(<file>.pre_degree_fix_backup.pt) unless --no_backup is passed.

Usage:
    python scripts/analysis/fix_node_degree_leakage.py --graph_dir graphs/HI-Small_Trans_RAT_pristine
    python scripts/analysis/fix_node_degree_leakage.py --graph_dir graphs_dyrep/HI-Small_Trans_RAT_pristine

    # splits_dir is auto-derived (graphs/X -> splits/X, graphs_dyrep/X ->
    # splits_dyrep/X) but can be overridden, e.g. for the _chrono variant:
    python scripts/analysis/fix_node_degree_leakage.py \\
        --graph_dir graphs/HI-Small_Trans_chrono --splits_dir splits/HI-Small_Trans_chrono

Run this AFTER create_splits.py has produced train_edge_idx.pt for the
graph, and BEFORE training on it.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Must match the fixed leading-column order every graph builder listed above
# uses when it builds node_feat_df: degree columns first, then entity-type
# one-hot dummies. Verified against all 4 builders' source.
DEGREE_COLS = [
    "out_degree", "in_degree", "total_degree",
    "log_out_degree", "log_in_degree", "log_total_degree",
]
N_DEGREE_COLS = len(DEGREE_COLS)


def _default_splits_dir(graph_dir: Path) -> Path:
    graph_dir = graph_dir.resolve()
    graphs_root_name = graph_dir.parent.name  # "graphs" or "graphs_dyrep"
    splits_root_name = "splits_dyrep" if graphs_root_name == "graphs_dyrep" else "splits"
    return graph_dir.parent.parent / splits_root_name / graph_dir.name


def _detect_format(graph_dir: Path) -> str:
    files = set(os.listdir(graph_dir))
    if "edge_index.pt" in files and "x.pt" in files:
        return "static"
    if "src.pt" in files and "dst.pt" in files and "node_features.pt" in files:
        return "dyrep"
    raise ValueError(
        f"{graph_dir}: cannot detect graph format. Expected either "
        f"edge_index.pt + x.pt (static) or src.pt + dst.pt + node_features.pt (DyRep)."
    )


def _compute_degree_block(src_train: np.ndarray, dst_train: np.ndarray, num_nodes: int) -> np.ndarray:
    out_deg = np.bincount(src_train, minlength=num_nodes).astype(np.float64)
    in_deg = np.bincount(dst_train, minlength=num_nodes).astype(np.float64)
    total_deg = out_deg + in_deg
    block = np.stack([
        out_deg, in_deg, total_deg,
        np.log1p(out_deg), np.log1p(in_deg), np.log1p(total_deg),
    ], axis=1).astype(np.float32)
    return block


def _backup(path: Path, do_backup: bool):
    if not do_backup:
        return
    bak = path.with_name(path.name + ".pre_degree_fix_backup.pt")
    if not bak.exists():
        shutil.copy(path, bak)
        print(f"  Backup saved: {bak.name}")
    else:
        print(f"  Backup already exists, leaving as-is: {bak.name}")


def fix_node_degree_leakage(graph_dir: str, splits_dir: str = None, backup: bool = True):
    graph_dir = Path(graph_dir)
    if not graph_dir.is_dir():
        raise FileNotFoundError(f"{graph_dir} does not exist.")

    splits_dir = Path(splits_dir) if splits_dir else _default_splits_dir(graph_dir)
    train_idx_path = splits_dir / "train_edge_idx.pt"
    if not train_idx_path.exists():
        raise FileNotFoundError(
            f"{train_idx_path} not found. Run create_splits.py on this graph first:\n"
            f"  python scripts/create_splits.py --graph_folder {graph_dir}"
        )

    fmt = _detect_format(graph_dir)
    print(f"Graph:  {graph_dir}  (format={fmt})")
    print(f"Splits: {splits_dir}")

    train_idx = torch.load(train_idx_path).numpy()

    if fmt == "static":
        edge_index = torch.load(graph_dir / "edge_index.pt")
        x = torch.load(graph_dir / "x.pt")
        num_nodes = x.shape[0]
        num_node_feats = x.shape[1]
        src_train = edge_index[0].numpy()[train_idx]
        dst_train = edge_index[1].numpy()[train_idx]
        node_feat_path = graph_dir / "x.pt"
        x_arr = x.numpy().copy()
    else:
        src_all = torch.load(graph_dir / "src.pt")
        dst_all = torch.load(graph_dir / "dst.pt")
        node_features = torch.load(graph_dir / "node_features.pt")
        num_nodes = node_features.shape[0]
        num_node_feats = node_features.shape[1]
        src_train = src_all.numpy()[train_idx]
        dst_train = dst_all.numpy()[train_idx]
        node_feat_path = graph_dir / "node_features.pt"
        x_arr = node_features.numpy().copy()

    if num_node_feats < N_DEGREE_COLS:
        raise ValueError(
            f"{node_feat_path} has only {num_node_feats} columns, expected at least "
            f"{N_DEGREE_COLS} (degree columns). Refusing to touch it -- this graph "
            f"doesn't look like it was built by one of the 4 builders this script targets."
        )

    print(f"Num nodes: {num_nodes:,}  |  train edges: {len(train_idx):,} / "
          f"total edges available in graph")

    old_block = x_arr[:, :N_DEGREE_COLS].copy()
    new_block = _compute_degree_block(src_train, dst_train, num_nodes)

    changed = int((old_block[:, 2] != new_block[:, 2]).sum())  # total_degree column
    print(f"total_degree changed for {changed:,} / {num_nodes:,} nodes "
          f"(nodes whose train-only degree differs from full-dataset degree)")
    print(f"  old total_degree: mean={old_block[:,2].mean():.3f} max={old_block[:,2].max():.0f}")
    print(f"  new total_degree: mean={new_block[:,2].mean():.3f} max={new_block[:,2].max():.0f}")

    _backup(node_feat_path, backup)

    x_arr[:, :N_DEGREE_COLS] = new_block
    torch.save(torch.tensor(x_arr, dtype=torch.float32), node_feat_path)
    print(f"  Overwrote {node_feat_path.name} in place with train-only degree columns.")

    meta = {
        "fix": "node_degree_leakage",
        "degree_cols": DEGREE_COLS,
        "num_nodes": int(num_nodes),
        "num_train_edges": int(len(train_idx)),
        "nodes_with_changed_total_degree": changed,
    }
    with open(graph_dir / "node_degree_fix.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("DONE.")


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild node degree features (out/in/total degree + log "
                     "versions) using only train-split edges, closing a "
                     "full-dataset-degree leakage channel present in every "
                     "graph builder."
    )
    parser.add_argument("--graph_dir", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, default=None,
                        help="Default: sibling splits/ (or splits_dyrep/) dir "
                             "with the same basename as --graph_dir.")
    parser.add_argument("--no_backup", action="store_true",
                        help="Skip saving a *.pre_degree_fix_backup.pt copy "
                             "of the original node feature file.")
    args = parser.parse_args()

    fix_node_degree_leakage(args.graph_dir, args.splits_dir, backup=not args.no_backup)


if __name__ == "__main__":
    main()
