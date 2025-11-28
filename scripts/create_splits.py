"""
create_splits.py (DyRep-patched version)
-------------------------------------------------
Supports:

STATIC GRAPHS:
    - GraphSAGE
    - GraphSAGE-T
TEMPORAL GRAPHS:
    - TGAT
    - DyRep  (using src.pt, dst.pt, ts.pt, labels.pt)

Static → stratified random 60/20/20
Temporal → chronological 60/20/20
"""

import os
import json
import torch
import numpy as np
import warnings
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# AUTO-DETECT GRAPH TYPE (patched to support DyRep)
# ------------------------------------------------------------
def detect_graph_type(folder):
    files = os.listdir(folder)

    # NEW: DyRep event graph format
    if "src.pt" in files and "dst.pt" in files and "ts.pt" in files:
        print("Detected DyRep event graph format")
        return "temporal"

    # TGAT format
    if "src_nodes.pt" in files and "dst_nodes.pt" in files:
        print("Detected TGAT event graph format")
        return "temporal"

    # Static graph (GraphSAGE / GraphSAGE-T)
    if "edge_index.pt" in files:
        return "static"

    raise ValueError(
        f" Cannot detect graph type in {folder}.\n"
        f"   Expected one of:\n"
        f"     - DyRep: src.pt, dst.pt, ts.pt\n"
        f"     - TGAT:  src_nodes.pt, dst_nodes.pt\n"
        f"     - Static: edge_index.pt\n"
    )


# ------------------------------------------------------------
# STATIC GRAPH SPLIT (GraphSAGE / GraphSAGE-T)
# ------------------------------------------------------------
def split_static_graph(folder, out_dir,
                       train_ratio=0.60, val_ratio=0.20, test_ratio=0.20,
                       random_seed=42):

    print("\n" + "=" * 70)
    print("STATIC GRAPH DETECTED (GraphSAGE / GraphSAGE-T)")
    print("=" * 70)

    edge_index = torch.load(os.path.join(folder, "edge_index.pt"))
    y_edge = torch.load(os.path.join(folder, "y_edge.pt"))
    x = torch.load(os.path.join(folder, "x.pt"))

    num_nodes = x.shape[0]
    num_edges = edge_index.shape[1]
    all_idx = np.arange(num_edges)

    # Stratified split
    try:
        train_idx, temp_idx = train_test_split(
            all_idx,
            test_size=(val_ratio + test_ratio),
            stratify=y_edge.numpy(),
            random_state=random_seed
        )

        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            stratify=y_edge.numpy()[temp_idx],
            random_state=random_seed
        )

    except Exception as e:
        warnings.warn(
            f"Stratified split failed: {e}\n"
            f"Falling back to random split."
        )
        train_idx, temp_idx = train_test_split(
            all_idx,
            test_size=(val_ratio + test_ratio),
            random_state=random_seed
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            random_state=random_seed
        )

    # Node masks (nodes touched by edges)
    train_nodes = torch.zeros(num_nodes, dtype=torch.bool)
    val_nodes = torch.zeros(num_nodes, dtype=torch.bool)
    test_nodes = torch.zeros(num_nodes, dtype=torch.bool)

    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    for i in train_idx:
        train_nodes[src[i]] = True
        train_nodes[dst[i]] = True
    for i in val_idx:
        val_nodes[src[i]] = True
        val_nodes[dst[i]] = True
    for i in test_idx:
        test_nodes[src[i]] = True
        test_nodes[dst[i]] = True

    # Save
    os.makedirs(out_dir, exist_ok=True)
    torch.save(torch.tensor(train_idx), os.path.join(out_dir, "train_edge_idx.pt"))
    torch.save(torch.tensor(val_idx), os.path.join(out_dir, "val_edge_idx.pt"))
    torch.save(torch.tensor(test_idx), os.path.join(out_dir, "test_edge_idx.pt"))
    torch.save(train_nodes, os.path.join(out_dir, "train_node_mask.pt"))
    torch.save(val_nodes, os.path.join(out_dir, "val_node_mask.pt"))
    torch.save(test_nodes, os.path.join(out_dir, "test_node_mask.pt"))

    meta = {
        "type": "static",
        "split_method": "stratified_random",
        "num_edges": int(num_edges),
        "num_nodes": int(num_nodes)
    }
    with open(os.path.join(out_dir, "split_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✓ Static splits saved → {out_dir}")


# ------------------------------------------------------------
# TEMPORAL GRAPH SPLIT (TGAT + DyRep)
# ------------------------------------------------------------
def split_temporal_graph(folder, out_dir,
                         train_ratio=0.60, val_ratio=0.20, test_ratio=0.20):

    print("\n" + "=" * 70)
    print("TEMPORAL GRAPH DETECTED (TGAT / DyRep)")
    print("=" * 70)

    # Support both TGAT and DyRep formats
    # timestamps
    ts_file = "timestamps.pt" if "timestamps.pt" in os.listdir(folder) else "ts.pt"
    timestamps = torch.load(os.path.join(folder, ts_file))
    timestamps_np = timestamps.numpy()

    # edge labels
    label_file = "labels.pt" if "labels.pt" in os.listdir(folder) else "y_edge.pt"
    y_edge = torch.load(os.path.join(folder, label_file))

    num_events = len(timestamps_np)

    # Ensure sorted
    if not np.all(timestamps_np[:-1] <= timestamps_np[1:]):
        print("⚠️ WARNING: timestamps not sorted — sorting now")
        order = np.argsort(timestamps_np)
    else:
        order = np.arange(num_events)

    # Chronological split
    train_end = int(train_ratio * num_events)
    val_end = int((train_ratio + val_ratio) * num_events)

    train_idx = order[:train_end]
    val_idx = order[train_end:val_end]
    test_idx = order[val_end:]

    # Save
    os.makedirs(out_dir, exist_ok=True)
    torch.save(torch.tensor(train_idx), os.path.join(out_dir, "train_edge_idx.pt"))
    torch.save(torch.tensor(val_idx), os.path.join(out_dir, "val_edge_idx.pt"))
    torch.save(torch.tensor(test_idx), os.path.join(out_dir, "test_edge_idx.pt"))

    meta = {
        "type": "temporal",
        "split_method": "chronological",
        "num_events": int(num_events),
        "ts_file": ts_file,
        "label_file": label_file
    }
    with open(os.path.join(out_dir, "split_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✓ Temporal splits saved → {out_dir}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def create_splits(graph_folder, out_dir=None,
                  train_ratio=0.60, val_ratio=0.20, test_ratio=0.20,
                  random_seed=42):

    dataset_name = os.path.basename(graph_folder)

    if out_dir is None:
        base = os.path.join(os.path.dirname(graph_folder), "..", "splits")
        out_dir = os.path.join(base, dataset_name)

    out_dir = os.path.abspath(out_dir)

    print("\n" + "=" * 70)
    print(f"CREATING SPLITS FOR: {dataset_name}")
    print("=" * 70)

    graph_type = detect_graph_type(graph_folder)

    if graph_type == "static":
        split_static_graph(
            graph_folder, out_dir,
            train_ratio, val_ratio, test_ratio, random_seed
        )
    else:
        split_temporal_graph(
            graph_folder, out_dir,
            train_ratio, val_ratio, test_ratio
        )


# ------------------------------------------------------------
# EXECUTE
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create train/val/test splits")

    parser.add_argument("--graph_folder", type=str, help="Graph directory")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.60)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--test_ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    create_splits(
        args.graph_folder,
        args.out_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
