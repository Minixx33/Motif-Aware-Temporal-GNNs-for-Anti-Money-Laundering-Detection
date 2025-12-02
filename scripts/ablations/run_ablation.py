# run_ablation.py

import os
import json
import torch
import shutil

def run_ablation(full_graph_dir, output_dir, keep_features):
    """
    full_graph_dir: original DyRep graph
    output_dir: new ablation graph directory
    keep_features: list of feature names to retain
    """

    os.makedirs(output_dir, exist_ok=True)

    # Files to copy directly without modification
    static_files = [
        "src.pt", "dst.pt", "ts.pt", "event_type.pt",
        "node_features.pt", "labels.pt", "y_node.pt",
        "node_mapping.json", "graph_stats.json",
        "train_idx.pt", "val_idx.pt", "test_idx.pt"  # IMPORTANT
    ]

    for f in static_files:
        src = os.path.join(full_graph_dir, f)
        dst = os.path.join(output_dir, f)
        if os.path.exists(src):
            shutil.copy(src, dst)

    # Load full edge attribute matrix and feature names
    full_edge_attr = torch.load(os.path.join(full_graph_dir, "edge_attr.pt"))
    with open(os.path.join(full_graph_dir, "edge_attr_cols.json")) as f:
        full_cols = json.load(f)

    # Determine column indices to keep
    idxs = []
    for feat in keep_features:
        if feat not in full_cols:
            raise ValueError(f"Feature '{feat}' not found in edge_attr_cols.json")
        idxs.append(full_cols.index(feat))

    # Subset features
    new_edge_attr = full_edge_attr[:, idxs]

    # Save updated files
    torch.save(new_edge_attr, os.path.join(output_dir, "edge_attr.pt"))

    with open(os.path.join(output_dir, "edge_attr_cols.json"), "w") as f:
        json.dump(keep_features, f, indent=2)

    print(f" Ablation graph created at: {output_dir}")
