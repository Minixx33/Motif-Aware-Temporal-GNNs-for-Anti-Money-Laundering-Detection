# run_ablation_static.py
#
# Static-graph equivalent of run_ablation.py (which targets DyRep graphs).
# Subsets edge_attr.pt by feature name; copies all other static graph files
# unchanged. Output is a sibling directory ready for create_splits.py and
# GraphSAGE-T training.

import os
import json
import torch
import shutil


def run_ablation_static(full_graph_dir, output_dir, keep_features):
    """
    full_graph_dir : original static graph (edge_index.pt format)
    output_dir     : new ablation graph directory
    keep_features  : list of edge feature names to retain
    """

    os.makedirs(output_dir, exist_ok=True)

    # Files copied without modification. graph_stats.json is intentionally
    # NOT in this list -- it's regenerated below so num_edge_features reflects
    # the ablated subset instead of the full source graph's column count.
    static_files = [
        "edge_index.pt",
        "x.pt",
        "timestamps.pt",
        "y_edge.pt",
        "y_node.pt",
        "node_mapping.json",
    ]

    for f in static_files:
        src = os.path.join(full_graph_dir, f)
        dst = os.path.join(output_dir, f)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"  [WARN] {f} not found in source — skipping")

    # Load full edge attribute matrix and feature names
    full_edge_attr = torch.load(os.path.join(full_graph_dir, "edge_attr.pt"))
    with open(os.path.join(full_graph_dir, "edge_attr_cols.json")) as f:
        full_cols = json.load(f)

    # Resolve column indices to keep
    idxs = []
    for feat in keep_features:
        if feat not in full_cols:
            raise ValueError(f"Feature '{feat}' not found in edge_attr_cols.json")
        idxs.append(full_cols.index(feat))

    # Subset and save
    new_edge_attr = full_edge_attr[:, idxs]
    torch.save(new_edge_attr, os.path.join(output_dir, "edge_attr.pt"))

    with open(os.path.join(output_dir, "edge_attr_cols.json"), "w") as f:
        json.dump(keep_features, f, indent=2)

    # Regenerate graph_stats.json: node/edge counts, label distribution, etc.
    # are unaffected by feature ablation, so carry those through from the
    # source stats, but num_edge_features must reflect the actual subset.
    src_stats_path = os.path.join(full_graph_dir, "graph_stats.json")
    if os.path.exists(src_stats_path):
        with open(src_stats_path) as f:
            stats = json.load(f)
        stats["num_edge_features"] = len(keep_features)
        with open(os.path.join(output_dir, "graph_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
    else:
        print("  [WARN] graph_stats.json not found in source — skipping")

    print(f"  Ablation graph created at: {output_dir}")
    print(f"  Edge features: {len(full_cols)} → {len(keep_features)}")
