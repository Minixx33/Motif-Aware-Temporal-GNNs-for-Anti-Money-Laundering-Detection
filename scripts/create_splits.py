"""
create_splits.py (Enhanced Version)
-------------------------------------------------
Auto-detect graph format (static vs TGAT) and 
produce 60/20/20 splits for ALL datasets.

Static graph = GraphSAGE / GraphSAGE-T (stratified random split)
Temporal graph = TGAT (chronological split)

Outputs (in splits/<dataset_name>/):

STATIC (GraphSAGE/GraphSAGE-T):
  - train_edge_idx.pt      [60% of edges]
  - val_edge_idx.pt        [20% of edges]
  - test_edge_idx.pt       [20% of edges]
  - train_node_mask.pt     [nodes touched by train edges]
  - val_node_mask.pt       [nodes touched by val edges]
  - test_node_mask.pt      [nodes touched by test edges]
  - split_metadata.json

TEMPORAL (TGAT):
  - train_edge_idx.pt      [first 60% chronologically]
  - val_edge_idx.pt        [next 20%]
  - test_edge_idx.pt       [last 20%]
  - split_metadata.json

Enhancements:
  - Label distribution per split
  - Temporal leakage detection
  - Validation checks
  - More detailed metadata
  - Custom output directory support
"""

import os
import json
import torch
import numpy as np
import warnings
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# AUTO-DETECT GRAPH TYPE
# ------------------------------------------------------------
def detect_graph_type(folder):
    """
    Detect if graph is static (GraphSAGE) or temporal (TGAT).
    
    Returns:
        str: "temporal" or "static"
    """
    files = os.listdir(folder)
    
    # TGAT format has separate src/dst node files
    if "src_nodes.pt" in files and "dst_nodes.pt" in files:
        return "temporal"
    # GraphSAGE format has edge_index
    elif "edge_index.pt" in files:
        return "static"
    else:
        raise ValueError(
            f" Cannot detect graph type in {folder}.\n"
            f"   Missing required files.\n"
            f"   Expected: 'edge_index.pt' (static) or 'src_nodes.pt' + 'dst_nodes.pt' (temporal)"
        )


# ------------------------------------------------------------
# SPLIT FOR STATIC GRAPH (GraphSAGE / GraphSAGE-T)
# ------------------------------------------------------------
def split_static_graph(folder, out_dir, train_ratio=0.60, val_ratio=0.20, 
                       test_ratio=0.20, random_seed=42):
    """
    Create stratified random splits for static graphs.
    Preserves class distribution across splits.
    """
    
    print("\n" + "=" * 70)
    print("STATIC GRAPH DETECTED (GraphSAGE / GraphSAGE-T)")
    print("=" * 70)
    print("Split type: Stratified random (class-balanced)")
    print(f"Ratio: {int(train_ratio*100)}/{int(val_ratio*100)}/{int(test_ratio*100)}")
    
    # Load data
    edge_index = torch.load(os.path.join(folder, "edge_index.pt"))
    y_edge = torch.load(os.path.join(folder, "y_edge.pt"))
    x = torch.load(os.path.join(folder, "x.pt"))
    
    num_nodes = x.shape[0]
    num_edges = edge_index.shape[1]
    
    print(f"\nGraph statistics:")
    print(f"  Nodes: {num_nodes:,}")
    print(f"  Edges: {num_edges:,}")
    
    # Check label distribution
    num_positive = y_edge.sum().item()
    pct_positive = 100 * num_positive / num_edges
    print(f"\nLabel distribution:")
    print(f"  Positive (laundering): {num_positive:,} ({pct_positive:.2f}%)")
    print(f"  Negative (normal):     {num_edges - num_positive:,} ({100 - pct_positive:.2f}%)")
    
    if pct_positive < 1:
        print(f"  Severe imbalance detected - using stratified split")
    
    # Create indices
    all_idx = np.arange(num_edges)
    
    # Stratified split: 60% train, 40% temp (for val+test)
    try:
        train_idx, temp_idx = train_test_split(
            all_idx, 
            test_size=(val_ratio + test_ratio),
            stratify=y_edge.numpy(), 
            random_state=random_seed
        )
        
        # Split temp into val (50%) and test (50%)
        # This gives us 20% val and 20% test overall
        val_idx, test_idx = train_test_split(
            temp_idx, 
            test_size=0.50,
            stratify=y_edge.numpy()[temp_idx], 
            random_state=random_seed
        )
        
    except ValueError as e:
        warnings.warn(
            f"Stratified split failed (not enough samples per class?): {e}\n"
            f"Falling back to random split without stratification."
        )
        # Fallback to non-stratified split
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
    
    print(f"\n Split sizes:")
    print(f"  Train: {len(train_idx):,} edges ({100*len(train_idx)/num_edges:.1f}%)")
    print(f"  Val:   {len(val_idx):,} edges ({100*len(val_idx)/num_edges:.1f}%)")
    print(f"  Test:  {len(test_idx):,} edges ({100*len(test_idx)/num_edges:.1f}%)")
    
    # Report label distribution per split
    train_labels = y_edge[train_idx]
    val_labels = y_edge[val_idx]
    test_labels = y_edge[test_idx]
    
    print(f"\n📋 Label distribution per split:")
    print(f"  Train: {train_labels.sum():,} positive ({100*train_labels.float().mean():.2f}%)")
    print(f"  Val:   {val_labels.sum():,} positive ({100*val_labels.float().mean():.2f}%)")
    print(f"  Test:  {test_labels.sum():,} positive ({100*test_labels.float().mean():.2f}%)")
    
    # Create node-level masks (nodes touched by edges in each split)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    
    for idx in train_idx:
        train_mask[src[idx]] = True
        train_mask[dst[idx]] = True
    
    for idx in val_idx:
        val_mask[src[idx]] = True
        val_mask[dst[idx]] = True
    
    for idx in test_idx:
        test_mask[src[idx]] = True
        test_mask[dst[idx]] = True
    
    print(f"\n🔗 Node coverage:")
    print(f"  Train nodes: {train_mask.sum():,} ({100*train_mask.float().mean():.1f}%)")
    print(f"  Val nodes:   {val_mask.sum():,} ({100*val_mask.float().mean():.1f}%)")
    print(f"  Test nodes:  {test_mask.sum():,} ({100*test_mask.float().mean():.1f}%)")
    
    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    
    torch.save(torch.tensor(train_idx, dtype=torch.long), os.path.join(out_dir, "train_edge_idx.pt"))
    torch.save(torch.tensor(val_idx, dtype=torch.long), os.path.join(out_dir, "val_edge_idx.pt"))
    torch.save(torch.tensor(test_idx, dtype=torch.long), os.path.join(out_dir, "test_edge_idx.pt"))
    
    torch.save(train_mask, os.path.join(out_dir, "train_node_mask.pt"))
    torch.save(val_mask, os.path.join(out_dir, "val_node_mask.pt"))
    torch.save(test_mask, os.path.join(out_dir, "test_node_mask.pt"))
    
    # Save metadata
    meta = {
        "type": "static",
        "split_method": "stratified_random",
        "num_edges": int(num_edges),
        "num_nodes": int(num_nodes),
        "train_edges": int(len(train_idx)),
        "val_edges": int(len(val_idx)),
        "test_edges": int(len(test_idx)),
        "train_positive": int(train_labels.sum()),
        "val_positive": int(val_labels.sum()),
        "test_positive": int(test_labels.sum()),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "random_seed": int(random_seed),
    }
    
    with open(os.path.join(out_dir, "split_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n✅ Static graph splits saved to:")
    print(f"   {out_dir}")
    print("=" * 70)


# ------------------------------------------------------------
# SPLIT FOR TEMPORAL GRAPH (TGAT)
# ------------------------------------------------------------
def split_temporal_graph(folder, out_dir, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20):
    """
    Create chronological splits for temporal graphs (TGAT).
    Maintains temporal ordering to prevent data leakage.
    """
    
    print("\n" + "=" * 70)
    print("TEMPORAL GRAPH DETECTED (TGAT)")
    print("=" * 70)
    print("Split type: Chronological (time-ordered)")
    print(f"Ratio: {int(train_ratio*100)}/{int(val_ratio*100)}/{int(test_ratio*100)}")
    
    # Load data
    timestamps = torch.load(os.path.join(folder, "timestamps.pt"))
    y_edge = torch.load(os.path.join(folder, "y_edge.pt"))
    
    num_events = len(timestamps)
    
    print(f"\nGraph statistics:")
    print(f"  Total events: {num_events:,}")
    
    # Check label distribution
    num_positive = y_edge.sum().item()
    pct_positive = 100 * num_positive / num_events
    print(f"\nLabel distribution:")
    print(f"  Positive (laundering): {num_positive:,} ({pct_positive:.2f}%)")
    print(f"  Negative (normal):     {num_events - num_positive:,} ({100 - pct_positive:.2f}%)")
    
    # Verify timestamps are sorted
    timestamps_np = timestamps.numpy()
    is_sorted = np.all(timestamps_np[:-1] <= timestamps_np[1:])
    
    if not is_sorted:
        print("  ⚠️ WARNING: Timestamps not sorted! Sorting now...")
        order = np.argsort(timestamps_np)
    else:
        print("  ✓ Timestamps already sorted (chronological order)")
        order = np.arange(num_events)
    
    # Create chronological splits
    train_end = int(train_ratio * num_events)
    val_end = int((train_ratio + val_ratio) * num_events)
    
    train_idx = order[:train_end]
    val_idx = order[train_end:val_end]
    test_idx = order[val_end:]
    
    print(f"\n📊 Split sizes:")
    print(f"  Train: {len(train_idx):,} events ({100*len(train_idx)/num_events:.1f}%)")
    print(f"  Val:   {len(val_idx):,} events ({100*len(val_idx)/num_events:.1f}%)")
    print(f"  Test:  {len(test_idx):,} events ({100*len(test_idx)/num_events:.1f}%)")
    
    # Report label distribution per split
    train_labels = y_edge[train_idx]
    val_labels = y_edge[val_idx]
    test_labels = y_edge[test_idx]
    
    print(f"\n📋 Label distribution per split:")
    print(f"  Train: {train_labels.sum():,} positive ({100*train_labels.float().mean():.2f}%)")
    print(f"  Val:   {val_labels.sum():,} positive ({100*val_labels.float().mean():.2f}%)")
    print(f"  Test:  {test_labels.sum():,} positive ({100*test_labels.float().mean():.2f}%)")
    
    # Temporal leakage check
    train_times = timestamps_np[train_idx]
    val_times = timestamps_np[val_idx]
    test_times = timestamps_np[test_idx]
    
    print(f"\n⏰ Temporal boundaries:")
    print(f"  Train: {train_times.min()} → {train_times.max()}")
    print(f"  Val:   {val_times.min()} → {val_times.max()}")
    print(f"  Test:  {test_times.min()} → {test_times.max()}")
    
    # Check for temporal leakage
    if train_times.max() > val_times.min():
        print("  ⚠️ WARNING: Train overlaps with val (temporal leakage!)")
    if val_times.max() > test_times.min():
        print("  ⚠️ WARNING: Val overlaps with test (temporal leakage!)")
    if train_times.max() <= val_times.min() <= val_times.max() <= test_times.min():
        print("  ✓ No temporal leakage detected")
    
    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    
    # Use consistent naming with static graphs (train_edge_idx.pt)
    torch.save(torch.tensor(train_idx, dtype=torch.long), os.path.join(out_dir, "train_edge_idx.pt"))
    torch.save(torch.tensor(val_idx, dtype=torch.long), os.path.join(out_dir, "val_edge_idx.pt"))
    torch.save(torch.tensor(test_idx, dtype=torch.long), os.path.join(out_dir, "test_edge_idx.pt"))
    
    # Save metadata
    meta = {
        "type": "temporal",
        "split_method": "chronological",
        "num_events": int(num_events),
        "train_events": int(len(train_idx)),
        "val_events": int(len(val_idx)),
        "test_events": int(len(test_idx)),
        "train_positive": int(train_labels.sum()),
        "val_positive": int(val_labels.sum()),
        "test_positive": int(test_labels.sum()),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "train_time_range": [int(train_times.min()), int(train_times.max())],
        "val_time_range": [int(val_times.min()), int(val_times.max())],
        "test_time_range": [int(test_times.min()), int(test_times.max())],
    }
    
    with open(os.path.join(out_dir, "split_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n✅ TGAT event splits saved to:")
    print(f"   {out_dir}")
    print("=" * 70)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def create_splits(graph_folder, out_dir=None, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, random_seed=42):
    """
    Main function to create train/val/test splits.
    
    Parameters:
        graph_folder: Path to graph directory
        out_dir: Custom output directory (optional). If None, uses splits/<dataset_name>
        train_ratio: Fraction for training (default 0.60)
        val_ratio: Fraction for validation (default 0.20)
        test_ratio: Fraction for testing (default 0.20)
        random_seed: Random seed for reproducibility (static graphs only)
    """
    
    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    dataset_name = os.path.basename(graph_folder)
    
    # Allow custom output directory, otherwise use default
    if out_dir is None:
        # Output to splits/<dataset_name>/
        parent_dir = os.path.dirname(graph_folder)
        splits_base = os.path.join(parent_dir, "..", "splits")
        out_dir = os.path.join(splits_base, dataset_name)
    
    out_dir = os.path.abspath(out_dir)
    
    print(f"\n{'='*70}")
    print(f"CREATING SPLITS FOR: {dataset_name}")
    print(f"{'='*70}")
    print(f"Graph folder: {graph_folder}")
    print(f"Output folder: {out_dir}")
    
    # Detect graph type and create splits
    graph_type = detect_graph_type(graph_folder)
    
    if graph_type == "static":
        split_static_graph(graph_folder, out_dir, train_ratio, val_ratio, test_ratio, random_seed)
    else:
        split_temporal_graph(graph_folder, out_dir, train_ratio, val_ratio, test_ratio)


# ------------------------------------------------------------
# BATCH PROCESS MULTIPLE DATASETS
# ------------------------------------------------------------
def batch_create_splits(base_dir, graph_type="graphs", train_ratio=0.60, val_ratio=0.20, test_ratio=0.20):
    """
    Create splits for all datasets in a directory.
    
    Parameters:
        base_dir: Base directory containing graph folders
        graph_type: "graphs" or "tgat_graphs"
        train_ratio, val_ratio, test_ratio: Split ratios
    """
    graph_dir = os.path.join(base_dir, graph_type)
    
    if not os.path.exists(graph_dir):
        print(f"❌ Directory not found: {graph_dir}")
        return
    
    datasets = [d for d in os.listdir(graph_dir) 
                if os.path.isdir(os.path.join(graph_dir, d))]
    
    print(f"\nFound {len(datasets)} datasets in {graph_dir}")
    print(f"Datasets: {', '.join(datasets)}\n")
    
    for dataset in datasets:
        folder = os.path.join(graph_dir, dataset)
        try:
            create_splits(folder, out_dir=None, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        except Exception as e:
            print(f"\n❌ Failed to process {dataset}: {e}\n")
    
    print(f"\n✅ Batch processing complete! Processed {len(datasets)} datasets.")


# ------------------------------------------------------------
# EXECUTE
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create train/val/test splits for graph datasets"
    )
    parser.add_argument(
        "--graph_folder",
        type=str,
        help="Path to single graph directory (e.g., graphs/HI-Small_Trans)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Custom output directory (e.g., splits/HI-Small_Trans_temporal). If not specified, uses splits/<dataset_name>"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all datasets in graphs/ and tgat_graphs/ directories"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help="Base directory for batch processing (default: current directory)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.60,
        help="Training set ratio (default: 0.60)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.20,
        help="Validation set ratio (default: 0.20)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.20,
        help="Test set ratio (default: 0.20)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for static graph splits (default: 42)"
    )

    args = parser.parse_args()
    
    if args.batch:
        # Batch process all datasets
        print("="*70)
        print("BATCH MODE: Processing all datasets")
        print("="*70)
        
        batch_create_splits(
            args.base_dir,
            "graphs",
            args.train_ratio,
            args.val_ratio,
            args.test_ratio
        )
        batch_create_splits(
            args.base_dir,
            "tgat_graphs",
            args.train_ratio,
            args.val_ratio,
            args.test_ratio
        )
    elif args.graph_folder:
        # Single dataset
        create_splits(
            args.graph_folder,
            args.out_dir,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed
        )
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  # Use default output (splits/HI-Small_Trans/)")
        print("  python create_splits.py --graph_folder graphs/HI-Small_Trans")
        print("\n  # Custom output directory (won't override existing splits)")
        print("  python create_splits.py --graph_folder graphs/HI-Small_Trans --out_dir splits/HI-Small_Trans_temporal")
        print("\n  # Batch process all datasets")
        print("  python create_splits.py --batch --base_dir /path/to/data")