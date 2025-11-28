#!/usr/bin/env python3
"""
GraphSAGE-T metrics recovery - Windows compatible
Properly handles Windows paths
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.evaluation_utils import evaluate_binary_classifier

# Model classes
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.2, aggr="mean"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_dim, hidden_dim, aggr=aggr))
        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = x
        for conv in self.layers:
            h = conv(h, edge_index)
            h = self.relu(h)
            h = self.drop(h)
        return h


class TemporalClassifier(nn.Module):
    def __init__(self, node_dim, feat_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_src, h_dst, feat):
        return self.mlp(torch.cat([h_src, h_dst, feat], dim=-1)).view(-1)


class GraphSAGE_T(nn.Module):
    def __init__(self, node_dim, feat_dim, hidden_dim=128, num_layers=2, dropout=0.2, aggr="mean"):
        super().__init__()
        self.encoder = GraphSAGEEncoder(node_dim, hidden_dim, num_layers, dropout, aggr)
        self.classifier = TemporalClassifier(hidden_dim, feat_dim, hidden_dim, dropout)

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def classify(self, h, eidx, feat_batch):
        src, dst = eidx
        return self.classifier(h[src], h[dst], feat_batch)


def build_sinusoidal_time_encoding(timestamps, time_dim=16):
    assert time_dim % 2 == 0
    ts = timestamps.float()
    t_min, t_max = ts.min(), ts.max()
    
    if t_max > t_min:
        ts_norm = (ts - t_min) / (t_max - t_min)
    else:
        ts_norm = torch.zeros_like(ts)
    
    ts_norm = ts_norm.view(-1, 1)
    half = time_dim // 2
    div_term = torch.exp(
        torch.arange(half, device=ts.device) * -(np.log(10000.0) / half)
    ).view(1, -1)
    
    angles = ts_norm * div_term
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


@torch.no_grad()
def evaluate_split_minibatch(model, x, edge_index, feat, y, split_idx, batch_size, device):
    model.eval()
    all_probs = []

    for start in range(0, split_idx.size(0), batch_size):
        end = min(start + batch_size, split_idx.size(0))
        idx = split_idx[start:end]

        h = model.encode(x, edge_index)
        logits = model.classify(h, edge_index[:, idx], feat[idx])
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    labels = y[split_idx].cpu().numpy()

    metrics = evaluate_binary_classifier(
        labels, all_probs,
        threshold=0.5,
        auto_threshold=True,
        compute_top_k=True,
        k_values=[100, 500, 1000],
        verbose=False
    )

    return metrics, all_probs


def print_metrics(metrics, split_name):
    print(f"\n{split_name}:")
    print(f"  AUPR:           {metrics['aupr']:.4f}")
    print(f"  F1:             {metrics['f1']:.4f}")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")
    print(f"  ROC-AUC:        {metrics['roc_auc']:.4f}")
    if 'top_k_metrics' in metrics:
        print(f"  Precision@100:  {metrics['top_k_metrics']['precision_at_100']:.2f}")
        print(f"  Precision@500:  {metrics['top_k_metrics']['precision_at_500']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--graph_folder", default="graphs")
    parser.add_argument("--split_folder", default="splits")
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--time_dim", type=int, default=16)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"GraphSAGE-T Metrics Recovery")
    print(f"{'='*60}\n")

    # Use Path for proper Windows/Linux compatibility
    results_dir = Path(args.results_dir).absolute()
    graph_folder = Path(args.graph_folder).absolute()
    split_folder = Path(args.split_folder).absolute()

    print(f"Results dir:  {results_dir}")
    print(f"Graph folder: {graph_folder}")
    print(f"Split folder: {split_folder}\n")

    # Check paths exist
    best_model_path = results_dir / "best_model.pt"
    if not best_model_path.exists():
        print(f" ERROR: best_model.pt not found at: {best_model_path}")
        return

    if not graph_folder.exists():
        print(f" ERROR: Graph folder not found: {graph_folder}")
        return

    if not split_folder.exists():
        print(f" ERROR: Split folder not found: {split_folder}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load data - using Path objects
    print("Loading graph data...")
    try:
        x = torch.load(graph_folder / "x.pt").to(device)
        edge_index = torch.load(graph_folder / "edge_index.pt").to(device)
        edge_attr = torch.load(graph_folder / "edge_attr.pt").to(device)
        y_edge = torch.load(graph_folder / "y_edge.pt").to(device)
        timestamps = torch.load(graph_folder / "timestamps.pt").to(device)
        print(f" Graph: {x.size(0)} nodes, {edge_index.size(1)} edges")
    except Exception as e:
        print(f" ERROR loading graph: {e}")
        return

    print("Loading splits...")
    try:
        train_idx = torch.load(split_folder / "train_edge_idx.pt").to(device)
        val_idx = torch.load(split_folder / "val_edge_idx.pt").to(device)
        test_idx = torch.load(split_folder / "test_edge_idx.pt").to(device)
        print(f" Splits: train={train_idx.size(0)}, val={val_idx.size(0)}, test={test_idx.size(0)}")
    except Exception as e:
        print(f" ERROR loading splits: {e}")
        return

    print(f"Building time encoding (dim={args.time_dim})...")
    time_enc = build_sinusoidal_time_encoding(timestamps, args.time_dim)
    feat = torch.cat([edge_attr, time_enc], dim=1)
    print(f" Features: {feat.size(1)} dims")

    print(f"\nCreating model (hidden={args.hidden_dim}, layers={args.num_layers})...")
    model = GraphSAGE_T(
        node_dim=x.size(1),
        feat_dim=feat.size(1),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.2,
        aggr="mean"
    ).to(device)

    print(f"Loading model from: {best_model_path}")
    try:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("✓ Model loaded\n")
    except Exception as e:
        print(f" ERROR loading model: {e}")
        print("Try different --hidden_dim or --num_layers")
        return

    print("="*60)
    print("Evaluating...")
    print("="*60)

    print("\n[1/3] Train split...")
    train_metrics, train_probs = evaluate_split_minibatch(
        model, x, edge_index, feat, y_edge, train_idx, args.batch_size, device
    )

    print("[2/3] Val split...")
    val_metrics, val_probs = evaluate_split_minibatch(
        model, x, edge_index, feat, y_edge, val_idx, args.batch_size, device
    )

    print("[3/3] Test split...")
    test_metrics, test_probs = evaluate_split_minibatch(
        model, x, edge_index, feat, y_edge, test_idx, args.batch_size, device
    )

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print_metrics(train_metrics, "TRAIN")
    print_metrics(val_metrics, "VAL")
    print_metrics(test_metrics, "TEST")

    # Save
    print("\n" + "="*60)
    print("Saving...")
    print("="*60 + "\n")

    output = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "best_val_aupr": float(val_metrics["aupr"]),
        "batch_size": args.batch_size,
    }

    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"✓ {metrics_path}")

    torch.save(torch.tensor(train_probs), results_dir / "train_pred_probs.pt")
    torch.save(torch.tensor(val_probs), results_dir / "val_pred_probs.pt")
    torch.save(torch.tensor(test_probs), results_dir / "test_pred_probs.pt")
    print(f"✓ Prediction probabilities saved")

    print("\n" + "="*60)
    print(" COMPLETE!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()