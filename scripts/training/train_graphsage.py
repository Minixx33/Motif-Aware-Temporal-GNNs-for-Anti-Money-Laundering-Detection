# train_graphsage_GPU.py
# -----------------------------------------------------------
# GPU-optimized GraphSAGE with edge-level mini-batching
# Stable version:
#   - No AMP
#   - No detach
#   - Recompute node embeddings per batch (simple & safe)
# -----------------------------------------------------------
# Usage
# Example: (default)
# python scripts/training/train_graphsage.py \
#     --config configs/tgat.yaml \
#     --dataset configs/datasets/baseline.yaml \
#     --base_config configs/base.yaml
#
# Example: (low-intensity HiSmallRat)
# python scripts/training/train_graphsage.py \
#     --config configs/tgat.yaml \
#     --dataset configs/datasets/hismall_rat_low.yaml \
#     --intensity low \
#     --base_config configs/base.yaml
# -----------------------------------------------------------

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import SAGEConv
from torch.utils.tensorboard import SummaryWriter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from scripts.utils.config_utils import (
    setup_experiment,
    save_experiment_config,
)
from scripts.utils.evaluation_utils import (
    evaluate_binary_classifier,
    print_metrics
)

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# -----------------------------------------------------------
# Model
# -----------------------------------------------------------

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.2, aggregator="mean"):
        super().__init__()
        assert num_layers >= 1

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim, aggr=aggregator))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.relu(h)
            h = self.dropout(h)
        return h


class EdgeClassifier(nn.Module):
    def __init__(self, node_emb_dim, edge_attr_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        in_dim = node_emb_dim * 2 + edge_attr_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_src, h_dst, edge_attr):
        z = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        return self.mlp(z).view(-1)


class GraphSAGE_EdgeModel(nn.Module):
    def __init__(self, in_dim_node, in_dim_edge, cfg_model):
        super().__init__()

        hidden_dim = cfg_model["hidden_dim"]
        num_layers = cfg_model.get("num_layers", 2)
        dropout = cfg_model.get("dropout", 0.2)
        aggregator = cfg_model.get("aggregator", "mean")

        self.encoder = GraphSAGEEncoder(
            in_dim=in_dim_node,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            aggregator=aggregator,
        )

        self.classifier = EdgeClassifier(
            node_emb_dim=hidden_dim,
            edge_attr_dim=in_dim_edge,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def encode_nodes(self, x, edge_index):
        return self.encoder(x, edge_index)
    
    def classify_edges(self, h, edge_index_batch, edge_attr_batch):
        src, dst = edge_index_batch
        h_src = h[src]
        h_dst = h[dst]
        return self.classifier(h_src, h_dst, edge_attr_batch)


# -----------------------------------------------------------
# Mini-batch training (no AMP, no detach)
# -----------------------------------------------------------

def run_epoch_minibatch(
    model,
    optimizer,
    loss_fn,
    x,
    edge_index,
    edge_attr,
    y_edge,
    train_idx,
    batch_size,
    device,
    max_grad_norm=1.0,
):
    """
    Train with edge mini-batches.
    Both encoder and classifier learn end-to-end.
    For each batch:
      - Recompute node embeddings h = encode_nodes(x, edge_index)
      - Classify only batch edges
      - Backprop and step
    This is heavier but much more robust.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Shuffle training edges
    perm = torch.randperm(len(train_idx), device=device)
    shuffled_idx = train_idx[perm]

    for start in range(0, len(shuffled_idx), batch_size):
        end = min(start + batch_size, len(shuffled_idx))
        batch_idx = shuffled_idx[start:end]

        optimizer.zero_grad(set_to_none=True)

        # 1) Encode ALL nodes for current model params
        h = model.encode_nodes(x, edge_index)

        # 2) Pick batch edges
        edge_index_batch = edge_index[:, batch_idx]
        edge_attr_batch = edge_attr[batch_idx]
        labels_batch = y_edge[batch_idx].float()

        # 3) Forward on batch
        logits = model.classify_edges(h, edge_index_batch, edge_attr_batch)
        loss = loss_fn(logits, labels_batch)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[WARN] NaN/Inf loss at batch {start}:{end}, skipping.")
            continue

        # 4) Backward + step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        print("[WARN] No valid batches this epoch; returning loss=0.")
        return 0.0

    return total_loss / num_batches


# -----------------------------------------------------------
# Evaluation
# -----------------------------------------------------------

@torch.no_grad()
def evaluate_split_minibatch(
    model,
    x,
    edge_index,
    edge_attr,
    y_edge,
    split_idx,
    batch_size,
    device,
    eval_cfg,
):
    """
    Evaluate edge classifier on a given split using edge mini-batches.
    Encoder is run once in inference mode.
    """
    model.eval()

    # Encode nodes once (no grad)
    h = model.encode_nodes(x, edge_index)

    all_probs = []
    num_edges = len(split_idx)

    for start in range(0, num_edges, batch_size):
        end = min(start + batch_size, num_edges)
        batch_idx = split_idx[start:end]

        edge_index_batch = edge_index[:, batch_idx]
        edge_attr_batch = edge_attr[batch_idx]

        logits = model.classify_edges(h, edge_index_batch, edge_attr_batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    all_probs = np.concatenate(all_probs)
    labels = y_edge[split_idx].cpu().numpy()

    metrics = evaluate_binary_classifier(
        y_true=labels,
        y_pred_probs=all_probs,
        threshold=eval_cfg.get("threshold", 0.5),
        auto_threshold=eval_cfg.get("auto_threshold", True),
        compute_top_k=eval_cfg.get("compute_top_k", True),
        k_values=eval_cfg.get("top_k_values", [100, 500, 1000]),
        verbose=False,
    )
    return metrics, all_probs


def compute_pos_weight(y_train):
    pos = (y_train == 1).sum().item()
    neg = (y_train == 0).sum().item()
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / max(pos, 1))


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPU-optimized GraphSAGE (stable)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--intensity", default=None)
    parser.add_argument("--base_config", default="configs/base.yaml")
    args = parser.parse_args()

    setup = setup_experiment(
        model_config_path=args.config,
        dataset_config_path=args.dataset,
        intensity=args.intensity,
        base_config_path=args.base_config,
        verbose=True,
        enable_logging=True,
    )

    base_cfg = setup["base_cfg"]
    model_cfg = setup["model_cfg"]
    dataset_cfg = setup["dataset_cfg"]
    paths = setup["paths"]
    device = setup["device"]
    experiment_name = setup["experiment_name"]
    intensity = setup.get("intensity")
    logger = setup.get("logger")

    eval_cfg = base_cfg["evaluation"]
    training_cfg = model_cfg["training"]
    loss_cfg = model_cfg["loss"]

    # Batch sizes
    batch_size = int(training_cfg.get("batch_size", 8192))
    eval_batch_size = int(training_cfg.get("eval_batch_size", 16384))

    print(f"\n{'='*70}")
    print(f"GPU Mini-Batch Training (GraphSAGE Stable)")
    print(f"{'='*70}")
    print(f"Train batch size: {batch_size}")
    print(f"Eval batch size:  {eval_batch_size}")
    print(f"Mixed Precision:  False (disabled)")
    print(f"Device:           {device}")
    print(f"{'='*70}\n")

    # TensorBoard
    tb_log_dir = os.path.join(paths["logs_dir"], "tb")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Load graph
    print("Loading graph tensors...")
    graph_folder = paths["graph_folder"]

    edge_index = torch.load(os.path.join(graph_folder, "edge_index.pt")).to(device)
    edge_attr = torch.load(os.path.join(graph_folder, "edge_attr.pt")).to(device)
    x = torch.load(os.path.join(graph_folder, "x.pt")).to(device)
    y_edge = torch.load(os.path.join(graph_folder, "y_edge.pt")).to(device)

    print(f"Nodes: {x.size(0):,}, Edges: {edge_index.size(1):,}")
    print(f"Node features: {x.size(1)}, Edge features: {edge_attr.size(1)}")

    # Load splits
    split_folder = paths["split_folder"]
    train_idx = torch.load(os.path.join(split_folder, "train_edge_idx.pt")).to(device)
    val_idx = torch.load(os.path.join(split_folder, "val_edge_idx.pt")).to(device)
    test_idx = torch.load(os.path.join(split_folder, "test_edge_idx.pt")).to(device)

    print(f"Train: {len(train_idx):,}, Val: {len(val_idx):,}, Test: {len(test_idx):,}")

    # Model
    model = GraphSAGE_EdgeModel(
        in_dim_node=x.size(1),
        in_dim_edge=edge_attr.size(1),
        cfg_model=model_cfg["model"],
    ).to(device)

    # Loss
    pos_weight = loss_cfg.get("pos_weight", None)
    if pos_weight is None:
        pos_weight = compute_pos_weight(y_edge[train_idx])
        pos_weight = min(pos_weight.item(), 100.0)  # cap for stability
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    print(f"pos_weight: {pos_weight.item():.2f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer_cfg = training_cfg.get("optimizer", {})
    lr = float(training_cfg.get("lr", 5e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    betas = tuple(float(b) for b in optimizer_cfg.get("betas", [0.9, 0.999]))
    eps = float(optimizer_cfg.get("eps", 1e-8))

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    # Training loop
    results_dir = paths["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    best_val = -1e9
    best_epoch = -1
    patience = 0
    max_patience = int(training_cfg.get("early_stopping_patience", 15))
    epochs = int(training_cfg.get("epochs", 100))
    best_model_path = os.path.join(results_dir, "best_model.pt")

    print(f"\nStarting training for {epochs} epochs...\n")
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # Train one epoch
        loss = run_epoch_minibatch(
            model,
            optimizer,
            loss_fn,
            x,
            edge_index,
            edge_attr,
            y_edge,
            train_idx,
            batch_size,
            device,
        )

        # Validate
        val_metrics, _ = evaluate_split_minibatch(
            model,
            x,
            edge_index,
            edge_attr,
            y_edge,
            val_idx,
            eval_batch_size,
            device,
            eval_cfg,
        )

        val_aupr = val_metrics["aupr"]
        val_f1 = val_metrics["f1"]

        epoch_time = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch:03d} | loss={loss:.4f} | "
            f"val_F1={val_f1:.4f} | val_AUPR={val_aupr:.4f} | "
            f"time={epoch_time:.2f}s"
        )

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Val/F1", val_f1, epoch)
        writer.add_scalar("Val/AUPR", val_aupr, epoch)
        writer.add_scalar("Time/epoch_seconds", epoch_time, epoch)

        # Early stopping on val AUPR
        if val_aupr > best_val:
            best_val = val_aupr
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience += 1

        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    total_time = time.perf_counter() - total_start
    print(f"\nTotal training time: {total_time:.2f}s ({total_time/60:.1f} min)")

    # Final evaluation
    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("Evaluating on train/val/test...")
    train_metrics, train_probs = evaluate_split_minibatch(
        model,
        x,
        edge_index,
        edge_attr,
        y_edge,
        train_idx,
        eval_batch_size,
        device,
        eval_cfg,
    )
    val_metrics, val_probs = evaluate_split_minibatch(
        model,
        x,
        edge_index,
        edge_attr,
        y_edge,
        val_idx,
        eval_batch_size,
        device,
        eval_cfg,
    )
    test_metrics, test_probs = evaluate_split_minibatch(
        model,
        x,
        edge_index,
        edge_attr,
        y_edge,
        test_idx,
        eval_batch_size,
        device,
        eval_cfg,
    )

    print_metrics(train_metrics, model_name=f"{experiment_name} TRAIN")
    print_metrics(val_metrics, model_name=f"{experiment_name} VAL")
    print_metrics(test_metrics, model_name=f"{experiment_name} TEST")

    # Save metrics & preds
    out = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "best_epoch": best_epoch,
        "best_val_aupr": float(best_val),
        "total_training_time_sec": float(total_time),
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
    }

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    torch.save(torch.tensor(train_probs), os.path.join(results_dir, "train_pred_probs.pt"))
    torch.save(torch.tensor(val_probs), os.path.join(results_dir, "val_pred_probs.pt"))
    torch.save(torch.tensor(test_probs), os.path.join(results_dir, "test_pred_probs.pt"))

    save_experiment_config(
        save_dir=results_dir,
        base_cfg=base_cfg,
        model_cfg=model_cfg,
        dataset_cfg=dataset_cfg,
        intensity=intensity,
        additional_info={
            "total_training_time_sec": float(total_time),
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
        },
        filename="experiment_config.json",
    )

    writer.close()
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)

    if logger:
        logger.close()


if __name__ == "__main__":
    main()
