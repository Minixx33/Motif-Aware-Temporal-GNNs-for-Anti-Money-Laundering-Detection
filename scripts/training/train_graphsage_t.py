# train_graphsage_t_GPU.py
# -----------------------------------------------------------
# GPU-optimized GraphSAGE-T (Temporal GraphSAGE)
# Stable version:
#   - NO AMP (AMP caused NaNs)
#   - NO DETACH (full end-to-end)
#   - Temporal correctness: encode inside minibatch
#   - Gradient clipping
#   - STANDARD LOGGING identical to GraphSAGE + TGAT
# -----------------------------------------------------------
# Usage
# Example: (default)
# python scripts/training/train_graphsage_t.py \
#     --config configs/tgat.yaml \
#     --dataset configs/datasets/baseline.yaml \
#     --base_config configs/base.yaml
#
# Example: (low-intensity HiSmallRat)
# python scripts/training/train_graphsage_t.py \
#     --config configs/tgat.yaml \
#     --dataset configs/datasets/hismall_rat_low.yaml \
#     --intensity low \
#     --base_config configs/base.yaml
# -----------------------------------------------------------

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import SAGEConv
from torch.utils.tensorboard import SummaryWriter

from scripts.utils.config_utils import setup_experiment, save_experiment_config
from scripts.utils.evaluation_utils import evaluate_binary_classifier, print_metrics

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------------------------------------
# Time Encoding (TGAT sinusoidal)
# -----------------------------------------------------------

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


# -----------------------------------------------------------
# GraphSAGE-T Model
# -----------------------------------------------------------

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
    def __init__(self, node_dim, feat_dim, cfg):
        super().__init__()
        hid = cfg["hidden_dim"]
        layers = cfg.get("num_layers", 2)
        drop = cfg.get("dropout", 0.2)
        aggr = cfg.get("aggregator", "mean")

        self.encoder = GraphSAGEEncoder(node_dim, hid, layers, drop, aggr)
        self.classifier = TemporalClassifier(hid, feat_dim, hid, drop)

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def classify(self, h, eidx, feat_batch):
        src, dst = eidx
        return self.classifier(h[src], h[dst], feat_batch)


# -----------------------------------------------------------
# Mini-batch training — STANDARDIZED
# -----------------------------------------------------------

def run_epoch_minibatch(
    model, optimizer, loss_fn,
    x, edge_index, feat, y, train_idx,
    batch_size, device
):
    model.train()
    total_loss = 0.0
    steps = 0

    perm = torch.randperm(train_idx.size(0), device=device)
    idx = train_idx[perm]

    for start in range(0, idx.size(0), batch_size):
        end = min(start + batch_size, idx.size(0))
        batch_edges = idx[start:end]

        optimizer.zero_grad(set_to_none=True)

        # Temporal correctness: encode inside each batch
        h = model.encode(x, edge_index)

        eidx = edge_index[:, batch_edges]
        fb = feat[batch_edges]
        yb = y[batch_edges].float()

        logits = model.classify(h, eidx, fb)
        loss = loss_fn(logits, yb)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate_minibatch(
    model, loss_fn,
    x, edge_index, feat, y, split_idx,
    batch_size, device, eval_cfg
):
    model.eval()
    total_loss = 0.0
    steps = 0

    all_probs = []

    for start in range(0, split_idx.size(0), batch_size):
        end = min(start + batch_size, split_idx.size(0))
        idx = split_idx[start:end]

        h = model.encode(x, edge_index)

        logits = model.classify(h, edge_index[:, idx], feat[idx])
        probs = torch.sigmoid(logits)
        labels = y[idx].float()

        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        steps += 1

        all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    labels = y[split_idx].cpu().numpy()

    metrics = evaluate_binary_classifier(
        labels, all_probs,
        threshold=eval_cfg.get("threshold", 0.5),
        auto_threshold=eval_cfg.get("auto_threshold", True),
        compute_top_k=eval_cfg.get("compute_top_k", True),
        k_values=eval_cfg.get("top_k_values", [100, 500, 1000]),
        verbose=False
    )

    avg_loss = total_loss / max(steps, 1)
    return metrics, all_probs, avg_loss


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--intensity", default=None)
    parser.add_argument("--base_config", default="configs/base.yaml")
    args = parser.parse_args()

    setup = setup_experiment(
        args.config, args.dataset,
        intensity=args.intensity,
        base_config_path=args.base_config,
        verbose=True,
        enable_logging=True
    )

    base_cfg = setup["base_cfg"]
    model_cfg = setup["model_cfg"]
    eval_cfg = base_cfg["evaluation"]
    train_cfg = model_cfg["training"]
    loss_cfg = model_cfg["loss"]
    paths = setup["paths"]
    device = setup["device"]
    experiment_name = setup["experiment_name"]
    logger = setup.get("logger")

    batch_size = train_cfg.get("batch_size", 8192)
    eval_batch_size = train_cfg.get("eval_batch_size", 16384)

    print("\n======= GRAPH SAGE-T TRAINING (FP32) =======")
    print(f"Batch size:       {batch_size}")
    print(f"Eval batch size:  {eval_batch_size}")
    print(f"Device:           {device}")
    print("===========================================\n")

    writer = SummaryWriter(os.path.join(paths["logs_dir"], "tb"))

    # Load tensors
    graph = paths["graph_folder"]
    x = torch.load(f"{graph}/x.pt").to(device)
    edge_index = torch.load(f"{graph}/edge_index.pt").to(device)
    edge_attr = torch.load(f"{graph}/edge_attr.pt").to(device)
    y_edge = torch.load(f"{graph}/y_edge.pt").to(device)
    timestamps = torch.load(f"{graph}/timestamps.pt").to(device)

    # Time encoding
    time_dim = model_cfg["model"].get("time_dim", 16)
    time_enc = build_sinusoidal_time_encoding(timestamps, time_dim)
    feat = torch.cat([edge_attr, time_enc], dim=1)

    # Splits
    split_folder = paths["split_folder"]
    train_idx = torch.load(f"{split_folder}/train_edge_idx.pt").to(device)
    val_idx = torch.load(f"{split_folder}/val_edge_idx.pt").to(device)
    test_idx = torch.load(f"{split_folder}/test_edge_idx.pt").to(device)

    # Model
    model = GraphSAGE_T(
        node_dim=x.size(1),
        feat_dim=feat.size(1),
        cfg=model_cfg["model"]
    ).to(device)

    # Loss
    pw = compute_pos_weight(y_edge[train_idx])
    pw = torch.tensor(min(pw.item(), 100.0), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    # Optimizer
    opt_cfg = train_cfg.get("optimizer", {})
    lr = float(train_cfg.get("lr", 5e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    betas = tuple(float(b) for b in opt_cfg.get("betas", [0.9, 0.999]))
    eps = float(opt_cfg.get("eps", 1e-8))

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
    best_model_path = f"{results_dir}/best_model.pt"

    best_val = -1e9
    best_epoch = -1
    patience = 0
    max_patience = train_cfg.get("early_stopping_patience", 15)
    epochs = train_cfg.get("epochs", 100)

    print(f"\nStarting training for {epochs} epochs...\n")
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        train_loss = run_epoch_minibatch(
            model, optimizer, loss_fn,
            x, edge_index, feat, y_edge,
            train_idx, batch_size, device
        )

        val_metrics, _, val_loss = evaluate_minibatch(
            model, loss_fn,
            x, edge_index, feat, y_edge,
            val_idx, eval_batch_size, device, eval_cfg
        )

        val_p = val_metrics.get("precision", 0.0)
        val_r = val_metrics.get("recall", 0.0)
        val_f1 = val_metrics.get("f1", 0.0)
        val_roc = val_metrics.get("roc_auc", 0.0)
        val_aupr = val_metrics.get("aupr", 0.0)

        elapsed = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"P={val_p:.3f} R={val_r:.3f} F1={val_f1:.3f} "
            f"ROC-AUC={val_roc:.3f} AUPR={val_aupr:.3f} "
            f"time={elapsed:.2f}s"
        )

        # TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Val/Precision", val_p, epoch)
        writer.add_scalar("Val/Recall", val_r, epoch)
        writer.add_scalar("Val/F1", val_f1, epoch)
        writer.add_scalar("Val/ROC_AUC", val_roc, epoch)
        writer.add_scalar("Val/AUPR", val_aupr, epoch)
        writer.add_scalar("Time/epoch_seconds", elapsed, epoch)

        # Early stopping on AUPR
        if val_aupr > best_val:
            best_val = val_aupr
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience += 1

        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    total_time = time.perf_counter() - total_start
    print(f"\nTotal training time: {total_time:.2f}s ({total_time/60:.1f} min)")

    # Load best model + final evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("\nEvaluating final model...")

    train_m, train_p, train_loss_f = evaluate_minibatch(
        model, loss_fn, x, edge_index, feat, y_edge,
        train_idx, eval_batch_size, device, eval_cfg
    )
    val_m, val_p, val_loss_f = evaluate_minibatch(
        model, loss_fn, x, edge_index, feat, y_edge,
        val_idx, eval_batch_size, device, eval_cfg
    )
    test_m, test_p, test_loss_f = evaluate_minibatch(
        model, loss_fn, x, edge_index, feat, y_edge,
        test_idx, eval_batch_size, device, eval_cfg
    )

    print_metrics(train_m, experiment_name + " TRAIN")
    print_metrics(val_m, experiment_name + " VAL")
    print_metrics(test_m, experiment_name + " TEST")

    writer.close()
    if logger:
        logger.close()


if __name__ == "__main__":
    main()
