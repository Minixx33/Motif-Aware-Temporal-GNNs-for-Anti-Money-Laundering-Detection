# train_graphsage_t_GPU.py
# -----------------------------------------------------------
# GPU-optimized GraphSAGE-T (Temporal GraphSAGE)
# Stable version:
#   - NO AMP (AMP caused NaNs)
#   - NO DETACH (full training)
#   - Encode nodes INSIDE each minibatch for temporal correctness
#   - Gradient clipping
#   - Handles 5M edges on RTX 4080
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

from scripts.utils.config_utils import (
    setup_experiment,
    save_experiment_config,
)
from scripts.utils.evaluation_utils import (
    evaluate_binary_classifier,
    print_metrics,
)

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


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
    div = torch.exp(
        torch.arange(half, device=ts.device) * -(np.log(10000.0) / half)
    ).view(1, -1)

    angles = ts_norm * div
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
        z = torch.cat([h_src, h_dst, feat], dim=-1)
        return self.mlp(z).view(-1)


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

    def classify(self, h, batch_edge_index, feat_batch):
        src, dst = batch_edge_index
        return self.classifier(h[src], h[dst], feat_batch)


# -----------------------------------------------------------
# Mini-batch training — STABLE VERSION (NO AMP, NO DETACH)
# -----------------------------------------------------------

def run_epoch_minibatch(
    model, optimizer, loss_fn,
    x, edge_index, feat, y, train_idx,
    batch_size, device
):
    model.train()

    total_loss = 0.0
    steps = 0

    # Shuffle edges
    perm = torch.randperm(len(train_idx), device=device)
    idx = train_idx[perm]

    for start in range(0, len(idx), batch_size):
        end = min(start + batch_size, len(idx))
        batch_edges = idx[start:end]

        optimizer.zero_grad(set_to_none=True)

        # Encode nodes INSIDE batch loop (temporal-aware)
        h = model.encode(x, edge_index)

        # Extract batch
        eidx = edge_index[:, batch_edges]
        fb = feat[batch_edges]
        yb = y[batch_edges].float()

        # Classify
        logits = model.classify(h, eidx, fb)
        loss = loss_fn(logits, yb)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / steps


@torch.no_grad()
def evaluate_minibatch(
    model, x, edge_index, feat, y, split_idx,
    batch_size, device, eval_cfg
):
    model.eval()

    # Encode once for evaluation
    h = model.encode(x, edge_index)

    probs = []

    for start in range(0, len(split_idx), batch_size):
        end = min(start + batch_size, len(split_idx))
        idx = split_idx[start:end]

        logits = model.classify(h, edge_index[:, idx], feat[idx])
        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p)

    probs = np.concatenate(probs)
    labels = y[split_idx].cpu().numpy()

    metrics = evaluate_binary_classifier(
        labels, probs,
        threshold=eval_cfg.get("threshold", 0.5),
        auto_threshold=eval_cfg.get("auto_threshold", True),
        compute_top_k=eval_cfg.get("compute_top_k", True),
        k_values=eval_cfg.get("top_k_values", [100,500,1000]),
        verbose=False
    )
    return metrics, probs

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
        verbose=True, enable_logging=True
    )

    base_cfg = setup["base_cfg"]
    model_cfg = setup["model_cfg"]
    dataset_cfg = setup["dataset_cfg"]
    eval_cfg = base_cfg["evaluation"]
    train_cfg = model_cfg["training"]
    loss_cfg = model_cfg["loss"]
    paths = setup["paths"]
    device = setup["device"]
    experiment_name = setup["experiment_name"]
    logger = setup.get("logger")

    # Batch sizes
    batch_size = train_cfg.get("batch_size", 8192)
    eval_batch_size = train_cfg.get("eval_batch_size", 16384)

    print("\n================ GRAPH SAGE-T TRAINING ================")
    print(f"Batch size: {batch_size}")
    print(f"Eval batch size: {eval_batch_size}")
    print(f"AMP: DISABLED")
    print(f"Device: {device}")
    print("=======================================================\n")

    writer = SummaryWriter(os.path.join(paths["logs_dir"], "tb"))

    # Load tensors
    graph_folder = paths["graph_folder"]
    x = torch.load(f"{graph_folder}/x.pt").to(device)
    edge_index = torch.load(f"{graph_folder}/edge_index.pt").to(device)
    edge_attr = torch.load(f"{graph_folder}/edge_attr.pt").to(device)
    y_edge = torch.load(f"{graph_folder}/y_edge.pt").to(device)
    timestamps = torch.load(f"{graph_folder}/timestamps.pt").to(device)

    # Time encoding
    tdim = model_cfg["model"].get("time_dim", 16)
    time_enc = build_sinusoidal_time_encoding(timestamps, tdim)
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
    pw = min(pw.item(), 100)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))

   # Optimizer configuration
    opt_cfg = train_cfg.get("optimizer", {})

    # Learning rate
    lr = float(train_cfg.get("lr", 5e-4))

    # Weight decay
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))

    # Betas
    betas_raw = opt_cfg.get("betas", [0.9, 0.999])
    betas = tuple(float(b) for b in betas_raw)

    # eps
    eps_raw = opt_cfg.get("eps", 1e-8)
    eps = float(eps_raw)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )



    # Training loop
    best_val = -1e9
    best_epoch = -1
    patience = 0
    max_patience = train_cfg.get("early_stopping_patience", 15)
    epochs = train_cfg.get("epochs", 100)
    best_path = f"{paths['results_dir']}/best_model.pt"

    print("\nStarting training...\n")
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        loss = run_epoch_minibatch(
            model, optimizer, loss_fn,
            x, edge_index, feat, y_edge,
            train_idx, batch_size, device
        )

        val_m, _ = evaluate_minibatch(
            model, x, edge_index, feat, y_edge,
            val_idx, eval_batch_size, device, eval_cfg
        )

        val_aupr, val_f1 = val_m["aupr"], val_m["f1"]

        print(
            f"Epoch {epoch:03d} | loss={loss:.4f} | "
            f"val_F1={val_f1:.4f} | val_AUPR={val_aupr:.4f} | "
            f"time={time.perf_counter() - t0:.1f}s"
        )

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Val/AUPR", val_aupr, epoch)
        writer.add_scalar("Val/F1", val_f1, epoch)

        if val_aupr > best_val:
            best_val = val_aupr
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1

        if patience >= max_patience:
            print("\nEarly stopping.")
            break

    # Final evaluation
    model.load_state_dict(torch.load(best_path, map_location=device))

    train_m, train_p = evaluate_minibatch(
        model, x, edge_index, feat, y_edge,
        train_idx, eval_batch_size, device, eval_cfg
    )
    val_m, val_p = evaluate_minibatch(
        model, x, edge_index, feat, y_edge,
        val_idx, eval_batch_size, device, eval_cfg
    )
    test_m, test_p = evaluate_minibatch(
        model, x, edge_index, feat, y_edge,
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
