# train_graphsage_t.py
# -----------------------------------------------------------
# ASCII-safe GraphSAGE-T training script for AML edge detection.
# Temporal variant: uses timestamp sinusoidal encoding.
#
# Usage (RAT medium example):
#   python train_graphsage_t.py \
#       --config configs/models/graphsage_t.yaml \
#       --dataset configs/datasets/rat.yaml \
#       --intensity medium
#
# Baseline:
#   python train_graphsage_t.py \
#       --config configs/models/graphsage_t.yaml \
#       --dataset configs/datasets/baseline.yaml
# -----------------------------------------------------------

import os
import sys

# Add project root to Python path (one level above "scripts")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# Now utils can be imported as:
# from scripts.utils.config_utils import ...

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

# Suppress FutureWarnings to prevent infinite loop with Logger class
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# -----------------------------------------------------------
# Sinusoidal Time Encoding (TGAT-style)
# -----------------------------------------------------------

def build_sinusoidal_time_encoding(timestamps, time_dim=16):
    """
    TGAT-style sinusoidal time encoding.
    timestamps: [E] tensor of UNIX timestamps (seconds)
    Returns: [E, time_dim] time embedding
    """
    assert time_dim % 2 == 0, "time_dim must be even"

    ts = timestamps.float()

    # Normalize to [0,1]
    t_min = ts.min()
    t_max = ts.max()
    if t_max > t_min:
        ts_norm = (ts - t_min) / (t_max - t_min)
    else:
        ts_norm = torch.zeros_like(ts)

    ts_norm = ts_norm.view(-1, 1)

    # sinusoidal construction
    half = time_dim // 2
    div_term = torch.exp(
        torch.arange(half, dtype=torch.float32, device=ts.device)
        * -(np.log(10000.0) / half)
    ).view(1, -1)

    angles = ts_norm * div_term
    sin_part = torch.sin(angles)
    cos_part = torch.cos(angles)

    return torch.cat([sin_part, cos_part], dim=-1)


# -----------------------------------------------------------
# GraphSAGE-T Model Definitions
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


class TemporalEdgeClassifier(nn.Module):
    def __init__(self, node_emb_dim, edge_attr_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        in_dim = node_emb_dim * 2 + edge_attr_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h, edge_index, feat):
        src, dst = edge_index
        h_src = h[src]
        h_dst = h[dst]
        z = torch.cat([h_src, h_dst, feat], dim=-1)
        return self.mlp(z).view(-1)


class GraphSAGE_T_EdgeModel(nn.Module):
    def __init__(self, in_dim_node, in_dim_edge_plus_time, cfg_model):
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

        self.classifier = TemporalEdgeClassifier(
            node_emb_dim=hidden_dim,
            edge_attr_dim=in_dim_edge_plus_time,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x, edge_index, edge_attr_with_time):
        h = self.encoder(x, edge_index)
        return self.classifier(h, edge_index, edge_attr_with_time)


# -----------------------------------------------------------
# Training / Evaluation Helpers
# -----------------------------------------------------------

def compute_pos_weight(y_train):
    pos = (y_train == 1).sum().item()
    neg = (y_train == 0).sum().item()
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / max(pos, 1))


@torch.no_grad()
def evaluate_split(model, x, edge_index, feat, y_edge, split_idx, device, eval_cfg):
    model.eval()
    logits_all = model(x, edge_index, feat)
    logits = logits_all[split_idx]
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = y_edge[split_idx].cpu().numpy()

    metrics = evaluate_binary_classifier(
        y_true=labels,
        y_pred_probs=probs,
        threshold=eval_cfg.get("threshold", 0.5),
        auto_threshold=eval_cfg.get("auto_threshold", True),
        compute_top_k=eval_cfg.get("compute_top_k", True),
        k_values=eval_cfg.get("top_k_values", [100, 500, 1000]),
        verbose=False,
    )
    return metrics, probs


def run_epoch(model, optimizer, loss_fn,
              x, edge_index, feat,
              y_edge, train_idx):
    model.train()
    optimizer.zero_grad()

    logits_all = model(x, edge_index, feat)
    logits = logits_all[train_idx]
    labels = y_edge[train_idx].float()

    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ASCII-safe GraphSAGE-T training")
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
    eval_cfg = base_cfg["evaluation"]
    training_cfg = model_cfg["training"]
    loss_cfg = model_cfg["loss"]
    paths = setup["paths"]
    device = setup["device"]
    experiment_name = setup["experiment_name"]
    intensity = setup["intensity"]
    logger = setup.get("logger")

    # Time dim from YAML
    time_dim = model_cfg["model"].get("time_dim", 16)

    # TensorBoard
    tb_log_dir = os.path.join(paths["logs_dir"], "tb")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(tb_log_dir)

    print("TensorBoard log dir:", tb_log_dir)

    # ---------------------------------------
    # Load graph tensors
    # ---------------------------------------
    print("Loading graph tensors...")
    graph_folder = paths["graph_folder"]
    edge_index = torch.load(os.path.join(graph_folder, "edge_index.pt")).to(device)
    edge_attr = torch.load(os.path.join(graph_folder, "edge_attr.pt")).to(device)
    x = torch.load(os.path.join(graph_folder, "x.pt")).to(device)
    y_edge = torch.load(os.path.join(graph_folder, "y_edge.pt")).to(device)
    timestamps = torch.load(os.path.join(graph_folder, "timestamps.pt")).to(device)

    # Generate sinusoidal encoding
    print("Generating sinusoidal time encoding...")
    sinus = build_sinusoidal_time_encoding(timestamps, time_dim=time_dim)

    # Combine edge_attr + sinusoidal time
    feat = torch.cat([edge_attr, sinus], dim=1).to(device)

    # ---------------------------------------
    # Load splits
    # ---------------------------------------
    split_folder = paths["split_folder"]
    train_idx = torch.load(os.path.join(split_folder, "train_edge_idx.pt")).to(device)
    val_idx = torch.load(os.path.join(split_folder, "val_edge_idx.pt")).to(device)
    test_idx = torch.load(os.path.join(split_folder, "test_edge_idx.pt")).to(device)

    # ---------------------------------------
    # Model, Loss, Optimizer
    # ---------------------------------------
    model = GraphSAGE_T_EdgeModel(
        in_dim_node=x.size(1),
        in_dim_edge_plus_time=feat.size(1),
        cfg_model=model_cfg["model"],
    ).to(device)

    pos_weight = loss_cfg.get("pos_weight")
    if pos_weight is None:
        pos_weight = compute_pos_weight(y_edge[train_idx]).to(device)
    else:
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer_cfg = training_cfg.get("optimizer", {})
    betas = optimizer_cfg.get("betas", [0.9, 0.999])
    eps = optimizer_cfg.get("eps", 1e-8)
    
    # Convert to proper types (YAML may load as strings)
    lr = float(training_cfg.get("lr", 5e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    eps = float(eps)
    betas = tuple([float(b) for b in betas])

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    # ---------------------------------------
    # Train
    # ---------------------------------------
    results_dir = paths["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    best_val = -1e9
    best_epoch = -1
    patience = 0
    max_patience = training_cfg.get("early_stopping_patience", 5)
    epochs = training_cfg.get("epochs", 15)
    best_model_path = os.path.join(results_dir, "best_model.pt")

    print("Starting GraphSAGE-T training...")
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()

        loss = run_epoch(
            model, optimizer, loss_fn,
            x, edge_index, feat,
            y_edge, train_idx,
        )

        val_metrics, _ = evaluate_split(
            model, x, edge_index, feat,
            y_edge, val_idx, device, eval_cfg,
        )
        val_aupr = val_metrics["aupr"]
        val_f1 = val_metrics["f1"]

        epoch_time = time.perf_counter() - start

        print(
            "Epoch %03d | loss=%.4f | val_F1=%.4f | val_AUPR=%.4f | time=%.2fs"
            % (epoch, loss, val_f1, val_aupr, epoch_time)
        )

        # TensorBoard logs
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Val/AUPR", val_aupr, epoch)
        writer.add_scalar("Val/F1", val_f1, epoch)
        writer.add_scalar("Time/epoch_seconds", epoch_time, epoch)

        # Early stopping
        if val_aupr > best_val:
            best_val = val_aupr
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience += 1

        if patience >= max_patience:
            print("Early stopping at epoch", epoch)
            break

    total_time = time.perf_counter() - total_start
    print("Total training time: %.2f seconds" % total_time)
    writer.add_scalar("Time/total_seconds", total_time, 0)

    # ---------------------------------------
    # Evaluate best
    # ---------------------------------------
    print("Loading best model from:", best_model_path)
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("Evaluating on train/val/test...")

    train_metrics, train_probs = evaluate_split(
        model, x, edge_index, feat,
        y_edge, train_idx, device, eval_cfg,
    )
    val_metrics, val_probs = evaluate_split(
        model, x, edge_index, feat,
        y_edge, val_idx, device, eval_cfg,
    )
    test_metrics, test_probs = evaluate_split(
        model, x, edge_index, feat,
        y_edge, test_idx, device, eval_cfg,
    )

    print_metrics(train_metrics, experiment_name + " TRAIN")
    print_metrics(val_metrics, experiment_name + " VAL")
    print_metrics(test_metrics, experiment_name + " TEST")

    # ---------------------------------------
    # Save outputs
    # ---------------------------------------
    out = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "best_epoch": best_epoch,
        "best_val_aupr": float(best_val),
        "total_training_time_sec": float(total_time),
    }

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    torch.save(torch.tensor(train_probs), os.path.join(results_dir, "train_pred_probs.pt"))
    torch.save(torch.tensor(val_probs), os.path.join(results_dir, "val_pred_probs.pt"))
    torch.save(torch.tensor(test_probs), os.path.join(results_dir, "test_pred_probs.pt"))

    save_experiment_config(
        results_dir, base_cfg, model_cfg, dataset_cfg,
        intensity=intensity,
        additional_info={"total_training_time_sec": float(total_time)},
    )

    writer.close()

    print("Done.")

    if logger:
        logger.close()


if __name__ == "__main__":
    main()