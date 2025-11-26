# train_tgat.py
# -----------------------------------------------------------
# ASCII-safe TGAT-style training script for AML edge detection.
# Uses temporal event streams produced by graph_builder_tgat_events.py
# and chronological splits from create_splits.py.
#
# Usage (RAT medium example):
#   python train_tgat.py \
#       --config configs/models/tgat.yaml \
#       --dataset configs/datasets/rat.yaml \
#       --intensity medium
#
# Baseline:
#   python train_tgat.py \
#       --config configs/models/tgat.yaml \
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
from torch.utils.tensorboard import SummaryWriter

from scripts.utils.config_utils import (
    setup_experiment,
    save_experiment_config,
)
from scripts.utils.evaluation_utils import (
    evaluate_binary_classifier,
    print_metrics
)

# Suppress FutureWarnings to prevent infinite loop with Logger class
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# -----------------------------------------------------------
# Time encoding (sinusoidal, ASCII-safe)
# -----------------------------------------------------------

def build_sinusoidal_time_encoding(timestamps, time_dim=16):
    """
    Sinusoidal time encoding similar in spirit to TGAT.

    Args:
        timestamps: torch.LongTensor of shape [E]
                    UNIX timestamps in seconds (monotonic).
        time_dim:   dimension of time encoding (must be even)

    Returns:
        time_enc: torch.FloatTensor of shape [E, time_dim]
    """
    if time_dim % 2 != 0:
        raise ValueError("time_dim must be even for sinusoidal encoding")

    ts = timestamps.float()

    # Normalize timestamps to [0, 1] for numerical stability
    t_min = ts.min()
    t_max = ts.max()
    if t_max > t_min:
        ts_norm = (ts - t_min) / (t_max - t_min)
    else:
        ts_norm = torch.zeros_like(ts)

    # Expand to [E, 1]
    ts_norm = ts_norm.view(-1, 1)

    # Compute sinusoidal encodings
    dim_half = time_dim // 2
    div_term = torch.exp(
        torch.arange(dim_half, dtype=torch.float32, device=ts.device)
        * -(np.log(10000.0) / float(dim_half))
    ).view(1, -1)

    angles = ts_norm * div_term  # [E, dim_half]
    sin_part = torch.sin(angles)
    cos_part = torch.cos(angles)

    time_enc = torch.cat([sin_part, cos_part], dim=-1)  # [E, time_dim]
    return time_enc


# -----------------------------------------------------------
# Simple TGAT-style edge model (ASCII-safe)
# -----------------------------------------------------------

class NodeEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TGATEdgeClassifier(nn.Module):
    """
    Simplified TGAT-style edge model:
      - Node encoder produces static node embeddings from x_node.
      - Edge representation is concatenation of:
          [h_src, h_dst, edge_attr, time_encoding]
      - MLP produces edge logits.
    Note: This is a lightweight temporal model, not a full TGAT implementation
    with neighborhood attention and memory. It still leverages time encodings.
    """

    def __init__(self, in_dim_node, in_dim_edge, time_dim, cfg_model):
        super().__init__()

        hidden_dim = cfg_model["hidden_dim"]
        dropout = cfg_model.get("dropout", 0.2)

        self.time_dim = time_dim

        # Node encoder
        self.node_encoder = NodeEncoder(
            in_dim=in_dim_node,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # Optional projection for edge attributes
        self.edge_proj = nn.Linear(in_dim_edge, hidden_dim)

        # Final classifier MLP
        in_dim = hidden_dim * 2 + hidden_dim + time_dim  # src + dst + edge + time
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_node, src_nodes, dst_nodes, edge_attr, time_enc):
        """
        Args:
            x_node:   [N, F_n]
            src_nodes, dst_nodes: [E]
            edge_attr: [E, F_e]
            time_enc: [E, T]

        Returns:
            logits: [E]
        """
        h_nodes = self.node_encoder(x_node)        # [N, H]
        h_src = h_nodes[src_nodes]                # [E, H]
        h_dst = h_nodes[dst_nodes]                # [E, H]
        h_edge = self.edge_proj(edge_attr)        # [E, H]

        z = torch.cat([h_src, h_dst, h_edge, time_enc], dim=-1)
        logits = self.mlp(z).view(-1)
        return logits


# -----------------------------------------------------------
# Training helpers
# -----------------------------------------------------------

def compute_pos_weight(y_train):
    pos = (y_train == 1).sum().item()
    neg = (y_train == 0).sum().item()
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / max(pos, 1))


def run_epoch(model, optimizer, loss_fn,
              x_node, src_nodes, dst_nodes, edge_attr, time_enc,
              y_edge, train_idx):
    model.train()
    optimizer.zero_grad()

    logits_all = model(x_node, src_nodes, dst_nodes, edge_attr, time_enc)
    logits = logits_all[train_idx]
    labels = y_edge[train_idx].float()

    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate_split(model,
                   x_node, src_nodes, dst_nodes, edge_attr, time_enc,
                   y_edge, split_idx, eval_cfg):
    model.eval()

    logits_all = model(x_node, src_nodes, dst_nodes, edge_attr, time_enc)
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
        verbose=False
    )

    return metrics, probs


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ASCII-safe TGAT-style training")
    parser.add_argument("--config", required=True,
                        help="Path to model config (tgat.yaml)")
    parser.add_argument("--dataset", required=True,
                        help="Path to dataset config (rat.yaml, slt.yaml, baseline.yaml)")
    parser.add_argument("--intensity", default=None,
                        help="Dataset intensity level (low, medium, high)")
    parser.add_argument("--base_config", default="configs/base.yaml",
                        help="Path to base config YAML")
    args = parser.parse_args()

    # Setup experiment
    setup = setup_experiment(
        model_config_path=args.config,
        dataset_config_path=args.dataset,
        intensity=args.intensity,
        base_config_path=args.base_config,
        verbose=True,
        enable_logging=True
    )

    base_cfg = setup["base_cfg"]
    model_cfg = setup["model_cfg"]
    dataset_cfg = setup["dataset_cfg"]
    paths = setup["paths"]
    device = setup["device"]
    experiment_name = setup["experiment_name"]
    intensity = setup["intensity"]
    logger = setup.get("logger", None)

    eval_cfg = base_cfg["evaluation"]
    training_cfg = model_cfg["training"]
    loss_cfg = model_cfg["loss"]
    time_dim = model_cfg["model"].get("time_dim", 16)

    # TensorBoard writer
    tb_log_dir = os.path.join(paths["logs_dir"], "tb")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print("TensorBoard log dir:", tb_log_dir)

    # -------------------------------------------------------
    # Load TGAT graph tensors
    # -------------------------------------------------------
    print("Loading TGAT event tensors...")
    graph_folder = paths["graph_folder"]

    src_nodes = torch.load(os.path.join(graph_folder, "src_nodes.pt")).to(device)
    dst_nodes = torch.load(os.path.join(graph_folder, "dst_nodes.pt")).to(device)
    timestamps = torch.load(os.path.join(graph_folder, "timestamps.pt")).to(device)
    edge_attr = torch.load(os.path.join(graph_folder, "edge_attr.pt")).to(device)
    y_edge = torch.load(os.path.join(graph_folder, "y_edge.pt")).to(device)
    x_node = torch.load(os.path.join(graph_folder, "x_node.pt")).to(device)

    num_events = src_nodes.size(0)
    num_nodes = x_node.size(0)
    print("Num nodes:  %d" % num_nodes)
    print("Num events: %d" % num_events)

    # Build sinusoidal time encoding
    print("Generating sinusoidal time encoding...")
    time_enc = build_sinusoidal_time_encoding(timestamps, time_dim=time_dim)

    # -------------------------------------------------------
    # Load chronological splits (indices into events)
    # -------------------------------------------------------
    split_folder = paths["split_folder"]
    train_idx = torch.load(os.path.join(split_folder, "train_idx.pt")).to(device)
    val_idx = torch.load(os.path.join(split_folder, "val_idx.pt")).to(device)
    test_idx = torch.load(os.path.join(split_folder, "test_idx.pt")).to(device)

    # -------------------------------------------------------
    # Build model, loss, optimizer
    # -------------------------------------------------------
    model = TGATEdgeClassifier(
        in_dim_node=x_node.size(1),
        in_dim_edge=edge_attr.size(1),
        time_dim=time_dim,
        cfg_model=model_cfg["model"]
    ).to(device)

    pos_weight = loss_cfg.get("pos_weight", None)
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
        eps=eps
    )

    # -------------------------------------------------------
    # Training loop with early stopping on val AUPR
    # -------------------------------------------------------
    results_dir = paths["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    best_val = -1e9
    best_epoch = -1
    patience = 0
    max_patience = training_cfg.get("early_stopping_patience", 5)
    epochs = training_cfg.get("epochs", 15)
    best_model_path = os.path.join(results_dir, "best_model.pt")

    print("Starting TGAT-style training...")
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        loss = run_epoch(
            model, optimizer, loss_fn,
            x_node, src_nodes, dst_nodes, edge_attr, time_enc,
            y_edge, train_idx
        )

        val_metrics, _ = evaluate_split(
            model,
            x_node, src_nodes, dst_nodes, edge_attr, time_enc,
            y_edge, val_idx, eval_cfg
        )

        val_aupr = val_metrics["aupr"]
        val_f1 = val_metrics["f1"]
        epoch_time = time.perf_counter() - epoch_start

        print(
            "Epoch %03d | loss=%.4f | val_F1=%.4f | val_AUPR=%.4f | time=%.2fs"
            % (epoch, loss, val_f1, val_aupr, epoch_time)
        )

        # TensorBoard logging
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Val/F1", val_f1, epoch)
        writer.add_scalar("Val/AUPR", val_aupr, epoch)
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
            print("Early stopping triggered at epoch %d" % epoch)
            break

    total_time = time.perf_counter() - total_start
    print("Total training time: %.2f seconds" % total_time)
    writer.add_scalar("Time/total_seconds", total_time, 0)

    # -------------------------------------------------------
    # Final evaluation (best model)
    # -------------------------------------------------------
    print("Loading best model from:", best_model_path)
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("Evaluating on train/val/test splits...")

    train_metrics, train_probs = evaluate_split(
        model,
        x_node, src_nodes, dst_nodes, edge_attr, time_enc,
        y_edge, train_idx, eval_cfg
    )
    val_metrics, val_probs = evaluate_split(
        model,
        x_node, src_nodes, dst_nodes, edge_attr, time_enc,
        y_edge, val_idx, eval_cfg
    )
    test_metrics, test_probs = evaluate_split(
        model,
        x_node, src_nodes, dst_nodes, edge_attr, time_enc,
        y_edge, test_idx, eval_cfg
    )

    print_metrics(train_metrics, model_name=experiment_name + " TRAIN")
    print_metrics(val_metrics, model_name=experiment_name + " VAL")
    print_metrics(test_metrics, model_name=experiment_name + " TEST")

    # -------------------------------------------------------
    # Save metrics, predictions, and config
    # -------------------------------------------------------
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
        save_dir=results_dir,
        base_cfg=base_cfg,
        model_cfg=model_cfg,
        dataset_cfg=dataset_cfg,
        intensity=intensity,
        additional_info={"total_training_time_sec": float(total_time)},
        filename="experiment_config.json"
    )

    writer.close()

    print("Done.")

    if logger is not None:
        logger.close()


if __name__ == "__main__":
    main()