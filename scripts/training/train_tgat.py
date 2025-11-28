# train_tgat_GPU.py
# -----------------------------------------------------------
# GPU-optimized TGAT-style temporal edge classifier
# Stable version:
#   - Event-level self-attention (no N×N over all nodes)
#   - Minibatch over events
#   - No AMP by default (can be added later if needed)
#   - Gradient clipping
#   - Uses same logging / outputs as GraphSAGE & GraphSAGE-T
# -----------------------------------------------------------
# Usage
# Example: (default)
# python scripts/training/train_tgat.py \
#     --config configs/tgat.yaml \
#     --dataset configs/datasets/baseline.yaml \
#     --base_config configs/base.yaml
#
# Example: (low-intensity HiSmallRat)
# python scripts/training/train_tgat.py \
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
# Sinusoidal Time Encoding
# -----------------------------------------------------------

def build_sinusoidal_time_encoding(timestamps, time_dim=16):
    """TGAT-style sinusoidal time encoding over edge timestamps."""
    assert time_dim % 2 == 0, "time_dim must be even"

    ts = timestamps.float()
    t_min, t_max = ts.min(), ts.max()

    if t_max > t_min:
        ts_norm = (ts - t_min) / (t_max - t_min)
    else:
        ts_norm = torch.zeros_like(ts)

    ts_norm = ts_norm.view(-1, 1)  # [E, 1]

    half = time_dim // 2
    div = torch.exp(
        torch.arange(half, device=ts.device, dtype=torch.float32)
        * -(np.log(10000.0) / half)
    ).view(1, -1)  # [1, half]

    angles = ts_norm * div  # [E, half]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [E, time_dim]


# -----------------------------------------------------------
# Event-level Temporal Self-Attention
# -----------------------------------------------------------

class TemporalSelfAttentionLayer(nn.Module):
    """
    Multi-head self-attention over a sequence of events in a minibatch.

    Input:  x [B, N, C]  (we'll use B=1, N=batch_size, C=hidden_dim)
    Output: same shape
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, N, 3*C]
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]

        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, N, D]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # [B, H, N, D]
        out = out.transpose(1, 2).contiguous().view(B, N, C)  # [B, N, C]
        out = self.out_proj(out)
        return out


# -----------------------------------------------------------
# TGAT-style Edge Model (event-level)
# -----------------------------------------------------------

class TGATEdgeModel(nn.Module):
    """
    TGAT-style temporal edge classifier.

    For each event (edge) e:
      - project src & dst node features
      - concatenate with edge_attr and time encoding
      - run several self-attention layers over the minibatch of events
      - classify each event with an MLP
    """
    def __init__(self, node_in_dim, edge_in_dim, time_dim, cfg_model):
        super().__init__()

        hidden_dim = cfg_model["hidden_dim"]
        num_layers = cfg_model.get("num_layers", 2)
        num_heads = cfg_model.get("num_heads", 4)
        dropout = cfg_model.get("dropout", 0.1)

        # Project node features to hidden_dim
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)

        # Project concatenated event features to hidden_dim
        event_in_dim = hidden_dim * 2 + edge_in_dim + time_dim
        self.event_proj = nn.Linear(event_in_dim, hidden_dim)

        # Temporal self-attention over events in a minibatch
        self.layers = nn.ModuleList([
            TemporalSelfAttentionLayer(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Final edge classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward_batch(self, x_node, src_idx, dst_idx, edge_attr_batch, time_enc_batch):
        """
        x_node:         [N, F_n]
        src_idx/dst_idx:[B]
        edge_attr_batch:[B, F_e]
        time_enc_batch: [B, T]
        """
        # Gather node features for this batch of events
        src_x = x_node[src_idx]  # [B, F_n]
        dst_x = x_node[dst_idx]  # [B, F_n]

        # Project nodes
        src_h = self.node_proj(src_x)  # [B, H]
        dst_h = self.node_proj(dst_x)  # [B, H]

        # Build event representations
        feat = torch.cat([src_h, dst_h, edge_attr_batch, time_enc_batch], dim=-1)  # [B, event_in_dim]
        h = self.event_proj(feat)  # [B, H]

        # Self-attention over events in this minibatch
        h_seq = h.unsqueeze(0)  # [1, B, H] → B(batch_dim)=1, N(seq_len)=B
        for layer in self.layers:
            residual = h_seq
            h_seq = layer(h_seq)
            h_seq = self.norm(h_seq + residual)
            h_seq = self.dropout(h_seq)
        h_out = h_seq.squeeze(0)  # [B, H]

        # Classify each event
        logits = self.classifier(h_out).view(-1)  # [B]
        return logits


# -----------------------------------------------------------
# Training & Evaluation (minibatch, event-level)
# -----------------------------------------------------------

def run_epoch_minibatch(
    model, optimizer, loss_fn,
    x_node, src_nodes, dst_nodes, edge_attr, time_enc,
    y_edge, train_idx,
    batch_size, device
):
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Shuffle training events
    perm = torch.randperm(len(train_idx), device=device)
    idx = train_idx[perm]

    for start in range(0, len(idx), batch_size):
        end = min(start + batch_size, len(idx))
        batch_eids = idx[start:end]

        optimizer.zero_grad(set_to_none=True)

        src_batch = src_nodes[batch_eids]
        dst_batch = dst_nodes[batch_eids]
        edge_attr_batch = edge_attr[batch_eids]
        time_enc_batch = time_enc[batch_eids]
        labels_batch = y_edge[batch_eids].float()

        logits = model.forward_batch(x_node, src_batch, dst_batch, edge_attr_batch, time_enc_batch)
        loss = loss_fn(logits, labels_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_split_minibatch(
    model, x_node, src_nodes, dst_nodes, edge_attr, time_enc,
    y_edge, split_idx,
    batch_size, device, eval_cfg
):
    model.eval()
    all_probs = []

    for start in range(0, len(split_idx), batch_size):
        end = min(start + batch_size, len(split_idx))
        batch_eids = split_idx[start:end]

        src_batch = src_nodes[batch_eids]
        dst_batch = dst_nodes[batch_eids]
        edge_attr_batch = edge_attr[batch_eids]
        time_enc_batch = time_enc[batch_eids]

        logits = model.forward_batch(x_node, src_batch, dst_batch, edge_attr_batch, time_enc_batch)
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
    parser = argparse.ArgumentParser(description="GPU-optimized TGAT-style temporal edge classifier")
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
    intensity = setup.get("intensity")
    logger = setup.get("logger")

    time_dim = model_cfg["model"].get("time_dim", 16)

    # Batch sizes
    batch_size = training_cfg.get("batch_size", 2048)      # TGAT is heavier → smaller default
    eval_batch_size = training_cfg.get("eval_batch_size", 4096)

    print("\n" + "=" * 70)
    print("GPU Mini-Batch Training (TGAT-style)")
    print("=" * 70)
    print(f"Train batch size: {batch_size}")
    print(f"Eval batch size:  {eval_batch_size}")
    print(f"Mixed Precision:  False (disabled for stability)")
    print(f"Device:           {device}")
    print("=" * 70 + "\n")

    # TensorBoard
    tb_log_dir = os.path.join(paths["logs_dir"], "tb")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # -------------------------------------------------------
    # Load tensors
    # -------------------------------------------------------
    print("Loading TGAT tensors...")
    graph_folder = paths["graph_folder"]

    x_node = torch.load(os.path.join(graph_folder, "x_node.pt")).to(device)   # [N, F_n]
    src_nodes = torch.load(os.path.join(graph_folder, "src_nodes.pt")).to(device)  # [E]
    dst_nodes = torch.load(os.path.join(graph_folder, "dst_nodes.pt")).to(device)  # [E]
    edge_attr = torch.load(os.path.join(graph_folder, "edge_attr.pt")).to(device)  # [E, F_e]
    timestamps = torch.load(os.path.join(graph_folder, "timestamps.pt")).to(device)  # [E]
    y_edge = torch.load(os.path.join(graph_folder, "y_edge.pt")).to(device)   # [E]

    num_nodes = x_node.size(0)
    num_events = src_nodes.size(0)
    print(f"Num nodes:   {num_nodes:,}")
    print(f"Num events:  {num_events:,}")
    print(f"Node feats:  {x_node.size(1)}")
    print(f"Edge feats:  {edge_attr.size(1)}")

    # Time encoding
    print("Generating sinusoidal time encoding...")
    time_enc = build_sinusoidal_time_encoding(timestamps, time_dim=time_dim).to(device)  # [E, time_dim]

    # -------------------------------------------------------
    # Splits (reuse edge splits like GraphSAGE)
    # -------------------------------------------------------
    # TGAT uses temporal (chronological) splits
    split_folder_base = os.path.join(PROJECT_ROOT, "tgat_splits")
    dataset_name = os.path.basename(paths["graph_folder"])
    split_folder = os.path.join(split_folder_base, dataset_name)

    print(f"Loading TGAT temporal splits from: {split_folder}")

    train_idx = torch.load(os.path.join(split_folder, "train_edge_idx.pt")).to(device)
    val_idx = torch.load(os.path.join(split_folder, "val_edge_idx.pt")).to(device)
    test_idx = torch.load(os.path.join(split_folder, "test_edge_idx.pt")).to(device)

    print(f"Train events: {len(train_idx):,}")
    print(f"Val events:   {len(val_idx):,}")
    print(f"Test events:  {len(test_idx):,}")

    # -------------------------------------------------------
    # Model
    # -------------------------------------------------------
    model = TGATEdgeModel(
        node_in_dim=x_node.size(1),
        edge_in_dim=edge_attr.size(1),
        time_dim=time_dim,
        cfg_model=model_cfg["model"],
    ).to(device)

    # -------------------------------------------------------
    # Loss
    # -------------------------------------------------------
    pos_weight_cfg = loss_cfg.get("pos_weight", None)
    if pos_weight_cfg is None:
        pw = compute_pos_weight(y_edge[train_idx])
        pw = min(pw.item(), 100.0)
        pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor(float(pos_weight_cfg), dtype=torch.float32, device=device)

    print(f"pos_weight: {pos_weight.item():.2f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # -------------------------------------------------------
    # Optimizer (cast all to float to avoid YAML string issues)
    # -------------------------------------------------------
    opt_cfg = training_cfg.get("optimizer", {})

    lr = float(training_cfg.get("lr", 5e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))

    betas_raw = opt_cfg.get("betas", [0.9, 0.999])
    betas = tuple(float(b) for b in betas_raw)

    eps_raw = opt_cfg.get("eps", 1e-8)
    eps = float(eps_raw)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    # -------------------------------------------------------
    # Training loop
    # -------------------------------------------------------
    results_dir = paths["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    best_val = -1e9
    best_epoch = -1
    patience = 0
    max_patience = training_cfg.get("early_stopping_patience", 15)
    epochs = training_cfg.get("epochs", 100)
    best_model_path = os.path.join(results_dir, "best_model.pt")

    print(f"\nStarting training for {epochs} epochs...\n")
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        train_loss = run_epoch_minibatch(
            model, optimizer, loss_fn,
            x_node, src_nodes, dst_nodes, edge_attr, time_enc,
            y_edge, train_idx,
            batch_size, device,
        )

        val_metrics, _ = evaluate_split_minibatch(
            model, x_node, src_nodes, dst_nodes, edge_attr, time_enc,
            y_edge, val_idx,
            eval_batch_size, device, eval_cfg,
        )

        val_aupr = val_metrics["aupr"]
        val_f1 = val_metrics["f1"]
        epoch_time = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
            f"val_F1={val_f1:.4f} | val_AUPR={val_aupr:.4f} | "
            f"time={epoch_time:.2f}s"
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Val/F1", val_f1, epoch)
        writer.add_scalar("Val/AUPR", val_aupr, epoch)
        writer.add_scalar("Time/epoch_seconds", epoch_time, epoch)

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

    # -------------------------------------------------------
    # Final evaluation
    # -------------------------------------------------------
    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("Evaluating on train/val/test...")

    train_metrics, train_probs = evaluate_split_minibatch(
        model, x_node, src_nodes, dst_nodes, edge_attr, time_enc,
        y_edge, train_idx, eval_batch_size, device, eval_cfg,
    )
    val_metrics, val_probs = evaluate_split_minibatch(
        model, x_node, src_nodes, dst_nodes, edge_attr, time_enc,
        y_edge, val_idx, eval_batch_size, device, eval_cfg,
    )
    test_metrics, test_probs = evaluate_split_minibatch(
        model, x_node, src_nodes, dst_nodes, edge_attr, time_enc,
        y_edge, test_idx, eval_batch_size, device, eval_cfg,
    )

    print_metrics(train_metrics, model_name=f"{experiment_name} TRAIN")
    print_metrics(val_metrics, model_name=f"{experiment_name} VAL")
    print_metrics(test_metrics, model_name=f"{experiment_name} TEST")

    # Save metrics + probs
    out = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "best_epoch": best_epoch,
        "best_val_aupr": float(best_val),
        "total_training_time_sec": float(total_time),
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "use_amp": False,
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
            "use_amp": False,
        },
        filename="experiment_config.json",
    )

    writer.close()
    print("\n" + "=" * 70)
    print("TGAT Training Complete!")
    print("=" * 70)

    if logger:
        logger.close()


if __name__ == "__main__":
    main()
