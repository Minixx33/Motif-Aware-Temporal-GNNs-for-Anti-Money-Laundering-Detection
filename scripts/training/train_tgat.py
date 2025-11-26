# train_tgat_GPU.py
# -----------------------------------------------------------
# GPU-optimized TGAT with mixed precision (AMP)
# Ready for RTX 4080!
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
from torch.cuda.amp import autocast, GradScaler

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
# Time Encoding
# -----------------------------------------------------------

def build_sinusoidal_time_encoding(timestamps, time_dim=16):
    """Sinusoidal time encoding"""
    if time_dim % 2 != 0:
        raise ValueError("time_dim must be even")

    ts = timestamps.float()

    t_min = ts.min()
    t_max = ts.max()
    if t_max > t_min:
        ts_norm = (ts - t_min) / (t_max - t_min)
    else:
        ts_norm = torch.zeros_like(ts)

    ts_norm = ts_norm.view(-1, 1)

    dim_half = time_dim // 2
    div_term = torch.exp(
        torch.arange(dim_half, dtype=torch.float32, device=ts.device)
        * -(np.log(10000.0) / float(dim_half))
    ).view(1, -1)

    angles = ts_norm * div_term
    sin_part = torch.sin(angles)
    cos_part = torch.cos(angles)

    time_enc = torch.cat([sin_part, cos_part], dim=-1)
    return time_enc


# -----------------------------------------------------------
# Temporal Attention Layer
# -----------------------------------------------------------

class TemporalAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        return out


# -----------------------------------------------------------
# TGAT Model
# -----------------------------------------------------------

class TGATNodeEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            TemporalAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.input_proj(x)
        h = h.unsqueeze(0)
        
        for layer in self.layers:
            h = h + layer(h)
            h = self.norm(h)
            h = self.dropout(h)
        
        return h.squeeze(0)


class EdgeClassifier(nn.Module):
    def __init__(self, node_emb_dim, edge_attr_dim, time_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        in_dim = node_emb_dim * 2 + edge_attr_dim + time_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, h_src, h_dst, edge_attr, time_enc):
        z = torch.cat([h_src, h_dst, edge_attr, time_enc], dim=-1)
        return self.mlp(z).view(-1)


class TGATEdgeClassifier(nn.Module):
    def __init__(self, in_dim_node, in_dim_edge, time_dim, cfg_model):
        super().__init__()

        hidden_dim = cfg_model["hidden_dim"]
        num_layers = cfg_model.get("num_layers", 2)
        num_heads = cfg_model.get("num_heads", 4)
        dropout = cfg_model.get("dropout", 0.1)

        self.encoder = TGATNodeEncoder(
            in_dim=in_dim_node,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        self.classifier = EdgeClassifier(
            node_emb_dim=hidden_dim,
            edge_attr_dim=in_dim_edge,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def encode_nodes(self, x_node):
        """Encode all nodes once"""
        return self.encoder(x_node)
    
    def classify_events(self, h, src_nodes, dst_nodes, edge_attr, time_enc):
        """Classify a batch of events"""
        h_src = h[src_nodes]
        h_dst = h[dst_nodes]
        return self.classifier(h_src, h_dst, edge_attr, time_enc)


# -----------------------------------------------------------
# Mini-batch training with AMP
# -----------------------------------------------------------

def run_epoch_minibatch(model, optimizer, loss_fn,
                       x_node, src_nodes, dst_nodes, edge_attr, time_enc,
                       y_edge, train_idx, batch_size, device, scaler=None):
    model.train()
    use_amp = scaler is not None
    
    # Encode nodes once with mixed precision
    with autocast(enabled=use_amp):
        h = model.encode_nodes(x_node)
    
    total_loss = 0
    num_batches = 0
    
    perm = torch.randperm(len(train_idx), device='cpu').to(device)
    shuffled_idx = train_idx[perm]
    
    for start in range(0, len(shuffled_idx), batch_size):
        end = min(start + batch_size, len(shuffled_idx))
        batch_idx = shuffled_idx[start:end]
        
        optimizer.zero_grad()
        
        # Get batch
        src_batch = src_nodes[batch_idx]
        dst_batch = dst_nodes[batch_idx]
        edge_attr_batch = edge_attr[batch_idx]
        time_enc_batch = time_enc[batch_idx]
        labels_batch = y_edge[batch_idx].float()
        
        # Forward with mixed precision
        with autocast(enabled=use_amp):
            logits = model.classify_events(h, src_batch, dst_batch, edge_attr_batch, time_enc_batch)
            loss = loss_fn(logits, labels_batch)
        
        # Backward with gradient scaling
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


@torch.no_grad()
def evaluate_split_minibatch(model, x_node, src_nodes, dst_nodes, edge_attr, time_enc,
                             y_edge, split_idx, batch_size, device, eval_cfg, use_amp=True):
    model.eval()
    
    # Encode nodes once with mixed precision
    with autocast(enabled=use_amp):
        h = model.encode_nodes(x_node)
    
    all_probs = []
    
    for start in range(0, len(split_idx), batch_size):
        end = min(start + batch_size, len(split_idx))
        batch_idx = split_idx[start:end]
        
        src_batch = src_nodes[batch_idx]
        dst_batch = dst_nodes[batch_idx]
        edge_attr_batch = edge_attr[batch_idx]
        time_enc_batch = time_enc[batch_idx]
        
        with autocast(enabled=use_amp):
            logits = model.classify_events(h, src_batch, dst_batch, edge_attr_batch, time_enc_batch)
        
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
        verbose=False
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
    parser = argparse.ArgumentParser(description="GPU-optimized TGAT with AMP")
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

    # Batch sizes and AMP
    batch_size = training_cfg.get("batch_size", 4096)
    eval_batch_size = training_cfg.get("eval_batch_size", 8192)
    use_amp = training_cfg.get("use_amp", True)
    
    print(f"\n{'='*70}")
    print(f"GPU Mini-Batch Training (TGAT)")
    print(f"{'='*70}")
    print(f"Train batch size: {batch_size}")
    print(f"Eval batch size:  {eval_batch_size}")
    print(f"Mixed Precision:  {use_amp}")
    print(f"Device:           {device}")
    print(f"{'='*70}\n")

    # Initialize gradient scaler
    scaler = GradScaler() if use_amp else None

    # TensorBoard
    tb_log_dir = os.path.join(paths["logs_dir"], "tb")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Load TGAT tensors
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
    print(f"Num nodes:  {num_nodes:,}")
    print(f"Num events: {num_events:,}")
    print(f"Edge features: {edge_attr.size(1)}")
    print(f"Node features: {x_node.size(1)}")

    # Build time encoding
    print("Generating sinusoidal time encoding...")
    time_enc = build_sinusoidal_time_encoding(timestamps, time_dim=time_dim)

    # Load splits
    split_folder = paths["split_folder"]
    train_idx = torch.load(os.path.join(split_folder, "train_idx.pt")).to(device)
    val_idx = torch.load(os.path.join(split_folder, "val_idx.pt")).to(device)
    test_idx = torch.load(os.path.join(split_folder, "test_idx.pt")).to(device)

    print(f"Train events: {len(train_idx):,}")
    print(f"Val events:   {len(val_idx):,}")
    print(f"Test events:  {len(test_idx):,}")

    # Model
    model = TGATEdgeClassifier(
        in_dim_node=x_node.size(1),
        in_dim_edge=edge_attr.size(1),
        time_dim=time_dim,
        cfg_model=model_cfg["model"]
    ).to(device)

    # Loss
    pos_weight = loss_cfg.get("pos_weight", None)
    if pos_weight is None:
        pos_weight = compute_pos_weight(y_edge[train_idx])
        pos_weight = min(pos_weight.item(), 100.0)
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    print(f"pos_weight: {pos_weight.item():.2f}")
    
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer_cfg = training_cfg.get("optimizer", {})
    lr = float(training_cfg.get("lr", 5e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    betas = tuple([float(b) for b in optimizer_cfg.get("betas", [0.9, 0.999])])
    eps = float(optimizer_cfg.get("eps", 1e-8))

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps
    )

    # Training loop
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

        loss = run_epoch_minibatch(
            model, optimizer, loss_fn,
            x_node, src_nodes, dst_nodes, edge_attr, time_enc,
            y_edge, train_idx, batch_size, device, scaler
        )

        val_metrics, _ = evaluate_split_minibatch(
            model, x_node, src_nodes, dst_nodes, edge_attr, time_enc,
            y_edge, val_idx, eval_batch_size, device, eval_cfg, use_amp
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

        if val_aupr > best_val:
            best_val = val_aupr
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  → New best (val_AUPR={val_aupr:.4f})")
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
        model, x_node, src_nodes, dst_nodes, edge_attr, time_enc,
        y_edge, train_idx, eval_batch_size, device, eval_cfg, use_amp
    )
    val_metrics, val_probs = evaluate_split_minibatch(
        model, x_node, src_nodes, dst_nodes, edge_attr, time_enc,
        y_edge, val_idx, eval_batch_size, device, eval_cfg, use_amp
    )
    test_metrics, test_probs = evaluate_split_minibatch(
        model, x_node, src_nodes, dst_nodes, edge_attr, time_enc,
        y_edge, test_idx, eval_batch_size, device, eval_cfg, use_amp
    )

    print_metrics(train_metrics, model_name=f"{experiment_name} TRAIN")
    print_metrics(val_metrics, model_name=f"{experiment_name} VAL")
    print_metrics(test_metrics, model_name=f"{experiment_name} TEST")

    # Save
    out = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "best_epoch": best_epoch,
        "best_val_aupr": float(best_val),
        "total_training_time_sec": float(total_time),
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "use_amp": use_amp
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
            "use_amp": use_amp
        },
        filename="experiment_config.json"
    )

    writer.close()
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)

    if logger:
        logger.close()


if __name__ == "__main__":
    main()