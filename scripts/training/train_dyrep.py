# train_dyrep.py
# -----------------------------------------------------------
# DyRep-style temporal event model for AML experiments
#
# This implements a *simplified* DyRep-like edge classifier:
#   - Learns node embeddings + projected node features
#   - Uses sinusoidal time encoding (like TGAT)
#   - Uses trainable event-type embeddings
#   - Classifies each event (edge) with an MLP
#
# It plugs into the SAME config + results pipeline as GraphSAGE-T:
#   - Uses setup_experiment(...) from config_utils
#   - Uses evaluate_binary_classifier / print_metrics
#   - Saves metrics.json, *_pred_probs.pt, experiment_config.json
#
# Expected DyRep graph folder structure (graphs_dyrep/<dataset_name>/):
#   - src.pt            [E]  (int64 node ids)
#   - dst.pt            [E]
#   - ts.pt             [E]  (int64 UNIX seconds)
#   - event_type.pt     [E]  (int64 event type id)
#   - edge_attr.pt      [E, F_e] (float32)
#   - node_features.pt  [N, F_n] (float32)
#   - labels.pt         [E]  (0/1 laundering edge)
#
# Expected splits folder (dyrep_splits/<dataset_name>/):
#   - train_edge_idx.pt [#train_edges]
#   - val_edge_idx.pt   [#val_edges]
#   - test_edge_idx.pt  [#test_edges]
#
# NOTE: This is *not* the full Hawkes-intensity DyRep objective
# (no survival term). It is a temporal event classifier with
# DyRep-style ingredients, tailored to your large-scale AML dataset.
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
from torch.utils.tensorboard import SummaryWriter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from scripts.utils.config_utils import setup_experiment, save_experiment_config
from scripts.utils.evaluation_utils import evaluate_binary_classifier, print_metrics


# -----------------------------------------------------------
# Time Encoding (same style as your GraphSAGE-T script)
# -----------------------------------------------------------

def build_sinusoidal_time_encoding(timestamps: torch.Tensor, time_dim: int = 16):
    """
    Sinusoidal time encoding (TGAT-style), using normalized timestamps.
    timestamps: [E] float or long tensor (UNIX seconds)
    returns: [E, time_dim]
    """
    assert time_dim % 2 == 0, "time_dim must be even"

    ts = timestamps.float()
    t_min, t_max = ts.min(), ts.max()

    if t_max > t_min:
        ts_norm = (ts - t_min) / (t_max - t_min)
    else:
        ts_norm = torch.zeros_like(ts)

    ts_norm = ts_norm.view(-1, 1)  # [E,1]

    half = time_dim // 2
    div_term = torch.exp(
        torch.arange(half, device=ts.device) * -(np.log(10000.0) / half)
    ).view(1, -1)  # [1, half]

    angles = ts_norm * div_term  # [E, half]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [E,time_dim]


# -----------------------------------------------------------
# DyRep-style Event Model
# -----------------------------------------------------------

class DyRepEventModel(nn.Module):
    """
    Simplified DyRep-style temporal event classifier for edges.

    For each event (u,v,t,type,edge_feat):
      - h_u = node_emb[u] + W_node * node_features[u]
      - h_v = node_emb[v] + W_node * node_features[v]
      - te  = time encoding for t
      - et  = event-type embedding

      logits = MLP([h_u, h_v, edge_feat, te, et])
    """

    def __init__(
        self,
        num_nodes: int,
        num_event_types: int,
        node_feat_dim: int,
        edge_feat_dim: int,
        model_cfg: dict,
    ):
        super().__init__()

        hidden_dim = int(model_cfg.get("hidden_dim", 128))
        dropout = float(model_cfg.get("dropout", 0.2))
        time_dim = int(model_cfg.get("time_dim", 16))
        type_emb_dim = int(model_cfg.get("type_embedding_dim", 16))

        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.type_emb_dim = type_emb_dim

        # Node feature projection + learnable node embedding
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)

        # Event type embedding
        self.type_emb = nn.Embedding(num_event_types, type_emb_dim)

        # Edge MLP
        in_dim = hidden_dim * 2 + edge_feat_dim + time_dim + type_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        edge_feat: torch.Tensor,
        time_enc: torch.Tensor,
        event_type: torch.Tensor,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        src_idx, dst_idx: [B] long
        edge_feat: [B, F_e]
        time_enc: [B, time_dim]
        event_type: [B]
        node_features: [N, F_n]
        """
        # Node encodings
        node_feat_src = self.node_proj(node_features[src_idx])  # [B,H]
        node_feat_dst = self.node_proj(node_features[dst_idx])  # [B,H]

        node_emb_src = self.node_emb(src_idx)                   # [B,H]
        node_emb_dst = self.node_emb(dst_idx)                   # [B,H]

        h_src = node_feat_src + node_emb_src
        h_dst = node_feat_dst + node_emb_dst

        # Event type embedding
        type_vec = self.type_emb(event_type)                    # [B,type_emb_dim]

        # Concatenate everything
        x = torch.cat([h_src, h_dst, edge_feat, time_enc, type_vec], dim=-1)  # [B, *]

        logits = self.mlp(x).view(-1)  # [B]
        return logits


# -----------------------------------------------------------
# Utility: compute pos_weight
# -----------------------------------------------------------

def compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    """
    Compute class weight for BCEWithLogitsLoss: pos_weight = N_neg / N_pos
    Clipped to avoid extreme values.
    """
    pos = (labels == 1).sum().item()
    neg = (labels == 0).sum().item()
    if pos == 0:
        return torch.tensor(1.0)
    pw = neg / max(pos, 1)
    return torch.tensor(min(pw, 100.0))


# -----------------------------------------------------------
# Training / Evaluation loops
# -----------------------------------------------------------

def run_epoch_minibatch(
    model: DyRepEventModel,
    optimizer,
    loss_fn,
    src,
    dst,
    ts_enc,
    edge_attr,
    event_type,
    labels,
    node_features,
    train_idx,
    batch_size,
    device,
):
    model.train()
    total_loss = 0.0
    steps = 0

    # Shuffle training edges
    perm = torch.randperm(train_idx.size(0), device=device)
    idx = train_idx[perm]

    for start in range(0, idx.size(0), batch_size):
        end = min(start + batch_size, idx.size(0))
        b_idx = idx[start:end]

        s_b = src[b_idx]
        d_b = dst[b_idx]
        t_b = ts_enc[b_idx]
        e_b = edge_attr[b_idx]
        et_b = event_type[b_idx]
        y_b = labels[b_idx].float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(s_b, d_b, e_b, t_b, et_b, node_features)
        loss = loss_fn(logits, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate_minibatch(
    model: DyRepEventModel,
    loss_fn,
    src,
    dst,
    ts_enc,
    edge_attr,
    event_type,
    labels,
    node_features,
    split_idx,
    batch_size,
    device,
    eval_cfg,
):
    model.eval()
    total_loss = 0.0
    steps = 0
    all_probs = []

    for start in range(0, split_idx.size(0), batch_size):
        end = min(start + batch_size, split_idx.size(0))
        idx = split_idx[start:end]

        s_b = src[idx]
        d_b = dst[idx]
        t_b = ts_enc[idx]
        e_b = edge_attr[idx]
        et_b = event_type[idx]
        y_b = labels[idx].float()

        logits = model(s_b, d_b, e_b, t_b, et_b, node_features)
        loss = loss_fn(logits, y_b)
        total_loss += loss.item()
        steps += 1

        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    y_true = labels[split_idx].cpu().numpy()

    metrics = evaluate_binary_classifier(
        y_true,
        all_probs,
        threshold=eval_cfg.get("threshold", 0.5),
        auto_threshold=eval_cfg.get("auto_threshold", True),
        compute_top_k=eval_cfg.get("compute_top_k", True),
        k_values=eval_cfg.get("top_k_values", [100, 500, 1000]),
        verbose=False,
    )

    avg_loss = total_loss / max(steps, 1)
    return metrics, all_probs, avg_loss


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Model config YAML (DyRep)")
    parser.add_argument("--dataset", required=True, help="Dataset config YAML")
    parser.add_argument("--intensity", default=None, help="Theory intensity (low/medium/high)")
    parser.add_argument("--base_config", default="configs/base.yaml", help="Base config path")
    args = parser.parse_args()

    # Full experiment setup: configs, paths, device, logger
    setup = setup_experiment(
        args.config,
        args.dataset,
        intensity=args.intensity,
        base_config_path=args.base_config,
        verbose=True,
        enable_logging=True,
    )

    base_cfg = setup["base_cfg"]
    model_cfg = setup["model_cfg"]
    dataset_cfg = setup["dataset_cfg"]
    eval_cfg = base_cfg["evaluation"]
    train_cfg = model_cfg["training"]
    paths = setup["paths"]
    device = setup["device"]
    experiment_name = setup["experiment_name"]
    logger = setup.get("logger", None)

    # We will NOT use paths["graph_folder"] because that is for static/TGAT.
    # Instead, derive DyRep graph path from base_cfg["paths"]["dyrep_graphs_dir"]
    root = paths["root"]
    ds_name = paths["dataset_name"]

    dyrep_graphs_dir_rel = base_cfg["paths"].get("dyrep_graphs_dir", "graphs_dyrep")
    dyrep_splits_dir_rel = base_cfg["paths"].get("dyrep_splits_dir", "dyrep_splits")

    graph_folder = os.path.join(root, dyrep_graphs_dir_rel, ds_name)
    split_folder = os.path.join(root, dyrep_splits_dir_rel, ds_name)

    results_dir = paths["results_dir"]
    logs_dir = paths["logs_dir"]

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    batch_size = int(train_cfg.get("batch_size", 8192))
    eval_batch_size = int(train_cfg.get("eval_batch_size", 16384))

    print("\n======= DYREP-STYLE EVENT MODEL TRAINING (FP32) =======")
    print(f"Graph folder:    {graph_folder}")
    print(f"Splits folder:   {split_folder}")
    print(f"Results folder:  {results_dir}")
    print(f"Batch size:      {batch_size}")
    print(f"Eval batch size: {eval_batch_size}")
    print(f"Device:          {device}")
    print("=======================================================\n")

    writer = SummaryWriter(os.path.join(logs_dir, "tb"))

    # -------------------------------------------------------
    # Load DyRep graph tensors
    # -------------------------------------------------------
    src = torch.load(os.path.join(graph_folder, "src.pt")).long().to(device)
    dst = torch.load(os.path.join(graph_folder, "dst.pt")).long().to(device)
    ts = torch.load(os.path.join(graph_folder, "ts.pt")).long().to(device)
    edge_attr = torch.load(os.path.join(graph_folder, "edge_attr.pt")).float().to(device)
    event_type = torch.load(os.path.join(graph_folder, "event_type.pt")).long().to(device)
    labels = torch.load(os.path.join(graph_folder, "labels.pt")).long().to(device)
    node_features = torch.load(os.path.join(graph_folder, "node_features.pt")).float().to(device)

    num_nodes = node_features.size(0)
    node_feat_dim = node_features.size(1)
    edge_feat_dim = edge_attr.size(1)
    num_event_types = int(event_type.max().item() + 1)

    print(f"Num nodes:        {num_nodes:,}")
    print(f"Num events/edges: {src.size(0):,}")
    print(f"Node feat dim:    {node_feat_dim}")
    print(f"Edge feat dim:    {edge_feat_dim}")
    print(f"Event types:      {num_event_types}")

    # Precompute time encoding (global)
    time_dim = int(model_cfg["model"].get("time_dim", 16))
    ts_enc = build_sinusoidal_time_encoding(ts, time_dim=time_dim).to(device)  # [E,time_dim]

    # -------------------------------------------------------
    # Load splits (indices into events)
    # -------------------------------------------------------
    train_idx = torch.load(os.path.join(split_folder, "train_edge_idx.pt")).long().to(device)
    val_idx = torch.load(os.path.join(split_folder, "val_edge_idx.pt")).long().to(device)
    test_idx = torch.load(os.path.join(split_folder, "test_edge_idx.pt")).long().to(device)

    print("\nSplit sizes:")
    print(f"  Train: {train_idx.size(0):,}")
    print(f"  Val:   {val_idx.size(0):,}")
    print(f"  Test:  {test_idx.size(0):,}")

    # -------------------------------------------------------
    # Model, loss, optimizer
    # -------------------------------------------------------
    model = DyRepEventModel(
        num_nodes=num_nodes,
        num_event_types=num_event_types,
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        model_cfg=model_cfg["model"],
    ).to(device)

    pw = compute_pos_weight(labels[train_idx]).to(device)
    print(f"\npos_weight (train) = {pw.item():.3f}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

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

    # -------------------------------------------------------
    # Training loop with early stopping on val AUPR
    # -------------------------------------------------------
    best_val = -1e9
    best_epoch = -1
    patience = 0
    max_patience = int(train_cfg.get("early_stopping_patience", 15))
    epochs = int(train_cfg.get("epochs", 100))

    best_model_path = os.path.join(results_dir, "best_model.pt")

    print(f"\nStarting training for up to {epochs} epochs...\n")
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        train_loss = run_epoch_minibatch(
            model,
            optimizer,
            loss_fn,
            src,
            dst,
            ts_enc,
            edge_attr,
            event_type,
            labels,
            node_features,
            train_idx,
            batch_size,
            device,
        )

        val_metrics, _, val_loss = evaluate_minibatch(
            model,
            loss_fn,
            src,
            dst,
            ts_enc,
            edge_attr,
            event_type,
            labels,
            node_features,
            val_idx,
            eval_batch_size,
            device,
            eval_cfg,
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

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Val/Precision", val_p, epoch)
        writer.add_scalar("Val/Recall", val_r, epoch)
        writer.add_scalar("Val/F1", val_f1, epoch)
        writer.add_scalar("Val/ROC_AUC", val_roc, epoch)
        writer.add_scalar("Val/AUPR", val_aupr, epoch)
        writer.add_scalar("Time/epoch_seconds", elapsed, epoch)

        # Early stopping on val AUPR
        if val_aupr > best_val:
            best_val = val_aupr
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience += 1

        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch} (no val AUPR improvement for {max_patience} epochs)")
            break

    total_time = time.perf_counter() - total_start
    print(f"\nTotal training time: {total_time:.2f}s ({total_time/60:.1f} min)")

    # -------------------------------------------------------
    # Final evaluation on best model
    # -------------------------------------------------------
    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("Evaluating final model on train/val/test...")

    train_m, train_p, _ = evaluate_minibatch(
        model,
        loss_fn,
        src,
        dst,
        ts_enc,
        edge_attr,
        event_type,
        labels,
        node_features,
        train_idx,
        eval_batch_size,
        device,
        eval_cfg,
    )
    val_m, val_p, _ = evaluate_minibatch(
        model,
        loss_fn,
        src,
        dst,
        ts_enc,
        edge_attr,
        event_type,
        labels,
        node_features,
        val_idx,
        eval_batch_size,
        device,
        eval_cfg,
    )
    test_m, test_p, _ = evaluate_minibatch(
        model,
        loss_fn,
        src,
        dst,
        ts_enc,
        edge_attr,
        event_type,
        labels,
        node_features,
        test_idx,
        eval_batch_size,
        device,
        eval_cfg,
    )

    print_metrics(train_m, experiment_name + " TRAIN")
    print_metrics(val_m, experiment_name + " VAL")
    print_metrics(test_m, experiment_name + " TEST")

    # -------------------------------------------------------
    # Save metrics, probabilities, and config
    # -------------------------------------------------------
    print("\nSaving results...")

    out = {
        "train": train_m,
        "val": val_m,
        "test": test_m,
        "best_epoch": best_epoch,
        "best_val_aupr": float(best_val),
        "total_training_time_sec": float(total_time),
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
    }

    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  - Saved metrics to: {metrics_path}")

    torch.save(torch.tensor(train_p), os.path.join(results_dir, "train_pred_probs.pt"))
    torch.save(torch.tensor(val_p), os.path.join(results_dir, "val_pred_probs.pt"))
    torch.save(torch.tensor(test_p), os.path.join(results_dir, "test_pred_probs.pt"))
    print("  - Saved prediction probabilities")

    save_experiment_config(
        save_dir=results_dir,
        base_cfg=base_cfg,
        model_cfg=model_cfg,
        dataset_cfg=dataset_cfg,
        intensity=args.intensity,
        additional_info={
            "total_training_time_sec": float(total_time),
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "graph_folder": graph_folder,
            "split_folder": split_folder,
        },
        filename="experiment_config.json",
    )
    print("  - Saved experiment config")

    writer.close()

    print("\n" + "=" * 70)
    print("DyRep-style Training Complete!")
    print(f"Results saved to: {results_dir}")
    print("=" * 70)

    if logger:
        logger.close()


if __name__ == "__main__":
    main()
