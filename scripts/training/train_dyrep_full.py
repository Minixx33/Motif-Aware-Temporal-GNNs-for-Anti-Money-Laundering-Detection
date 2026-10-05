# train_dyrep_full.py
# -----------------------------------------------------------
# Full DyRep (memory-based temporal GNN) for edge-level AML classification,
# experiment tag causal_leakfix_v2. The old simplified model
# (train_dyrep.py / configs/models/dyrep.yaml, "DyRep-Lite") is untouched.
#
# Architecture -- DyRep (Trivedi et al., ICLR 2019) as instantiated in the TGN
# framework (Rossi et al., 2020; the "dyrep" setting of tgn/ in this repo):
#   * every account has a memory vector z_v (its DyRep representation),
#     initialised to zero -- no per-account ID embedding, so nothing is
#     memorised by account ID and late-appearing accounts are not noise
#   * every transaction (u, v, t, e) is an event; after it, u and v update their
#     memory with an RNN (DyRep's recurrent update):
#         msg_u = [z_u, emb_v(t), edge_feat(e), timeenc(t - t_u_last)]
#         z_u  <- RNNCell(mean of u's messages in the batch, z_u)
#     emb_v(t) is DyRep's localised structural aggregation: temporal
#     multi-head attention over v's K most recent incoming + outgoing
#     neighbours strictly before t (neighbour memory, edge features and
#     relative-time encoding) -- the TGN reproduction of DyRep's attention.
#   * readout is the memory itself (DyRep has no separate embedding layer)
#   * DyRep's own objective is kept as an auxiliary loss: the two-type
#     conditional intensity lambda_k(u,v) = psi_k * softplus(w_k.[z_u;z_v]/psi_k)
#     (k = association for the first-ever transfer between the pair,
#     communication otherwise), trained with -log lambda(event) + Monte-Carlo
#     survival term over randomly sampled non-event pairs.
#   * task head: edge classifier on [z_u, z_v, z_u*z_v, static node feats,
#     point-in-time node counts, edge features, pair history] -- the same
#     edge context the fixed GraphSAGE/GraphSAGE-T classifiers get.
#   * loss = BCE(pos_weight) + aux_weight * DyRep intensity loss
#
# Causality (strict, checked at run time on EVERY batch):
#   * events are processed in time order in batches that never split a
#     timestamp, so every event in batch b has a timestamp strictly greater
#     than every event in batches < b
#   * an event is scored from memory that contains ONLY batches < b
#     (TGN "memory update at start": batch b-1's messages are applied at the
#     start of batch b, with gradient, so the RNN/attention are trained)
#   * neighbour sampling, node counts and pair history use edges with
#     timestamp strictly earlier than the event
#   * labels are never part of any message or memory update
#   * one global event stream across splits: validation continues from the
#     memory reached by the training stream, test from validation's, with
#     timestamps shared across a split boundary handled correctly
#
# Data: same graphs/ + splits/ (chronological) and the same edge-feature
# preprocessing as the fixed GraphSAGE models (scripts/training/causal_sage.py):
# ts_normalized dropped, pf_code/rc_code embedded, train-fit standardisation,
# static degree columns dropped.
#
# Usage:
#   python scripts/training/train_dyrep_full.py \
#       --config configs/models/dyrep_full.yaml \
#       --dataset configs/datasets/baseline.yaml \
#       --base_config configs/base.yaml
# -----------------------------------------------------------

import os
import sys
import json
import time
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scripts.training.causal_sage import CausalGraph, TimeEncoder, EdgeEncoder, RELATIONS

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2


# ===========================================================
# Event stream (time-ordered batches that never split a timestamp)
# ===========================================================

class EventStream:
    def __init__(self, g, batch_size, verbose=True):
        log = print if verbose else (lambda *a, **k: None)
        E = g.num_edges
        ts = g.trel.cpu().numpy()

        split_of = np.full(E, -1, dtype=np.int64)
        for s, idx in ((SPLIT_TRAIN, g.train_idx), (SPLIT_VAL, g.val_idx), (SPLIT_TEST, g.test_idx)):
            idx = idx.cpu().numpy()
            if (split_of[idx] != -1).any():
                raise ValueError("an edge appears in more than one split")
            split_of[idx] = s
        if (split_of == -1).any():
            raise ValueError(f"{int((split_of == -1).sum())} edges are in no split")

        # chronological split required (a random/stratified split would make
        # "continue memory from train into val" meaningless)
        tr_max = ts[split_of == SPLIT_TRAIN].max()
        va_min, va_max = ts[split_of == SPLIT_VAL].min(), ts[split_of == SPLIT_VAL].max()
        te_min = ts[split_of == SPLIT_TEST].min()
        if not (tr_max <= va_min and va_max <= te_min):
            raise ValueError(
                "splits are not chronological (train max ts > val min ts or val max > test min). "
                "Use splits made with create_splits.py --split_mode chronological.")

        order = np.argsort(ts, kind="stable")
        ts_sorted = ts[order]
        change = np.flatnonzero(np.diff(ts_sorted)) + 1          # first index of each new timestamp
        bounds = [0]
        while bounds[-1] < E:
            target = bounds[-1] + batch_size
            if target >= E:
                bounds.append(E)
                break
            j = np.searchsorted(change, target, side="left")      # next timestamp start >= target
            bounds.append(int(change[j]) if j < len(change) else E)
        bounds = np.asarray(bounds, dtype=np.int64)

        # invariant: no timestamp straddles a batch boundary
        inner = bounds[1:-1]
        assert (ts_sorted[inner - 1] < ts_sorted[inner]).all()

        self.order = torch.from_numpy(order).to(g.device)
        self.bounds = bounds
        self.n_batches = len(bounds) - 1
        self.split_sorted = torch.from_numpy(split_of[order]).to(g.device)
        self.split_of = torch.from_numpy(split_of).to(g.device)

        # first/last batch containing each split
        bsplit = np.repeat(np.arange(self.n_batches), np.diff(bounds))
        so = split_of[order]
        self.first = {s: int(bsplit[so == s].min()) for s in (0, 1, 2)}
        self.last = {s: int(bsplit[so == s].max()) for s in (0, 1, 2)}
        sizes = np.diff(bounds)
        log(f"[stream] {self.n_batches:,} time-ordered batches (target {batch_size}, "
            f"median {int(np.median(sizes))}, max {int(sizes.max())}); "
            f"train batches {self.first[0]}-{self.last[0]}, val {self.first[1]}-{self.last[1]}, "
            f"test {self.first[2]}-{self.last[2]}")

    def batch(self, b):
        s, e = int(self.bounds[b]), int(self.bounds[b + 1])
        return self.order[s:e], self.split_sorted[s:e]


# ===========================================================
# Memory state
# ===========================================================

class MemoryState:
    def __init__(self, num_nodes, dim, device):
        self.M = torch.zeros(num_nodes, dim, device=device)
        self.last = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        self.pending = None           # event ids whose messages are not applied yet
        self.max_applied_t = -1       # latest event timestamp folded into memory

    def clone(self):
        c = MemoryState.__new__(MemoryState)
        c.M = self.M.clone()
        c.last = self.last.clone()
        c.pending = None if self.pending is None else self.pending.clone()
        c.max_applied_t = self.max_applied_t
        return c


# ===========================================================
# Model
# ===========================================================

class DyRepFull(nn.Module):
    def __init__(self, g, cfg):
        super().__init__()
        d = int(cfg.get("memory_dim", 128))
        t_dim = int(cfg.get("time_dim", 32))
        x_dim = int(cfg.get("node_feat_dim", 16))
        heads = int(cfg.get("attn_heads", 2))
        dropout = float(cfg.get("dropout", 0.1))
        self.K = int(cfg.get("n_neighbors", 10))          # per relation (in and out)
        self.d = d

        self.edge_enc = EdgeEncoder(len(g.cont_cols), g.cat_sizes, int(cfg.get("cat_dim", 8)))
        e_dim = self.edge_enc.out_dim
        self.time_enc = TimeEncoder(t_dim)
        self.x_proj = nn.Linear(g.x_ent.size(1), x_dim)

        # DyRep localised attention (TGN graph-attention, 1 layer)
        q_in = d + x_dim + t_dim
        kv_in = d + x_dim + e_dim + t_dim
        self.attn_dim = d
        self.q_lin = nn.Linear(q_in, d)
        self.k_lin = nn.Linear(kv_in, d)
        self.v_lin = nn.Linear(kv_in, d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.merge = nn.Sequential(nn.Linear(d + d + x_dim, d), nn.ReLU(), nn.Linear(d, d))

        # message + DyRep recurrent update
        msg_dim = d + d + e_dim + t_dim + 1
        self.rnn = nn.RNNCell(msg_dim, d, nonlinearity="tanh")

        # DyRep conditional intensity, k in {association, communication}
        self.omega = nn.ModuleList([nn.Linear(2 * d, 1) for _ in range(2)])
        self.psi_raw = nn.Parameter(torch.zeros(2))

        # edge classifier
        n_node_time = len(RELATIONS) + 2
        clf_in = 3 * d + 2 * x_dim + 2 * n_node_time + e_dim + g.pair_feat.size(1)
        H = int(cfg.get("hidden_dim", 128))
        self.classifier = nn.Sequential(
            nn.Linear(clf_in, H), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(H, H // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(H // 2, 1))

    # -------------------------------------------------------
    @staticmethod
    def node_time_feats(g, nodes, t):
        """Point-in-time (strictly < t) log-counts per relation + time since last event."""
        parts = [torch.log1p(g.prior_count(r, nodes, t).float()).unsqueeze(-1) for r in RELATIONS]
        last = g.last_event_time(nodes, t)
        has = (last >= 0)
        dt_h = torch.where(has, (t - last).float() / 3600.0, torch.zeros_like(t, dtype=torch.float))
        parts += [torch.log1p(dt_h).unsqueeze(-1), has.float().unsqueeze(-1)]
        return torch.cat(parts, dim=-1)

    def embed(self, g, M, nodes, t, check=False):
        """DyRep localised structural embedding of `nodes` at times t (strictly causal)."""
        nb, eb, mb = [], [], []
        for r in ("in", "out"):
            nbr, eid, mask, _ = g.sample(r, nodes, t, self.K, recent=True)
            nb.append(nbr); eb.append(eid); mb.append(mask)
        nbr = torch.cat(nb, 1); eid = torch.cat(eb, 1); mask = torch.cat(mb, 1)
        if check and mask.any():
            assert not ((g.trel[eid] >= t.view(-1, 1)) & mask).any(), \
                "attention neighbour edge not strictly earlier than the event"

        x_self = self.x_proj(g.x_ent[nodes])
        q = self.q_lin(torch.cat([M[nodes], x_self,
                                  self.time_enc(torch.zeros_like(t))], -1)).unsqueeze(1)
        dt = (t.view(-1, 1) - g.trel[eid]).clamp_min(0)
        kv = torch.cat([M[nbr], self.x_proj(g.x_ent[nbr]), self.edge_enc(g, eid),
                        self.time_enc(dt)], -1)
        k, v = self.k_lin(kv), self.v_lin(kv)
        has = mask.any(1)
        pad = ~mask
        pad[~has, 0] = False                       # avoid all-masked rows (NaN); zeroed below
        out, _ = self.attn(q, k, v, key_padding_mask=pad, need_weights=False)
        out = out.squeeze(1) * has.float().unsqueeze(-1)
        return self.merge(torch.cat([out, M[nodes], x_self], -1))

    def compute_updates(self, g, st, ev, check=False):
        """Memory updates from events `ev` (DyRep recurrence). Returns nodes, new memory, max t."""
        u, v, t = g.src[ev], g.dst[ev], g.trel[ev]
        a = torch.cat([u, v])                      # node being updated
        b = torch.cat([v, u])                      # its counterparty in the event
        tt = torch.cat([t, t])
        ee = torch.cat([ev, ev])
        emb_b = self.embed(g, st.M, b, tt, check=check)
        last_a = st.last[a]
        never = (last_a < 0)
        dt = torch.where(never, torch.zeros_like(tt), tt - last_a).clamp_min(0)
        msg = torch.cat([st.M[a], emb_b, self.edge_enc(g, ee), self.time_enc(dt),
                         never.float().unsqueeze(-1)], -1)
        uniq, inv = torch.unique(a, return_inverse=True)
        agg = torch.zeros(uniq.numel(), msg.size(1), device=msg.device).index_add_(0, inv, msg)
        cnt = torch.zeros(uniq.numel(), device=msg.device).index_add_(
            0, inv, torch.ones_like(tt, dtype=torch.float))
        agg = agg / cnt.unsqueeze(-1)
        new = self.rnn(agg, st.M[uniq])
        maxt = torch.full((uniq.numel(),), -1, dtype=torch.long, device=msg.device)
        maxt = maxt.scatter_reduce(0, inv, tt, reduce="amax", include_self=True)
        return uniq, new, maxt

    def intensity(self, z_u, z_v, k):
        psi = F.softplus(self.psi_raw) + 1e-3
        x = torch.cat([z_u, z_v], -1)
        w = torch.stack([self.omega[0](x).squeeze(-1), self.omega[1](x).squeeze(-1)], -1)
        w = w.gather(-1, k.view(-1, 1)).squeeze(-1)
        p = psi[k]
        return p * F.softplus(w / p)

    def classify(self, g, z_u, z_v, ev):
        u, v, t = g.src[ev], g.dst[ev], g.trel[ev]
        z = torch.cat([
            z_u, z_v, z_u * z_v,
            self.x_proj(g.x_ent[u]), self.x_proj(g.x_ent[v]),
            self.node_time_feats(g, u, t), self.node_time_feats(g, v, t),
            self.edge_enc(g, ev), g.pair_feat[ev]], -1)
        return self.classifier(z).view(-1)


# ===========================================================
# Stream processing
# ===========================================================

def step_batch(model, g, st, ev, train, aux_weight=0.0, check=True):
    """
    Process one time-ordered batch:
      1. apply pending messages (previous batch) -> updated memory for those nodes
         (with grad when train=True)
      2. score the batch's events from memory containing ONLY earlier batches
      3. queue this batch's events as pending
    Returns logits for ev, DyRep aux loss (or None), and a commit() callback that
    persists the memory update (call after backward).
    """
    t_min = int(g.trel[ev].min())
    if st.pending is not None and st.pending.numel():
        nodes_P, new_P, maxt_P = model.compute_updates(g, st, st.pending, check=check)
    else:
        nodes_P = new_P = maxt_P = None

    pos = torch.full((g.num_nodes,), -1, dtype=torch.long, device=ev.device)
    if nodes_P is not None:
        pos[nodes_P] = torch.arange(nodes_P.numel(), device=ev.device)

    def view(nodes):
        z = st.M[nodes]
        if nodes_P is None:
            return z
        p = pos[nodes]
        hit = p >= 0
        if hit.any():
            z = z.clone()
            z[hit] = new_P[p[hit]]
        return z

    if check:
        # everything folded into memory (incl. the pending batch) is strictly earlier
        applied = st.max_applied_t
        if maxt_P is not None:
            applied = max(applied, int(maxt_P.max()))
        assert applied < t_min, (
            f"causality violated: memory contains an event at t={applied} >= batch t_min={t_min}")

    u, v = g.src[ev], g.dst[ev]
    z_u, z_v = view(u), view(v)
    logits = model.classify(g, z_u, z_v, ev)

    aux = None
    if train and aux_weight > 0:
        k = ((g.pair_feat[ev, 0] > 0) | (g.pair_feat[ev, 1] > 0)).long()   # 0 assoc, 1 comm
        lam_pos = model.intensity(z_u, z_v, k)
        nv = torch.randint(0, g.num_nodes, (ev.numel(),), device=ev.device)
        nu = torch.randint(0, g.num_nodes, (ev.numel(),), device=ev.device)
        lam_neg = model.intensity(z_u, view(nv), k) + model.intensity(view(nu), z_v, k)
        aux = (-torch.log(lam_pos + 1e-9) + lam_neg)

    def commit():
        if nodes_P is not None:
            st.M[nodes_P] = new_P.detach()
            st.last[nodes_P] = torch.maximum(st.last[nodes_P], maxt_P)
            st.max_applied_t = max(st.max_applied_t, int(maxt_P.max()))
        st.pending = ev

    return logits, aux, commit


@torch.no_grad()
def run_eval_stream(model, g, stream, st, b_from, b_to, probs_out):
    """Eval-mode pass over batches [b_from, b_to]; writes sigmoid scores into probs_out[event_id]."""
    model.eval()
    for b in range(b_from, b_to + 1):
        ev, _ = stream.batch(b)
        logits, _, commit = step_batch(model, g, st, ev, train=False, check=True)
        probs_out[ev] = torch.sigmoid(logits)
        commit()
    return st


def train_epoch(model, g, stream, optimizer, loss_fn, aux_weight, clip):
    model.train()
    st = MemoryState(g.num_nodes, model.d, g.device)
    snapshot = None
    tot_cls, tot_aux, steps = 0.0, 0.0, 0
    for b in range(0, stream.last[SPLIT_TRAIN] + 1):
        if b == stream.first[SPLIT_VAL]:
            snapshot = st.clone()            # memory strictly before the first val batch
        ev, sp = stream.batch(b)
        logits, aux, commit = step_batch(model, g, st, ev, train=True, aux_weight=aux_weight,
                                         check=True)
        is_tr = sp == SPLIT_TRAIN
        if not bool(is_tr.any()):            # cannot happen for b <= last train batch; be safe
            with torch.no_grad():
                commit()
            continue
        loss_cls = loss_fn(logits[is_tr], g.y[ev][is_tr].float())
        loss = loss_cls
        if aux is not None:
            loss_aux = aux[is_tr].mean()
            loss = loss + aux_weight * loss_aux
            tot_aux += loss_aux.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        commit()
        tot_cls += loss_cls.item()
        steps += 1
    if snapshot is None:                     # no batch shared by train and val
        snapshot = st
    return tot_cls / max(steps, 1), tot_aux / max(steps, 1), snapshot


def metrics_for(probs_all, g, idx, eval_cfg, override_threshold=None):
    from scripts.utils.evaluation_utils import evaluate_binary_classifier
    p = probs_all[idx].cpu().numpy()
    yv = g.y[idx].cpu().numpy()
    if override_threshold is not None:
        kw = dict(threshold=override_threshold, auto_threshold=False)
    else:
        kw = dict(threshold=eval_cfg.get("threshold", 0.5),
                  auto_threshold=eval_cfg.get("auto_threshold", True))
    m = evaluate_binary_classifier(
        yv, p, compute_top_k=eval_cfg.get("compute_top_k", True),
        k_values=eval_cfg.get("top_k_values", [100, 500, 1000]), verbose=False, **kw)
    return m, p


# ===========================================================
# Main
# ===========================================================

def main():
    from torch.utils.tensorboard import SummaryWriter
    from scripts.utils.config_utils import setup_experiment, save_experiment_config
    from scripts.utils.evaluation_utils import print_metrics
    from scripts.utils.checkpoint_utils import (
        save_checkpoint, load_checkpoint, sync_experiment_to_drive, should_sync_this_epoch)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models/dyrep_full.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--intensity", default=None)
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--max_epochs", type=int, default=None, help="override epochs (smoke tests)")
    args = parser.parse_args()

    setup = setup_experiment(args.config, args.dataset, intensity=args.intensity,
                             base_config_path=args.base_config, verbose=True, enable_logging=True)
    base_cfg, model_cfg = setup["base_cfg"], setup["model_cfg"]
    dataset_cfg = setup.get("dataset_cfg", {})
    eval_cfg = base_cfg["evaluation"]
    train_cfg = model_cfg["training"]
    paths, device = setup["paths"], setup["device"]
    experiment_name, logger = setup["experiment_name"], setup.get("logger")

    batch_size = int(train_cfg.get("batch_size", 500))
    aux_weight = float(model_cfg["model"].get("aux_weight", 1.0))
    print("\n======= DYREP (full, causal v2) =======")
    print(f"Batch target: {batch_size} events (never splits a timestamp) | aux_weight: {aux_weight} "
          f"| Device: {device}")

    writer = SummaryWriter(os.path.join(paths["logs_dir"], "tb"))
    t0 = time.perf_counter()
    g = CausalGraph(paths["graph_folder"], paths["split_folder"], device)
    stream = EventStream(g, batch_size)
    print(f"[data] prepared in {time.perf_counter() - t0:.1f}s")
    train_idx, val_idx, test_idx = (g.train_idx.to(device), g.val_idx.to(device), g.test_idx.to(device))
    print(f"Train: {train_idx.numel():,}, Val: {val_idx.numel():,}, Test: {test_idx.numel():,}")

    model = DyRepFull(g, model_cfg["model"]).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} "
          f"(memory state {g.num_nodes:,} x {model.d}, not a parameter)")

    pw = (g.y[train_idx] == 0).sum().item() / max((g.y[train_idx] == 1).sum().item(), 1)
    pw = torch.tensor(min(pw, 100.0), dtype=torch.float32, device=device)
    print(f"pos_weight: {pw.item():.2f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    opt_cfg = train_cfg.get("optimizer", {})
    optimizer = optim.Adam(model.parameters(), lr=float(train_cfg.get("lr", 1e-4)),
                           weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
                           betas=tuple(float(b) for b in opt_cfg.get("betas", [0.9, 0.999])),
                           eps=float(opt_cfg.get("eps", 1e-8)))
    clip = float(train_cfg.get("gradient_clip", 1.0))

    results_dir = paths["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    best_model_path = os.path.join(results_dir, "best_model.pt")
    checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
    project_root = paths["root"]

    best_val, best_epoch, patience = -1e9, -1, 0
    max_patience = int(train_cfg.get("early_stopping_patience", 20))
    epochs = int(train_cfg.get("epochs", 200))
    if args.max_epochs is not None:
        epochs = args.max_epochs
    start_epoch = 1
    st_ck = load_checkpoint(checkpoint_path, model, optimizer, device)
    if st_ck is not None:
        start_epoch = st_ck["epoch"] + 1
        best_val, best_epoch, patience = st_ck["best_val"], st_ck["best_epoch"], st_ck["patience"]
        if patience >= max_patience:
            print("[RESUME] early stopping had already triggered; skipping training loop.")
            start_epoch = epochs + 1

    probs_all = torch.zeros(g.num_edges, device=device)
    total_start = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        t_ep = time.perf_counter()
        l_cls, l_aux, snap = train_epoch(model, g, stream, optimizer, loss_fn, aux_weight, clip)
        probs_all.zero_()
        run_eval_stream(model, g, stream, snap, stream.first[SPLIT_VAL], stream.last[SPLIT_VAL], probs_all)
        vm, _ = metrics_for(probs_all, g, val_idx, eval_cfg)
        del snap
        el = time.perf_counter() - t_ep
        print(f"Epoch {epoch:03d} | cls_loss={l_cls:.4f} dyrep_loss={l_aux:.4f} "
              f"P={vm.get('precision', 0):.3f} R={vm.get('recall', 0):.3f} F1={vm.get('f1', 0):.3f} "
              f"ROC-AUC={vm.get('roc_auc', 0):.3f} AUPR={vm.get('aupr', 0):.3f} time={el:.1f}s")
        writer.add_scalar("Loss/train_cls", l_cls, epoch)
        writer.add_scalar("Loss/train_dyrep", l_aux, epoch)
        for k_ in ("precision", "recall", "f1", "roc_auc", "aupr"):
            writer.add_scalar(f"Val/{k_}", vm.get(k_, 0.0), epoch)
        writer.add_scalar("Time/epoch_seconds", el, epoch)

        if vm.get("aupr", 0.0) > best_val:
            best_val, best_epoch, patience = vm["aupr"], epoch, 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience += 1
        save_checkpoint(checkpoint_path, epoch=epoch, model=model, optimizer=optimizer,
                        best_val=best_val, best_epoch=best_epoch, patience=patience)
        if should_sync_this_epoch(base_cfg, epoch):
            sync_experiment_to_drive(base_cfg, project_root, results_dir, paths["logs_dir"],
                                     label=experiment_name)
        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    total_time = time.perf_counter() - total_start
    print(f"\nTotal training time: {total_time:.1f}s ({total_time / 60:.1f} min)")

    # Final: best weights, one fresh eval-mode pass over the WHOLE stream
    # (train -> val -> test memory continuity), scores for every event.
    print(f"\nLoading best model from epoch {best_epoch}; final pass over the full event stream...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    probs_all.zero_()
    st = MemoryState(g.num_nodes, model.d, device)
    run_eval_stream(model, g, stream, st, 0, stream.n_batches - 1, probs_all)
    train_m, train_p = metrics_for(probs_all, g, train_idx, eval_cfg)
    val_m, val_p = metrics_for(probs_all, g, val_idx, eval_cfg)
    test_m, test_p = metrics_for(probs_all, g, test_idx, eval_cfg, override_threshold=val_m["threshold"])
    print_metrics(train_m, experiment_name + " TRAIN")
    print_metrics(val_m, experiment_name + " VAL")
    print_metrics(test_m, experiment_name + " TEST")

    out = {
        "train": train_m, "val": val_m, "test": test_m,
        "best_epoch": best_epoch, "best_val_aupr": float(best_val),
        "total_training_time_sec": float(total_time),
        "batch_size": batch_size, "aux_weight": aux_weight, "variant": "dyrep_full_causal_v2",
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    torch.save(torch.tensor(train_p), os.path.join(results_dir, "train_pred_probs.pt"))
    torch.save(torch.tensor(val_p), os.path.join(results_dir, "val_pred_probs.pt"))
    torch.save(torch.tensor(test_p), os.path.join(results_dir, "test_pred_probs.pt"))
    save_experiment_config(
        save_dir=results_dir, base_cfg=base_cfg, model_cfg=model_cfg, dataset_cfg=dataset_cfg,
        intensity=args.intensity,
        additional_info={"total_training_time_sec": float(total_time), "batch_size": batch_size,
                         "n_batches": stream.n_batches,
                         "categorical_edge_cols": g.cat_cols, "continuous_edge_cols": g.cont_cols},
        filename="experiment_config.json")
    writer.close()
    sync_experiment_to_drive(base_cfg, project_root, results_dir, paths["logs_dir"],
                             label=f"{experiment_name} (final)")
    print(f"\nDyRep (full) complete. Results: {results_dir}")
    if logger:
        logger.close()


if __name__ == "__main__":
    main()
