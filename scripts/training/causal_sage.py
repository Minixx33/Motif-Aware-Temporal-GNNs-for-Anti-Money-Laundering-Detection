# causal_sage.py
# -----------------------------------------------------------
# Fixed GraphSAGE / GraphSAGE-T for edge-level AML classification
# (experiment tag causal_leakfix_v2). Shared by train_graphsage_v2.py
# and train_graphsage_t_v2.py. The v1 scripts (train_graphsage.py,
# train_graphsage_t.py) are left untouched so v1 results stay reproducible.
#
# What changed vs v1 (numbers = flaws from the Oct 2026 architecture review):
#
#  (1) Edge features enter message passing: every message is
#      ReLU(W [h_neighbor, edge_feat(, time_enc)]), not just h_neighbor.
#  (2) Direction-aware: separate relations/weights for incoming edges
#      (who paid me), outgoing edges (who I paid) and self-loops.
#  (3) Aggregation = mean + max + degree-scaled mean (PNA-style scaler built
#      from the EXACT point-in-time count), plus exact log-counts in the node
#      input, so the model can count (2 vs 20 transfers is visible).
#  (4) Edge-specific context for the classifier: pair history of THIS
#      (src,dst) pair strictly before the edge (prior same-direction count,
#      prior reverse count, time since last same-pair transfer).
#  (5)/(7) Point-in-time correct encoder: each edge is scored with a
#      2-hop neighbourhood sampled ONLY from edges whose timestamp is strictly
#      earlier than that edge's timestamp (edges of any split -- structure and
#      features only, never labels). No train-period look-ahead, no stale
#      test-time graph, late-appearing accounts keep their real history.
#      (Replaces the v1 "train-edges-only" message-passing graph. Per-edge
#      cutoff is used instead of hourly snapshots: same guarantee, zero
#      staleness, and full-graph edge-MLP encodes would not fit in 24GB.)
#  (6) GraphSAGE lr raised to GraphSAGE-T's (set in the v2 yaml).
#  (8) GraphSAGE-T is temporal INSIDE the encoder: most-recent-K neighbour
#      sampling + learnable relative-time encoding of (t_edge - t_neighbor_edge)
#      on every message + time-since-last-activity node input. The absolute
#      timestamp sinusoid is gone. GraphSAGE (non-temporal) uses classic
#      uniform neighbour sampling over the same causal history and gets no
#      time encoding -- the only differences between the two are temporal.
#  (9) ts_normalized (absolute time, fit on the full date range) is dropped
#      from edge_attr at load time for both models.
#  Also: static train-only degree columns are dropped from x (replaced by the
#  exact point-in-time counts in (3)); pf_code / rc_code are embedded as
#  categories instead of being fed as ordinal numbers; continuous edge
#  features are standardised with TRAIN-split statistics (heavy-tailed
#  columns signed-log1p'd first, decision made on train rows only).
# -----------------------------------------------------------

import os
import sys
import json
import time
import math
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

RELATIONS = ("in", "out", "self")
N_DEGREE_COLS = 6            # builders always put the 6 degree columns first in x
DROP_EDGE_COLS = ("ts_normalized",)
CAT_EDGE_COLS = ("pf_code", "rc_code")
TIME_SHIFT = 1 << 21         # > max relative timestamp in seconds (~24 days)


# ===========================================================
# Data preparation
# ===========================================================

class CausalGraph:
    """
    Holds every tensor the encoder needs, on `device`:
      - per-relation temporal CSR (sorted by (node, time)) for causal sampling
      - preprocessed edge features (continuous + categorical codes)
      - pair-history features per edge
      - static node features (entity-type dummies only)
    """

    def __init__(self, graph_folder, split_folder, device, verbose=True):
        log = print if verbose else (lambda *a, **k: None)
        self.device = device

        edge_index = torch.load(os.path.join(graph_folder, "edge_index.pt")).long()
        edge_attr = torch.load(os.path.join(graph_folder, "edge_attr.pt")).float()
        x = torch.load(os.path.join(graph_folder, "x.pt")).float()
        y = torch.load(os.path.join(graph_folder, "y_edge.pt")).long()
        ts = torch.load(os.path.join(graph_folder, "timestamps.pt")).long()

        self.train_idx = torch.load(os.path.join(split_folder, "train_edge_idx.pt")).long()
        self.val_idx = torch.load(os.path.join(split_folder, "val_edge_idx.pt")).long()
        self.test_idx = torch.load(os.path.join(split_folder, "test_edge_idx.pt")).long()

        E = edge_index.size(1)
        N = x.size(0)
        assert edge_attr.size(0) == E and ts.size(0) == E and y.size(0) == E

        # v2 results must come from CHRONOLOGICAL splits (create_splits.py
        # --split_mode chronological). Refuse anything else, e.g. a stale
        # stratified-random splits/ folder.
        t_tr, t_va, t_te = ts[self.train_idx], ts[self.val_idx], ts[self.test_idx]
        if not (t_tr.max() <= t_va.min() and t_va.max() <= t_te.min()):
            raise ValueError(
                f"splits in {split_folder} are not chronological (train max ts > val min ts "
                f"or val max ts > test min ts). Recreate them with create_splits.py "
                f"--split_mode chronological.")
        if self.train_idx.numel() + self.val_idx.numel() + self.test_idx.numel() != E:
            raise ValueError("train+val+test sizes do not add up to the number of edges")
        self.num_nodes, self.num_edges = N, E
        log(f"[data] nodes={N:,} edges={E:,} edge_attr={tuple(edge_attr.shape)} x={tuple(x.shape)}")

        # ---------- edge feature columns ----------
        cols_path = os.path.join(graph_folder, "edge_attr_cols.json")
        if not os.path.exists(cols_path):
            raise FileNotFoundError(
                f"{cols_path} missing -- needed to locate ts_normalized / pf_code / "
                f"rc_code by name. Every current builder writes it; rebuild the graph.")
        with open(cols_path) as f:
            cols = json.load(f)
        assert len(cols) == edge_attr.size(1), "edge_attr_cols.json does not match edge_attr width"

        drop = [c for c in cols if c in DROP_EDGE_COLS]
        cat_cols = [c for c in cols if c in CAT_EDGE_COLS]
        cont_cols = [c for c in cols if c not in DROP_EDGE_COLS and c not in CAT_EDGE_COLS]
        log(f"[data] dropping {drop}; categorical {cat_cols}; {len(cont_cols)} continuous edge cols")
        self.cont_cols, self.cat_cols = cont_cols, cat_cols

        cont = edge_attr[:, [cols.index(c) for c in cont_cols]].clone()
        tr = self.train_idx
        # signed log1p for heavy-tailed columns (decided on TRAIN rows only)
        absmax = cont[tr].abs().max(dim=0).values
        heavy = absmax > 50.0
        if heavy.any():
            log(f"[data] signed-log1p on heavy-tailed cols: "
                f"{[c for c, h in zip(cont_cols, heavy.tolist()) if h]}")
            cont[:, heavy] = torch.sign(cont[:, heavy]) * torch.log1p(cont[:, heavy].abs())
        mu = cont[tr].mean(dim=0)
        sd = cont[tr].std(dim=0)
        sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
        cont = ((cont - mu) / sd).clamp_(-10.0, 10.0)
        self.edge_cont = cont.to(device)

        if cat_cols:
            codes = edge_attr[:, [cols.index(c) for c in cat_cols]].round().long().clamp_min(0)
            self.cat_sizes = [int(codes[:, j].max().item()) + 1 for j in range(codes.size(1))]
            self.edge_cat = codes.to(device)
        else:
            self.cat_sizes = []
            self.edge_cat = torch.zeros(E, 0, dtype=torch.long, device=device)

        # ---------- node features: drop static degree block ----------
        if x.size(1) > N_DEGREE_COLS:
            x_ent = x[:, N_DEGREE_COLS:]
        else:
            x_ent = torch.zeros(N, 1)
        log(f"[data] static node features kept (entity dummies): {x_ent.size(1)} "
            f"(dropped {min(N_DEGREE_COLS, x.size(1))} static degree cols)")
        self.x_ent = x_ent.to(device)

        # ---------- timestamps ----------
        self.t0 = int(ts.min().item())
        trel = ts - self.t0
        assert int(trel.max().item()) < TIME_SHIFT, "time span exceeds TIME_SHIFT; raise it"
        self.trel = trel.to(device)
        self.y = y.to(device)

        src, dst = edge_index[0], edge_index[1]
        self.src, self.dst = src.to(device), dst.to(device)

        # ---------- per-relation temporal CSR ----------
        self.rel = {}
        self_mask = src == dst
        eids = torch.arange(E)
        specs = {
            "out": (~self_mask, src, dst),     # node = payer, neighbour = payee
            "in": (~self_mask, dst, src),      # node = payee, neighbour = payer
            "self": (self_mask, src, src),
        }
        for r, (m, node, nbr) in specs.items():
            e = eids[m]
            key = node[m] * TIME_SHIFT + trel[m]
            order = torch.argsort(key, stable=True)
            self.rel[r] = {
                "key": key[order].contiguous().to(device),
                "nbr": nbr[m][order].contiguous().to(device),
                "eid": e[order].contiguous().to(device),
            }
            log(f"[data] relation {r:>4}: {int(m.sum()):,} edges")

        # PNA-style degree scaler normaliser per relation (mean log1p count at
        # the moment each TRAIN edge happens, for its src and dst)
        self.delta = {}
        with torch.no_grad():
            tr_d = self.train_idx.to(device)
            nodes = torch.cat([self.src[tr_d], self.dst[tr_d]])
            tt = torch.cat([self.trel[tr_d], self.trel[tr_d]])
            for r in RELATIONS:
                c = self.prior_count(r, nodes, tt).float()
                self.delta[r] = max(float(torch.log1p(c).mean().item()), 1e-2)
        log(f"[data] degree-scaler deltas: { {k: round(v, 3) for k, v in self.delta.items()} }")

        # ---------- pair history features (CPU, numpy) ----------
        self.pair_feat = self._pair_history(src.numpy(), dst.numpy(), trel.numpy(), N).to(device)
        log(f"[data] pair-history features: {self.pair_feat.size(1)}")

    # -------------------------------------------------------
    @staticmethod
    def _pair_history(src, dst, trel, N):
        """
        For every edge e=(s,d,t): number of earlier (strictly ts<t) s->d edges,
        number of earlier d->s edges, and time since the last earlier s->d edge.
        Pure function of timestamps/structure -- no labels.
        """
        pk = src.astype(np.int64) * N + dst.astype(np.int64)
        rk = dst.astype(np.int64) * N + src.astype(np.int64)
        tt = trel.astype(np.int64)
        keys = np.sort(pk * TIME_SHIFT + tt)

        lo_same = np.searchsorted(keys, pk * TIME_SHIFT, side="left")
        hi_same = np.searchsorted(keys, pk * TIME_SHIFT + tt, side="left")
        prior_same = hi_same - lo_same

        lo_rev = np.searchsorted(keys, rk * TIME_SHIFT, side="left")
        hi_rev = np.searchsorted(keys, rk * TIME_SHIFT + tt, side="left")
        prior_rev = hi_rev - lo_rev

        has_prior = prior_same > 0
        last_t = np.zeros_like(tt)
        last_t[has_prior] = keys[hi_same[has_prior] - 1] - pk[has_prior] * TIME_SHIFT
        dt_h = np.where(has_prior, (tt - last_t) / 3600.0, 0.0)

        feat = np.stack([
            np.log1p(prior_same),
            np.log1p(prior_rev),
            np.log1p(dt_h),
            has_prior.astype(np.float64),
        ], axis=1).astype(np.float32)
        return torch.from_numpy(feat)

    # -------------------------------------------------------
    def _range(self, r, nodes, t):
        """[start, end) of node's relation-r edges with timestamp strictly < t."""
        key = self.rel[r]["key"]
        base = nodes * TIME_SHIFT
        start = torch.searchsorted(key, base)
        end = torch.searchsorted(key, base + t)      # left: excludes ts == t
        return start, end

    def prior_count(self, r, nodes, t):
        s, e = self._range(r, nodes, t)
        return e - s

    def last_event_time(self, nodes, t):
        """Latest relative timestamp < t over all relations (-1 if none)."""
        last = torch.full_like(t, -1)
        for r in RELATIONS:
            s, e = self._range(r, nodes, t)
            has = e > s
            if has.any():
                k = self.rel[r]["key"][(e - 1).clamp_min(0)]
                tl = k - nodes * TIME_SHIFT
                last = torch.where(has, torch.maximum(last, tl), last)
        return last

    def sample(self, r, nodes, t, K, recent, generator=None):
        """
        Sample up to K relation-r edges of each node with timestamp < t.
        recent=True  -> the K most recent such edges (GraphSAGE-T)
        recent=False -> uniform over all such edges (classic GraphSAGE)
        Returns nbr [M,K], eid [M,K], mask [M,K] (bool), count [M].
        """
        M = nodes.numel()
        dev = nodes.device
        s, e = self._range(r, nodes, t)
        c = e - s
        ar = torch.arange(K, device=dev).view(1, K)
        if K == 0 or M == 0:
            z = torch.zeros(M, K, dtype=torch.long, device=dev)
            return z, z, z.bool(), c
        if recent:
            pos = e.view(-1, 1) - K + ar
            mask = pos >= s.view(-1, 1)
        else:
            seq = s.view(-1, 1) + ar
            rnd = s.view(-1, 1) + (torch.rand(M, K, device=dev, generator=generator)
                                   * c.view(-1, 1).float()).long()
            big = (c > K).view(-1, 1)
            pos = torch.where(big, rnd, seq)
            mask = torch.where(big, torch.ones_like(seq, dtype=torch.bool), ar < c.view(-1, 1))
        L = self.rel[r]["key"].numel()
        if L == 0:
            z = torch.zeros(M, K, dtype=torch.long, device=dev)
            return z, z, torch.zeros(M, K, dtype=torch.bool, device=dev), c
        pos = pos.clamp(0, L - 1)
        nbr = self.rel[r]["nbr"][pos]
        eid = self.rel[r]["eid"][pos]
        return nbr, eid, mask, c


# ===========================================================
# Model
# ===========================================================

class TimeEncoder(nn.Module):
    """GraphMixer/TGAT-style cos(w * dt + b), dt in seconds, learnable w,b."""

    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(1, dim)
        with torch.no_grad():
            self.lin.weight.copy_(
                torch.from_numpy(1.0 / 10 ** np.linspace(0, 9, dim)).float().view(dim, 1))
            self.lin.bias.zero_()

    def forward(self, dt):
        return torch.cos(self.lin(dt.unsqueeze(-1).float()))


class EdgeEncoder(nn.Module):
    def __init__(self, n_cont, cat_sizes, cat_dim=8):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, cat_dim) for n in cat_sizes])
        self.out_dim = n_cont + cat_dim * len(cat_sizes)

    def forward(self, g, eid):
        parts = [g.edge_cont[eid]]
        if len(self.embs):
            codes = g.edge_cat[eid]
            for j, emb in enumerate(self.embs):
                parts.append(emb(codes[..., j]))
        return torch.cat(parts, dim=-1)


class CausalSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, e_dim, t_dim, dropout):
        super().__init__()
        self.msg = nn.ModuleDict({
            r: nn.Linear(in_dim + e_dim + t_dim, out_dim) for r in RELATIONS})
        self.agg = nn.ModuleDict({r: nn.Linear(3 * out_dim, out_dim) for r in RELATIONS})
        self.root = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h_root, neigh, delta):
        """
        h_root: [M, in]
        neigh[r] = (h_nbr [M,K,in], e [M,K,e_dim], tenc [M,K,t_dim] or None,
                    mask [M,K] bool, count [M])
        """
        out = self.root(h_root)
        for r in RELATIONS:
            h_n, e, te, mask, cnt = neigh[r]
            if h_n.size(1) == 0:
                continue
            inp = [h_n, e] + ([te] if te is not None else [])
            m = torch.relu(self.msg[r](torch.cat(inp, dim=-1)))          # [M,K,H]
            mf = mask.unsqueeze(-1).float()
            n_s = mf.sum(1).clamp_min(1.0)
            mean = (m * mf).sum(1) / n_s
            mx = m.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
            mx = torch.where(mask.any(1, keepdim=True), mx, torch.zeros_like(mx))
            scaler = (torch.log1p(cnt.float()) / delta[r]).unsqueeze(-1)
            out = out + self.agg[r](torch.cat([mean, mx, mean * scaler], dim=-1))
        return self.drop(torch.relu(self.norm(out)))


class CausalSAGEEdgeModel(nn.Module):
    def __init__(self, g, cfg, temporal):
        super().__init__()
        H = int(cfg.get("hidden_dim", 128))
        dropout = float(cfg.get("dropout", 0.2))
        self.temporal = temporal
        self.fanout = [dict(f) for f in cfg.get("fanout", [
            {"in": 10, "out": 10, "self": 2},
            {"in": 5, "out": 5, "self": 1}])]
        assert len(self.fanout) == 2, "this implementation is 2-layer"
        self.t_dim = int(cfg.get("time_dim", 32)) if temporal else 0
        self.delta = g.delta

        self.edge_enc = EdgeEncoder(len(g.cont_cols), g.cat_sizes, int(cfg.get("cat_dim", 8)))
        e_dim = self.edge_enc.out_dim
        self.time_enc = TimeEncoder(self.t_dim) if temporal else None

        n_time_node = 2 if temporal else 0
        self.h0_dim = g.x_ent.size(1) + len(RELATIONS) + n_time_node

        self.layer1 = CausalSAGELayer(self.h0_dim, H, e_dim, self.t_dim, dropout)
        self.layer2 = CausalSAGELayer(H, H, e_dim, self.t_dim, dropout)

        clf_in = 3 * H + e_dim + g.pair_feat.size(1)
        self.classifier = nn.Sequential(
            nn.Linear(clf_in, H), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(H, H // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(H // 2, 1),
        )

    # -------------------------------------------------------
    def node_input(self, g, nodes, t):
        parts = [g.x_ent[nodes]]
        for r in RELATIONS:
            parts.append(torch.log1p(g.prior_count(r, nodes, t).float()).unsqueeze(-1))
        if self.temporal:
            last = g.last_event_time(nodes, t)
            has = (last >= 0).float()
            dt_h = torch.where(last >= 0, (t - last).float() / 3600.0, torch.zeros_like(t).float())
            parts += [torch.log1p(dt_h).unsqueeze(-1), has.unsqueeze(-1)]
        return torch.cat(parts, dim=-1)

    def _sample_level(self, g, nodes, t, fan, gen):
        out = {}
        for r in RELATIONS:
            out[r] = g.sample(r, nodes, t, int(fan.get(r, 0)), recent=self.temporal, generator=gen)
        return out

    def _neigh(self, g, samp, t, h_nbr_by_rel):
        neigh = {}
        for r in RELATIONS:
            nbr, eid, mask, cnt = samp[r]
            e = self.edge_enc(g, eid)
            te = None
            if self.temporal:
                dt = (t.view(-1, 1) - g.trel[eid]).clamp_min(0)
                te = self.time_enc(dt)
            neigh[r] = (h_nbr_by_rel[r], e, te, mask, cnt)
        return neigh

    def encode(self, g, roots, t, gen=None, return_samples=False):
        # level 1 (neighbours of roots), level 2 (neighbours of level-1 nodes),
        # every hop sampled with the ROOT's cutoff t -> strictly causal.
        s1 = self._sample_level(g, roots, t, self.fanout[0], gen)
        n1_nodes, n1_t, n1_shape = {}, {}, {}
        for r in RELATIONS:
            nbr = s1[r][0]
            n1_shape[r] = nbr.shape
            n1_nodes[r] = nbr.reshape(-1)
            n1_t[r] = t.view(-1, 1).expand_as(nbr).reshape(-1)
        N1 = torch.cat([n1_nodes[r] for r in RELATIONS])
        T1 = torch.cat([n1_t[r] for r in RELATIONS])

        s2 = self._sample_level(g, N1, T1, self.fanout[1], gen)

        # layer 1 on level-1 nodes
        h0_N1 = self.node_input(g, N1, T1)
        h0_N2 = {r: self.node_input(g, s2[r][0].reshape(-1), T1.view(-1, 1).expand_as(s2[r][0]).reshape(-1))
                 .view(*s2[r][0].shape, -1) for r in RELATIONS}
        h1_N1 = self.layer1(h0_N1, self._neigh(g, s2, T1, h0_N2), self.delta)

        # layer 1 on roots
        h0_R = self.node_input(g, roots, t)
        h0_N1_by_rel, h1_N1_by_rel = {}, {}
        off = 0
        for r in RELATIONS:
            n = n1_nodes[r].numel()
            h0_N1_by_rel[r] = h0_N1[off:off + n].view(*n1_shape[r], -1)
            h1_N1_by_rel[r] = h1_N1[off:off + n].view(*n1_shape[r], -1)
            off += n
        h1_R = self.layer1(h0_R, self._neigh(g, s1, t, h0_N1_by_rel), self.delta)

        # layer 2 on roots
        h2_R = self.layer2(h1_R, self._neigh(g, s1, t, h1_N1_by_rel), self.delta)
        if return_samples:
            return h2_R, (s1, s2, T1)
        return h2_R

    def forward(self, g, eids, gen=None, return_samples=False):
        src, dst, t = g.src[eids], g.dst[eids], g.trel[eids]
        B = eids.numel()
        roots = torch.cat([src, dst])
        tt = torch.cat([t, t])
        res = self.encode(g, roots, tt, gen=gen, return_samples=return_samples)
        h = res[0] if return_samples else res
        hs, hd = h[:B], h[B:]
        z = torch.cat([hs, hd, hs * hd, self.edge_enc(g, eids), g.pair_feat[eids]], dim=-1)
        logits = self.classifier(z).view(-1)
        if return_samples:
            return logits, res[1], tt
        return logits


# ===========================================================
# Causality check (run on the first batch of every run + in selftest)
# ===========================================================

@torch.no_grad()
def assert_causal(model, g, eids):
    _, (s1, s2, T1), tt = model(g, eids, return_samples=True)
    n_checked = 0
    for r in RELATIONS:
        nbr, eid, mask, _ = s1[r]
        if mask.any():
            bad = (g.trel[eid] >= tt.view(-1, 1)) & mask
            assert not bad.any(), f"level-1 {r}: neighbour edge not strictly earlier than root edge"
            n_checked += int(mask.sum())
        nbr, eid, mask, _ = s2[r]
        if mask.any():
            bad = (g.trel[eid] >= T1.view(-1, 1)) & mask
            assert not bad.any(), f"level-2 {r}: neighbour edge not strictly earlier than root edge"
            n_checked += int(mask.sum())
    return n_checked


# ===========================================================
# Train / eval loops
# ===========================================================

def run_epoch(model, g, optimizer, loss_fn, train_idx, batch_size, clip):
    model.train()
    perm = train_idx[torch.randperm(train_idx.numel(), device=train_idx.device)]
    total, steps = 0.0, 0
    for i in range(0, perm.numel(), batch_size):
        b = perm[i:i + batch_size]
        optimizer.zero_grad(set_to_none=True)
        logits = model(g, b)
        loss = loss_fn(logits, g.y[b].float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total += loss.item()
        steps += 1
    return total / max(steps, 1)


@torch.no_grad()
def evaluate(model, g, loss_fn, split_idx, batch_size, eval_cfg, eval_seed,
             override_threshold=None):
    from scripts.utils.evaluation_utils import evaluate_binary_classifier
    model.eval()
    gen = torch.Generator(device=split_idx.device)
    gen.manual_seed(int(eval_seed))          # deterministic sampling at eval
    probs, total, steps = [], 0.0, 0
    for i in range(0, split_idx.numel(), batch_size):
        b = split_idx[i:i + batch_size]
        logits = model(g, b, gen=gen)
        total += loss_fn(logits, g.y[b].float()).item()
        steps += 1
        probs.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs)
    labels = g.y[split_idx].cpu().numpy()
    if override_threshold is not None:
        kw = dict(threshold=override_threshold, auto_threshold=False)
    else:
        kw = dict(threshold=eval_cfg.get("threshold", 0.5),
                  auto_threshold=eval_cfg.get("auto_threshold", True))
    metrics = evaluate_binary_classifier(
        labels, probs,
        compute_top_k=eval_cfg.get("compute_top_k", True),
        k_values=eval_cfg.get("top_k_values", [100, 500, 1000]),
        verbose=False, **kw)
    return metrics, probs, total / max(steps, 1)


# ===========================================================
# Main
# ===========================================================

def main(temporal, default_config):
    from torch.utils.tensorboard import SummaryWriter
    from scripts.utils.config_utils import setup_experiment, save_experiment_config
    from scripts.utils.evaluation_utils import print_metrics
    from scripts.utils.checkpoint_utils import (
        save_checkpoint, load_checkpoint, sync_experiment_to_drive, should_sync_this_epoch)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--intensity", default=None)
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="override epochs (smoke tests)")
    args = parser.parse_args()

    setup = setup_experiment(args.config, args.dataset, intensity=args.intensity,
                             base_config_path=args.base_config, verbose=True,
                             enable_logging=True)
    base_cfg, model_cfg = setup["base_cfg"], setup["model_cfg"]
    dataset_cfg = setup.get("dataset_cfg", {})
    eval_cfg = base_cfg["evaluation"]
    train_cfg = model_cfg["training"]
    paths, device = setup["paths"], setup["device"]
    experiment_name, logger = setup["experiment_name"], setup.get("logger")
    seed = int(base_cfg.get("experiment", {}).get("seed", 42))

    name = "GRAPHSAGE-T (causal v2)" if temporal else "GRAPHSAGE (causal v2)"
    batch_size = int(train_cfg.get("batch_size", 4096))
    eval_batch_size = int(train_cfg.get("eval_batch_size", 8192))
    print(f"\n======= {name} =======")
    print(f"Batch size: {batch_size} | Eval batch size: {eval_batch_size} | Device: {device}")

    writer = SummaryWriter(os.path.join(paths["logs_dir"], "tb"))

    t_load = time.perf_counter()
    g = CausalGraph(paths["graph_folder"], paths["split_folder"], device)
    print(f"[data] prepared in {time.perf_counter() - t_load:.1f}s")
    train_idx = g.train_idx.to(device)
    val_idx = g.val_idx.to(device)
    test_idx = g.test_idx.to(device)
    print(f"Train: {train_idx.numel():,}, Val: {val_idx.numel():,}, Test: {test_idx.numel():,}")

    model = CausalSAGEEdgeModel(g, model_cfg["model"], temporal).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Fanout: {model.fanout} | sampling: {'most-recent' if temporal else 'uniform'} "
          f"| time encoding: {model.t_dim if temporal else 'none'}")

    n_checked = assert_causal(model, g, train_idx[:min(2048, train_idx.numel())])
    n_checked += assert_causal(model, g, test_idx[:min(2048, test_idx.numel())])
    print(f"[causality] OK -- {n_checked:,} sampled neighbour edges all strictly earlier than their root edge")

    pw = (g.y[train_idx] == 0).sum().item() / max((g.y[train_idx] == 1).sum().item(), 1)
    pw = torch.tensor(min(pw, 100.0), dtype=torch.float32, device=device)
    print(f"pos_weight: {pw.item():.2f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    opt_cfg = train_cfg.get("optimizer", {})
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("lr", 5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        betas=tuple(float(b) for b in opt_cfg.get("betas", [0.9, 0.999])),
        eps=float(opt_cfg.get("eps", 1e-8)),
    )
    clip = float(train_cfg.get("gradient_clip", 1.0))

    results_dir = paths["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    best_model_path = os.path.join(results_dir, "best_model.pt")
    checkpoint_path = os.path.join(results_dir, "checkpoint.pt")
    project_root = paths["root"]

    best_val, best_epoch, patience = -1e9, -1, 0
    max_patience = int(train_cfg.get("early_stopping_patience", 25))
    epochs = int(train_cfg.get("epochs", 350))
    if args.max_epochs is not None:
        epochs = args.max_epochs

    start_epoch = 1
    st = load_checkpoint(checkpoint_path, model, optimizer, device)
    if st is not None:
        start_epoch = st["epoch"] + 1
        best_val, best_epoch, patience = st["best_val"], st["best_epoch"], st["patience"]
        if patience >= max_patience:
            print("[RESUME] early stopping had already triggered; skipping training loop.")
            start_epoch = epochs + 1

    eval_seed_val = 1000 + seed
    total_start = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        t_ep = time.perf_counter()
        train_loss = run_epoch(model, g, optimizer, loss_fn, train_idx, batch_size, clip)
        vm, _, val_loss = evaluate(model, g, loss_fn, val_idx, eval_batch_size, eval_cfg, eval_seed_val)
        el = time.perf_counter() - t_ep
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"P={vm.get('precision', 0):.3f} R={vm.get('recall', 0):.3f} F1={vm.get('f1', 0):.3f} "
              f"ROC-AUC={vm.get('roc_auc', 0):.3f} AUPR={vm.get('aupr', 0):.3f} time={el:.1f}s")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        for k in ("precision", "recall", "f1", "roc_auc", "aupr"):
            writer.add_scalar(f"Val/{k}", vm.get(k, 0.0), epoch)
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

    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    train_m, train_p, _ = evaluate(model, g, loss_fn, train_idx, eval_batch_size, eval_cfg, 2000 + seed)
    val_m, val_p, _ = evaluate(model, g, loss_fn, val_idx, eval_batch_size, eval_cfg, eval_seed_val)
    test_m, test_p, _ = evaluate(model, g, loss_fn, test_idx, eval_batch_size, eval_cfg, 3000 + seed,
                                 override_threshold=val_m["threshold"])
    print_metrics(train_m, experiment_name + " TRAIN")
    print_metrics(val_m, experiment_name + " VAL")
    print_metrics(test_m, experiment_name + " TEST")

    out = {
        "train": train_m, "val": val_m, "test": test_m,
        "best_epoch": best_epoch, "best_val_aupr": float(best_val),
        "total_training_time_sec": float(total_time),
        "batch_size": batch_size, "eval_batch_size": eval_batch_size,
        "variant": "causal_v2", "temporal": temporal,
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    torch.save(torch.tensor(train_p), os.path.join(results_dir, "train_pred_probs.pt"))
    torch.save(torch.tensor(val_p), os.path.join(results_dir, "val_pred_probs.pt"))
    torch.save(torch.tensor(test_p), os.path.join(results_dir, "test_pred_probs.pt"))
    save_experiment_config(
        save_dir=results_dir, base_cfg=base_cfg, model_cfg=model_cfg, dataset_cfg=dataset_cfg,
        intensity=args.intensity,
        additional_info={"total_training_time_sec": float(total_time),
                         "batch_size": batch_size, "eval_batch_size": eval_batch_size,
                         "dropped_edge_cols": list(DROP_EDGE_COLS),
                         "categorical_edge_cols": g.cat_cols,
                         "continuous_edge_cols": g.cont_cols},
        filename="experiment_config.json")
    writer.close()
    sync_experiment_to_drive(base_cfg, project_root, results_dir, paths["logs_dir"],
                             label=f"{experiment_name} (final)")
    print(f"\n{name} complete. Results: {results_dir}")
    if logger:
        logger.close()
