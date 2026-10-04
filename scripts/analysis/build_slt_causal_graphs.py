"""
build_slt_causal_graphs.py
---------------------------------------------------------------------------
Builds a family of graph conditions that test the ACTUAL Social Learning
Theory (SLT) mechanism -- "an account is influenced by deviant associates" --
with label-aware, strictly causal peer-exposure features, plus the controls
needed so that any gain can be attributed to peer deviance and nothing else.

WHY (follow-up to the Oct 2026 ablations): the existing slt_natural features
come from an UNSUPERVISED peer-risk score (slt_injector.py, "no labels used").
They measure "your counterparties look unusual", not "your counterparties are
known launderers", and a GraphSAGE encoder already aggregates neighbour
information, so they showed little incremental value over structure. This
script reformulates the signal the way the theory states it.

ALL conditions are built on the plain BASELINE graph (12 edge features) --
no motif_*, no unsupervised SLT_*, no RAT_* columns -- so the delta of each
condition vs. baseline is attributable ONLY to the feature group added.

FEATURE GROUPS (disjoint; each is a separate condition's only addition)
  SELF  (2)  selfhist_{src,dst}_known: has this account ITSELF been flagged
             as laundering (train labels, strictly before t, after a
             reporting lag)? This is account reputation / recidivism, NOT
             SLT. It exists purely as a control so peer exposure can be
             separated from "this account is already known bad".
  PEER1 (6)  slt_c_{src,dst}_{peer_frac, peer_log_cnt, peer_amt_frac}:
             direct (1-hop) peer exposure -- share / count / log-amount-
             weighted share of the account's PRIOR interactions whose
             counterparty is a known launderer as of t.
  MULTI (4)  slt_c_{src,dst}_{peer_frac_decay, 2hop_mean_decay}:
             recency-decayed 1-hop exposure, and 2-hop exposure (the decayed
             average of the account's partners' OWN 1-hop exposure, partner
             exposure computed excluding the account itself).

CONDITIONS (graph folder name -> columns added to baseline)
  HI-Small_Trans_SLT_causal_1hop            PEER1
  HI-Small_Trans_SLT_causal_multihop        PEER1 + MULTI          (increment over 1hop = multi-hop/decay)
  HI-Small_Trans_selfhist_only              SELF                   (control: reputation, not SLT)
  HI-Small_Trans_selfhist_plus_SLT_causal   SELF + PEER1 + MULTI   (SLT beyond reputation = this vs selfhist_only)
  HI-Small_Trans_SLT_causal_placebo         PEER1 + MULTI computed from train labels PERMUTED across train edges
                                            (same code/shape/structure, no true peer-deviance information)

CAUSALITY GUARANTEES (all enforced by construction and checked by --selftest)
  * Only TRAIN-split labels are ever used. Val/test labels never enter any
    feature, for any row.
  * A feature at transaction time t uses only transactions with timestamp
    STRICTLY < t and flags with (flag time + lag) STRICTLY < t. The row's own
    label can never influence its own features, nor any feature at an earlier
    or equal time.
  * The current transaction's counterparty is EXCLUDED from peer exposure
    (otherwise peer features would silently re-encode "the other party is a
    known launderer", i.e. the SELF group of the counterparty).
  * Self-loop transactions never create peer records (an account is not its
    own peer).
  * Chronological split: train is the earliest period, so flags learned from
    train are all in the past of every val/test row. Val/test rows therefore
    see the (frozen) set of train-period launderers -- conservative; no
    val/test label feedback.

Usage:
    python scripts/analysis/build_slt_causal_graphs.py            # build all 5
    python scripts/analysis/build_slt_causal_graphs.py --selftest # numpy-only unit tests
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STATIC_COPY_FILES = [
    "edge_index.pt", "x.pt", "timestamps.pt", "y_edge.pt", "y_node.pt",
    "node_mapping.json",
]

SELF_COLS = ["selfhist_src_known", "selfhist_dst_known"]
PEER1_COLS = [f"slt_c_{s}_{f}" for s in ("src", "dst")
              for f in ("peer_frac", "peer_log_cnt", "peer_amt_frac")]
MULTI_COLS = [f"slt_c_{s}_{f}" for s in ("src", "dst")
              for f in ("peer_frac_decay", "2hop_mean_decay")]

# condition folder name -> (feature columns to append, use placebo labels?)
CONDITIONS = {
    "HI-Small_Trans_SLT_causal_1hop":          (PEER1_COLS, False),
    "HI-Small_Trans_SLT_causal_multihop":      (PEER1_COLS + MULTI_COLS, False),
    "HI-Small_Trans_selfhist_only":            (SELF_COLS, False),
    "HI-Small_Trans_selfhist_plus_SLT_causal": (SELF_COLS + PEER1_COLS + MULTI_COLS, False),
    "HI-Small_Trans_SLT_causal_placebo":       (PEER1_COLS + MULTI_COLS, True),
}

DAY = 86400.0


# ===========================================================================
# Core: grouped, strictly-earlier prefix sums
# ===========================================================================
class GroupedCum:
    """Sum of values over records of a group whose time key s is STRICTLY
    less than a query time. Sorted-cumsum + searchsorted, fully vectorised.

    key = group * 2^32 + s  (group >= 0, 0 <= s < 2^32, int64). A query for
    (group g, time t) is the count of records with key in [g<<32, (g<<32)+t).
    A query group of -1 (or any non-existent group) returns 0.
    """

    def __init__(self, group, s, vals):
        group = np.asarray(group, dtype=np.int64)
        s = np.asarray(s, dtype=np.int64)
        if s.size:
            assert s.min() >= 0 and s.max() < (1 << 32), "time key out of range"
            assert group.min() >= 0, "negative group id"
        key = (group << 32) + s
        order = np.argsort(key, kind="stable")
        self.key = key[order]
        self.cums = []
        for v in vals:
            c = np.zeros(len(order) + 1, dtype=np.float64)
            np.cumsum(np.asarray(v, dtype=np.float64)[order], out=c[1:])
            self.cums.append(c)

    def query(self, q_group, q_t):
        q_group = np.asarray(q_group, dtype=np.int64)
        q_t = np.asarray(q_t, dtype=np.int64)
        hi = np.searchsorted(self.key, (q_group << 32) + q_t, side="left")
        lo = np.searchsorted(self.key, (q_group << 32), side="left")
        return [c[hi] - c[lo] for c in self.cums]


def _safe_ratio(num, den, valid):
    out = np.zeros_like(num, dtype=np.float64)
    ok = valid & (den > 0)
    out[ok] = num[ok] / den[ok]
    return np.clip(out, 0.0, 1.0)


def compute_features(src, dst, t_sec, w_amt, y, train_mask, n_nodes,
                     lag_sec=86400, halflife_days=3.0):
    """Return dict feature_name -> float32 array (E,).

    src, dst   : int node ids per edge (transaction)
    t_sec      : int64 timestamps in seconds (any origin; shifted to 0)
    w_amt      : per-edge weight for amount-weighted exposure (log1p amount)
    y          : 0/1 laundering label per edge (ONLY train rows are used)
    train_mask : bool per edge, True for train-split edges
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    t_sec = np.asarray(t_sec, dtype=np.int64)
    t_sec = t_sec - t_sec.min()
    w_amt = np.asarray(w_amt, dtype=np.float64)
    y = np.asarray(y).astype(np.int64)
    train_mask = np.asarray(train_mask, dtype=bool)
    E = len(src)

    tau_days = float(halflife_days) / np.log(2.0)
    assert (t_sec.max() / DAY) / tau_days < 40.0, (
        "halflife too short for the dataset span (exp overflow / precision)")

    # ---- when does each account become a KNOWN launderer? ----------------
    # Only TRAIN-labelled laundering edges; known strictly after first flag
    # time + lag. (An edge's own label can only make an account known at
    # t_edge + lag >= t_edge, i.e. never visible to that edge itself.)
    flagged = train_mask & (y == 1)
    BIG = np.iinfo(np.int64).max
    first = np.full(n_nodes, BIG, dtype=np.int64)
    np.minimum.at(first, src[flagged], t_sec[flagged])
    np.minimum.at(first, dst[flagged], t_sec[flagged])
    has_k = first != BIG
    k = np.where(has_k, first + int(lag_sec), 0)

    feats = {}

    # ---- SELF group (reputation control; NOT SLT) -------------------------
    feats["selfhist_src_known"] = (has_k[src] & (k[src] < t_sec)).astype(np.float32)
    feats["selfhist_dst_known"] = (has_k[dst] & (k[dst] < t_sec)).astype(np.float32)

    # ---- interaction records (no self-loops) ------------------------------
    eid = np.nonzero(src != dst)[0]
    n_ne = len(eid)
    u, v = src[eid], dst[eid]
    R_exp = np.concatenate([u, v])               # exposed account
    R_par = np.concatenate([v, u])               # its partner in that interaction
    R_eid = np.concatenate([eid, eid])
    R_t = t_sec[R_eid]
    R_w = w_amt[R_eid]
    R_dec = np.exp(R_t / DAY / tau_days)          # exp(t''/tau); decay weight = this * exp(-t/tau)
    R_one = np.ones(2 * n_ne, dtype=np.float64)

    # directed (exposed, partner) pair id -- for excluding the CURRENT counterparty
    _, R_pair = np.unique(R_exp * np.int64(n_nodes) + R_par, return_inverse=True)
    R_pair = R_pair.astype(np.int64)

    # records whose partner is (or later becomes) known: counts from
    # s = max(interaction time, partner known time); i.e. only once BOTH the
    # interaction has happened AND the partner is known.
    dm = has_k[R_par]
    R_s_dev = np.maximum(R_t, k[R_par])

    node_tot = GroupedCum(R_exp, R_t, [R_one, R_w, R_dec])
    pair_tot = GroupedCum(R_pair, R_t, [R_one, R_w, R_dec])
    node_dev = GroupedCum(R_exp[dm], R_s_dev[dm], [R_one[dm], R_w[dm], R_dec[dm]])
    pair_dev = GroupedCum(R_pair[dm], R_s_dev[dm], [R_one[dm], R_w[dm], R_dec[dm]])

    # query pair ids for the current edge (self-loops -> -1 -> no exclusion)
    q_pair_src = np.full(E, -1, dtype=np.int64)
    q_pair_dst = np.full(E, -1, dtype=np.int64)
    q_pair_src[eid] = R_pair[:n_ne]
    q_pair_dst[eid] = R_pair[n_ne:]

    side_nodes = {"src": (src, q_pair_src), "dst": (dst, q_pair_dst)}
    stage1 = {}
    den_cache = {}
    for side, (a_node, q_pair) in side_nodes.items():
        tot_c, tot_w, tot_d = node_tot.query(a_node, t_sec)
        pt_c, pt_w, pt_d = pair_tot.query(q_pair, t_sec)
        dv_c, dv_w, dv_d = node_dev.query(a_node, t_sec)
        pdv_c, pdv_w, pdv_d = pair_dev.query(q_pair, t_sec)

        den_c = np.round(tot_c - pt_c)          # exact integers in float64
        den_w = np.maximum(tot_w - pt_w, 0.0)
        den_d = np.maximum(tot_d - pt_d, 0.0)
        num_c = np.maximum(np.round(dv_c - pdv_c), 0.0)
        num_w = np.maximum(dv_w - pdv_w, 0.0)
        num_d = np.maximum(dv_d - pdv_d, 0.0)
        has_den = den_c > 0

        frac = _safe_ratio(num_c, den_c, has_den)
        feats[f"slt_c_{side}_peer_frac"] = frac.astype(np.float32)
        feats[f"slt_c_{side}_peer_log_cnt"] = np.log1p(num_c).astype(np.float32)
        feats[f"slt_c_{side}_peer_amt_frac"] = _safe_ratio(num_w, den_w, has_den).astype(np.float32)
        feats[f"slt_c_{side}_peer_frac_decay"] = _safe_ratio(num_d, den_d, has_den).astype(np.float32)
        stage1[side] = frac
        den_cache[side] = (den_d, has_den)

    # ---- 2-hop: decayed mean of PARTNERS' own 1-hop exposure ---------------
    # For record (exposed a, partner p) at interaction e', value = p's
    # 1-hop peer_frac at e' (computed excluding a, by the pair exclusion
    # above) -- a strictly-earlier quantity, so no future information.
    v2 = np.concatenate([stage1["dst"][eid], stage1["src"][eid]])
    node2 = GroupedCum(R_exp, R_t, [v2 * R_dec])
    pair2 = GroupedCum(R_pair, R_t, [v2 * R_dec])
    for side, (a_node, q_pair) in side_nodes.items():
        num2 = np.maximum(node2.query(a_node, t_sec)[0] - pair2.query(q_pair, t_sec)[0], 0.0)
        den_d, has_den = den_cache[side]
        feats[f"slt_c_{side}_2hop_mean_decay"] = _safe_ratio(num2, den_d, has_den).astype(np.float32)

    return feats


# ===========================================================================
# Self-test (numpy only): brute-force reference + causality properties
# ===========================================================================
def _reference_edge_features(i, src, dst, t, w, y, train_mask, lag, halflife_days, n_nodes):
    """Slow, obviously-correct loop implementation for ONE edge."""
    tau = halflife_days / np.log(2.0)
    first = {}
    for e in range(len(src)):
        if train_mask[e] and y[e] == 1:
            for a in (src[e], dst[e]):
                first[a] = min(first.get(a, 10**18), t[e])
    known = lambda a: (a in first) and (first[a] + lag < t[i])

    out = {"selfhist_src_known": float(known(src[i])),
           "selfhist_dst_known": float(known(dst[i]))}

    def interactions(a, exclude_partner):
        res = []
        for e in range(len(src)):
            if src[e] == dst[e] or t[e] >= t[i]:
                continue
            if a == src[e]:
                p = dst[e]
            elif a == dst[e]:
                p = src[e]
            else:
                continue
            if p == exclude_partner:
                continue
            res.append((e, p))
        return res

    peer_frac_at = {}

    def peer_frac(a, e_idx, exclude_partner, t_q):
        # peer_frac of account a at time t_q (= t[e_idx]) excluding partner
        num = den = 0
        for e in range(len(src)):
            if src[e] == dst[e] or t[e] >= t_q:
                continue
            if a == src[e]:
                p = dst[e]
            elif a == dst[e]:
                p = src[e]
            else:
                continue
            if p == exclude_partner:
                continue
            den += 1
            if (p in first) and (first[p] + lag < t_q):
                num += 1
        return num / den if den else 0.0

    for side, a, c in (("src", src[i], dst[i]), ("dst", dst[i], src[i])):
        ints = interactions(a, c)
        den = len(ints)
        num = sum(1 for (e, p) in ints if known(p))
        num_w = sum(w[e] for (e, p) in ints if known(p))
        den_w = sum(w[e] for (e, p) in ints)
        dec = lambda e: np.exp(-(t[i] - t[e]) / DAY / tau)
        den_d = sum(dec(e) for (e, p) in ints)
        num_d = sum(dec(e) for (e, p) in ints if known(p))
        two = 0.0
        for (e, p) in ints:
            two += dec(e) * peer_frac(p, e, a, t[e])
        out[f"slt_c_{side}_peer_frac"] = num / den if den else 0.0
        out[f"slt_c_{side}_peer_log_cnt"] = float(np.log1p(num))
        out[f"slt_c_{side}_peer_amt_frac"] = (num_w / den_w) if den and den_w > 0 else 0.0
        out[f"slt_c_{side}_peer_frac_decay"] = (num_d / den_d) if den and den_d > 0 else 0.0
        out[f"slt_c_{side}_2hop_mean_decay"] = (two / den_d) if den and den_d > 0 else 0.0
    return out


def run_selftest():
    rng = np.random.default_rng(0)
    N, E = 60, 1500
    span = 15 * 86400
    src = rng.integers(0, N, E)
    dst = rng.integers(0, N, E)
    # a few self-loops and repeated pairs on purpose
    dst[:40] = src[:40]
    dst[40:200] = (src[40:200] + 1) % N
    # timestamps with ties (minute resolution)
    t = (rng.integers(0, span // 60, E) * 60).astype(np.int64)
    w = rng.uniform(1, 10, E)
    # laundering edges concentrate on some accounts so exposure is non-trivial
    bad = rng.choice(N, 10, replace=False)
    y = (np.isin(src, bad) | np.isin(dst, bad)) & (rng.random(E) < 0.5)
    y = y.astype(np.int64)
    cut = np.quantile(t, 0.6)
    train_mask = t <= cut
    lag = 3600
    H = 3.0
    kw = dict(lag_sec=lag, halflife_days=H)

    base = compute_features(src, dst, t, w, y, train_mask, N, **kw)
    names = list(base.keys())
    assert set(names) == set(SELF_COLS + PEER1_COLS + MULTI_COLS), names
    n_nonzero = {n: int((base[n] > 0).sum()) for n in names}
    assert all(c > 0 for c in n_nonzero.values()), f"degenerate feature(s): {n_nonzero}"

    t0 = t - t.min()
    # (1) brute-force reference equality on a random sample of edges
    sample = rng.choice(E, 60, replace=False)
    for i in sample:
        ref = _reference_edge_features(i, src, dst, t0, w, y, train_mask, lag, H, N)
        for n in names:
            a, b = float(base[n][i]), float(ref[n])
            assert abs(a - b) < 1e-4, f"edge {i} feature {n}: vectorised={a} reference={b}"
    print(f"[OK] (1) vectorised == brute-force reference on {len(sample)} edges x {len(names)} features")

    # (2) val/test labels never matter
    y2 = y.copy()
    y2[~train_mask] = 1 - y2[~train_mask]
    f2 = compute_features(src, dst, t, w, y2, train_mask, N, **kw)
    for n in names:
        assert np.array_equal(base[n], f2[n]), f"val/test label leaked into {n}"
    print("[OK] (2) flipping every val/test label changes no feature")

    # (3) no future / same-time / own-label influence: flip train labels at
    #     times >= T - lag, then every edge with t <= T must be unchanged.
    for q in (0.2, 0.4, 0.55):
        T = np.quantile(t, q)
        y3 = y.copy()
        m = train_mask & (t >= T - lag)
        y3[m] = 1 - y3[m]
        f3 = compute_features(src, dst, t, w, y3, train_mask, N, **kw)
        rows = t <= T
        assert rows.sum() > 50
        for n in names:
            assert np.array_equal(base[n][rows], f3[n][rows]), \
                f"{n}: features at t<=T changed when labels at t>=T-lag flipped (q={q})"
    print("[OK] (3) flipping labels at/after T-lag leaves all features at t<=T unchanged")

    # (4) an edge's own label never affects its own features
    for i in rng.choice(np.nonzero(train_mask)[0], 15, replace=False):
        y4 = y.copy()
        y4[i] = 1 - y4[i]
        f4 = compute_features(src, dst, t, w, y4, train_mask, N, **kw)
        for n in names:
            assert base[n][i] == f4[n][i], f"edge {i}'s own label changed its {n}"
    print("[OK] (4) an edge's own label never changes its own features")

    # (5) bounds
    for n in names:
        assert np.isfinite(base[n]).all()
        if n.endswith("log_cnt"):
            assert base[n].min() >= 0
        else:
            assert base[n].min() >= 0 and base[n].max() <= 1.0 + 1e-6, n
    print("[OK] (5) all features finite and in range")

    # (6) placebo (permuted train labels) really destroys the association
    yp = y.copy()
    tr = np.nonzero(train_mask)[0]
    yp[tr] = rng.permutation(y[tr])
    fp = compute_features(src, dst, t, w, yp, train_mask, N, **kw)
    assert any(not np.array_equal(base[n], fp[n]) for n in PEER1_COLS)
    print("[OK] (6) placebo features differ from true features (permutation active)")
    print("\nALL SELF-TESTS PASSED")


# ===========================================================================
# Graph building (needs torch; cluster)
# ===========================================================================
def build_all(args):
    import torch

    root = PROJECT_ROOT
    base_dir = Path(args.baseline_graph_dir or root / "graphs" / "HI-Small_Trans")
    base_split = Path(args.baseline_split_dir or root / "splits" / "HI-Small_Trans")
    out_root = Path(args.out_root or root / "graphs")
    split_root = Path(args.splits_root or root / "splits")
    for d in (base_dir, base_split):
        if not d.is_dir():
            raise FileNotFoundError(d)

    with open(base_dir / "edge_attr_cols.json") as f:
        base_cols = json.load(f)
    if "log_amt_paid" not in base_cols:
        raise RuntimeError("baseline graph lacks log_amt_paid")
    meta_path = base_split / "split_metadata.json"
    if meta_path.exists():
        sm = json.load(open(meta_path))
        if sm.get("split_method") != "chronological":
            raise RuntimeError(
                f"{meta_path}: split_method={sm.get('split_method')!r}. These "
                f"features assume a CHRONOLOGICAL split (train strictly earliest). "
                f"Rebuild the baseline split with --split_mode chronological first."
            )

    edge_index = torch.load(base_dir / "edge_index.pt").numpy()
    edge_attr = torch.load(base_dir / "edge_attr.pt")
    t_sec = torch.load(base_dir / "timestamps.pt").numpy().astype(np.int64)
    y = torch.load(base_dir / "y_edge.pt").numpy().astype(np.int64)
    train_idx = torch.load(base_split / "train_edge_idx.pt").numpy()
    n_nodes = int(torch.load(base_dir / "x.pt").shape[0])
    E = edge_index.shape[1]
    src, dst = edge_index[0].astype(np.int64), edge_index[1].astype(np.int64)
    w_amt = edge_attr[:, base_cols.index("log_amt_paid")].numpy().astype(np.float64)

    train_mask = np.zeros(E, dtype=bool)
    train_mask[train_idx] = True
    val_idx = torch.load(base_split / "val_edge_idx.pt").numpy()
    test_idx = torch.load(base_split / "test_edge_idx.pt").numpy()
    # chronological sanity: every train timestamp <= every val/test timestamp
    assert t_sec[train_idx].max() <= t_sec[val_idx].min(), "train overlaps val in time"
    assert t_sec[train_idx].max() <= t_sec[test_idx].min(), "train overlaps test in time"
    print(f"E={E:,} nodes={n_nodes:,} train={len(train_idx):,} "
          f"train positives={int(y[train_idx].sum()):,}")

    kw = dict(lag_sec=int(args.lag_hours * 3600), halflife_days=args.halflife_days)
    print("Computing TRUE-label causal features ...")
    true_f = compute_features(src, dst, t_sec, w_amt, y, train_mask, n_nodes, **kw)

    print("Computing PLACEBO features (train labels permuted across train edges) ...")
    rng = np.random.default_rng(args.seed)
    y_plc = y.copy()
    y_plc[train_idx] = rng.permutation(y[train_idx])
    plc_f = compute_features(src, dst, t_sec, w_amt, y_plc, train_mask, n_nodes, **kw)

    for name, (cols, use_placebo) in CONDITIONS.items():
        if args.only and name not in args.only:
            continue
        feats = plc_f if use_placebo else true_f
        new_block = torch.from_numpy(np.stack([feats[c] for c in cols], axis=1).astype(np.float32))
        new_attr = torch.cat([edge_attr, new_block], dim=1)
        new_cols = base_cols + cols

        out_dir = out_root / name
        os.makedirs(out_dir, exist_ok=True)
        torch.save(new_attr, out_dir / "edge_attr.pt")
        with open(out_dir / "edge_attr_cols.json", "w") as f:
            json.dump(new_cols, f, indent=2)
        for fn in STATIC_COPY_FILES:
            s = base_dir / fn
            if s.exists():
                shutil.copy(s, out_dir / fn)
        stats = json.load(open(base_dir / "graph_stats.json"))
        stats["num_edge_features"] = len(new_cols)
        stats["note"] = (f"baseline + {len(cols)} causal SLT-study columns "
                         f"({'PLACEBO: ' if use_placebo else ''}{cols[0]} ... {cols[-1]})")
        with open(out_dir / "graph_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        with open(out_dir / "slt_causal_meta.json", "w") as f:
            json.dump({
                "baseline_graph_dir": str(base_dir),
                "added_columns": cols, "placebo": use_placebo,
                "lag_hours": args.lag_hours, "halflife_days": args.halflife_days,
                "placebo_seed": args.seed,
                "labels_used": "train split only, strictly earlier than each row",
            }, f, indent=2)

        # identical split to the baseline (same rows, same order)
        sp_out = split_root / name
        if sp_out.exists():
            shutil.rmtree(sp_out)
        shutil.copytree(base_split, sp_out)
        print(f"  built {name}: +{len(cols)} cols -> {len(new_cols)} total; split copied")

    print("\nDONE. Next: fix_node_degree_leakage.py on each new graph (idempotent), then train.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--baseline_graph_dir", default=None)
    ap.add_argument("--baseline_split_dir", default=None)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--splits_root", default=None)
    ap.add_argument("--lag_hours", type=float, default=24.0,
                    help="reporting lag: an account becomes 'known' this long AFTER its first train-labelled laundering transaction")
    ap.add_argument("--halflife_days", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=123, help="placebo permutation seed")
    ap.add_argument("--only", nargs="*", default=None, help="subset of condition folder names")
    args = ap.parse_args()
    if args.selftest:
        run_selftest()
        return
    build_all(args)


if __name__ == "__main__":
    main()
