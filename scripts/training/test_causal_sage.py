# test_causal_sage.py
# -----------------------------------------------------------
# Self-test for the causal_leakfix_v2 GraphSAGE / GraphSAGE-T data pipeline.
# Brute-force reference checks (run on CPU, a few seconds-minutes):
#   1. point-in-time counts per relation == brute force
#   2. every sampled neighbour edge (both hops) is strictly earlier than the
#      edge being scored, belongs to the right node/relation, and recent
#      sampling returns exactly the K latest earlier edges
#   3. pair-history features == brute force
#   4. no label tensor is reachable from the encoder inputs (labels are only
#      read by the loss)
#   5. forward/backward runs for both models
#
# Usage:
#   python scripts/training/test_causal_sage.py --graph_dir graphs/HI-Small_Trans \
#       --split_dir splits/HI-Small_Trans [--n_checks 300]
# -----------------------------------------------------------
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

import numpy as np
import torch

from scripts.training.causal_sage import (
    CausalGraph, CausalSAGEEdgeModel, RELATIONS, assert_causal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_dir", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--n_checks", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cpu")
    g = CausalGraph(args.graph_dir, args.split_dir, dev)

    src = g.src.numpy(); dst = g.dst.numpy(); t = g.trel.numpy()
    selfm = src == dst
    E = g.num_edges
    picks = rng.choice(E, size=min(args.n_checks, E), replace=False)

    # ---- 1. counts ----
    for e in picks:
        for node in (src[e], dst[e]):
            ref = {
                "out": int(((src == node) & ~selfm & (t < t[e])).sum()),
                "in": int(((dst == node) & ~selfm & (t < t[e])).sum()),
                "self": int(((src == node) & selfm & (t < t[e])).sum()),
            }
            for r in RELATIONS:
                got = int(g.prior_count(r, torch.tensor([node]), torch.tensor([t[e]]))[0])
                assert got == ref[r], f"count mismatch rel={r} node={node} edge={e}: {got} vs {ref[r]}"
    print(f"[1] point-in-time counts match brute force ({len(picks)} edges x 2 nodes x 3 relations)")

    # ---- 2. sampling ----
    K = 7
    nodes = torch.from_numpy(np.concatenate([src[picks], dst[picks]]))
    tt = torch.from_numpy(np.concatenate([t[picks], t[picks]]))
    for r in RELATIONS:
        for recent in (True, False):
            nbr, eid, mask, cnt = g.sample(r, nodes, tt, K, recent=recent)
            for i in range(nodes.numel()):
                v, tv = int(nodes[i]), int(tt[i])
                es = eid[i][mask[i]].numpy()
                assert (t[es] < tv).all(), "sampled edge not strictly earlier"
                if r == "out":
                    assert (src[es] == v).all() and (~selfm[es]).all() and (dst[es] == nbr[i][mask[i]].numpy()).all()
                elif r == "in":
                    assert (dst[es] == v).all() and (~selfm[es]).all() and (src[es] == nbr[i][mask[i]].numpy()).all()
                else:
                    assert (src[es] == v).all() and selfm[es].all()
                assert int(mask[i].sum()) == min(int(cnt[i]), K)
                if recent and len(es):
                    if r == "out":
                        cand = np.where((src == v) & ~selfm & (t < tv))[0]
                    elif r == "in":
                        cand = np.where((dst == v) & ~selfm & (t < tv))[0]
                    else:
                        cand = np.where((src == v) & selfm & (t < tv))[0]
                    # K latest timestamps (ties broken arbitrarily -> compare times)
                    ref_t = np.sort(t[cand])[-K:]
                    assert np.array_equal(np.sort(t[es]), ref_t), "recent sampling is not the K latest"
    print(f"[2] sampling OK: strictly earlier, correct node/relation, recent = K latest")

    # ---- 3. pair history ----
    pf = g.pair_feat.numpy()
    for e in picks:
        s, d, te = src[e], dst[e], t[e]
        same = np.where((src == s) & (dst == d) & (t < te))[0]
        rev = np.where((src == d) & (dst == s) & (t < te))[0]
        assert abs(pf[e, 0] - np.log1p(len(same))) < 1e-5
        assert abs(pf[e, 1] - np.log1p(len(rev))) < 1e-5
        dt = (te - t[same].max()) / 3600.0 if len(same) else 0.0
        assert abs(pf[e, 2] - np.log1p(dt)) < 1e-4
        assert pf[e, 3] == float(len(same) > 0)
    print(f"[3] pair-history features match brute force ({len(picks)} edges)")

    # ---- 4. labels never feed the encoder ----
    y_backup = g.y.clone()
    cfg = {"hidden_dim": 32, "time_dim": 8,
           "fanout": [{"in": 4, "out": 4, "self": 1}, {"in": 3, "out": 3, "self": 1}]}
    eids = torch.from_numpy(picks[:256])
    for temporal in (False, True):
        torch.manual_seed(1)
        m = CausalSAGEEdgeModel(g, cfg, temporal).eval()
        gen = torch.Generator().manual_seed(5)
        a = m(g, eids, gen=gen)
        g.y = torch.randint(0, 2, g.y.shape)
        gen = torch.Generator().manual_seed(5)
        b = m(g, eids, gen=gen)
        g.y = y_backup
        assert torch.allclose(a, b), "logits changed when labels were permuted -> label leak"
    print("[4] logits identical under label permutation (no label path into the model)")

    # ---- 5. forward/backward + causal assertion for both models ----
    for temporal in (False, True):
        m = CausalSAGEEdgeModel(g, cfg, temporal).train()
        n = assert_causal(m, g, eids)
        logits = m(g, eids)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, g.y[eids].float())
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        print(f"[5] {'GraphSAGE-T' if temporal else 'GraphSAGE  '}: forward/backward OK, "
              f"{n:,} sampled neighbour edges causal, loss={loss.item():.4f}")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
