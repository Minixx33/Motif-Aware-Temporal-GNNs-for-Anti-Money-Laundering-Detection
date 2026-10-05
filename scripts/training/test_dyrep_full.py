# test_dyrep_full.py
# -----------------------------------------------------------
# Self-test for the full DyRep (causal_leakfix_v2). Run on CPU on any graph
# slice with chronological splits:
#   python scripts/training/test_dyrep_full.py --graph_dir graphs/HI-Small_Trans \
#       --split_dir splits/HI-Small_Trans [--max_batches 60]
#
#   1. batching: every event exactly once, time-ordered, no timestamp split
#      across batches (=> batch b strictly later than all batches < b)
#   2. strict memory causality: replaying the stream, the latest event folded
#      into memory is strictly earlier than every event being scored
#      (independent brute-force tracking, not the model's own counter)
#   3. attention neighbours strictly earlier than the event
#   4. label permutation leaves every score unchanged (no label path)
#   5. future-event invariance: changing the features of events in LATER
#      batches does not change any score of earlier batches
#   6. gradient reaches the memory updater (RNN) and DyRep attention
#   7. eval pass is deterministic
#   8. train-epoch snapshot == memory just before the first val batch
# -----------------------------------------------------------
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

import numpy as np
import torch
import torch.nn as nn

from scripts.training.causal_sage import CausalGraph
from scripts.training.train_dyrep_full import (
    EventStream, MemoryState, DyRepFull, step_batch, run_eval_stream, train_epoch,
    SPLIT_TRAIN, SPLIT_VAL)

CFG = {"memory_dim": 32, "time_dim": 8, "node_feat_dim": 8, "n_neighbors": 5,
       "attn_heads": 2, "hidden_dim": 32, "dropout": 0.0, "aux_weight": 1.0}


def eval_scores(model, g, stream, n_batches):
    probs = torch.zeros(g.num_edges)
    st = MemoryState(g.num_nodes, model.d, g.device)
    run_eval_stream(model, g, stream, st, 0, n_batches - 1, probs)
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_dir", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=500)
    ap.add_argument("--max_batches", type=int, default=60)
    args = ap.parse_args()
    torch.manual_seed(0)

    g = CausalGraph(args.graph_dir, args.split_dir, torch.device("cpu"))
    stream = EventStream(g, args.batch_size)
    nb = min(args.max_batches, stream.n_batches)
    ts = g.trel.numpy()

    # ---- 1. batching ----
    seen = np.concatenate([stream.batch(b)[0].numpy() for b in range(stream.n_batches)])
    assert len(seen) == g.num_edges and len(np.unique(seen)) == g.num_edges, "events not covered exactly once"
    prev_max = -1
    for b in range(stream.n_batches):
        e = stream.batch(b)[0].numpy()
        assert ts[e].min() > prev_max, f"batch {b} not strictly later than earlier batches"
        prev_max = ts[e].max()
    print(f"[1] {stream.n_batches:,} batches: every event once, batch b strictly later than all batches < b")

    model = DyRepFull(g, CFG).eval()

    # ---- 2 + 3. strict causality during replay (independent tracking) ----
    st = MemoryState(g.num_nodes, model.d, g.device)
    folded_max = -1          # brute force: max ts of events whose messages were applied
    pending_ts = None
    with torch.no_grad():
        for b in range(nb):
            ev, _ = stream.batch(b)
            if pending_ts is not None:
                folded_max = max(folded_max, pending_ts)
            assert folded_max < ts[ev.numpy()].min(), "memory contains a non-earlier event"
            # memory of nodes NOT touched by any applied event must still be zero
            logits, _, commit = step_batch(model, g, st, ev, train=False, check=True)
            commit()
            pending_ts = int(ts[ev.numpy()].max())
            touched = np.unique(np.concatenate([g.src[stream.batch(x)[0]].numpy() for x in range(b)] +
                                               [g.dst[stream.batch(x)[0]].numpy() for x in range(b)])) if b else np.array([], dtype=np.int64)
            untouched = np.setdiff1d(np.arange(g.num_nodes), touched)
            assert torch.all(st.M[torch.from_numpy(untouched)] == 0), "memory changed for a node with no applied event"
    print(f"[2] memory strictly causal over {nb} batches (brute-force tracking; untouched nodes stay zero)")
    print(f"[3] attention neighbours strictly earlier (asserted inside every embed call)")

    # ---- 4. label permutation ----
    a = eval_scores(model, g, stream, nb)
    y0 = g.y.clone()
    g.y = torch.randint(0, 2, g.y.shape)
    b_ = eval_scores(model, g, stream, nb)
    g.y = y0
    assert torch.allclose(a, b_), "scores changed when labels were permuted"
    print("[4] scores identical under label permutation")

    # ---- 5. future-event invariance ----
    cut = nb // 2
    later = torch.cat([stream.batch(x)[0] for x in range(cut, nb)])
    ec0 = g.edge_cont.clone()
    g.edge_cont[later] = torch.randn_like(g.edge_cont[later]) * 5
    c = eval_scores(model, g, stream, nb)
    g.edge_cont = ec0
    earlier = torch.cat([stream.batch(x)[0] for x in range(0, cut)])
    assert torch.allclose(a[earlier], c[earlier]), "a later event changed an earlier score"
    assert not torch.allclose(a[later], c[later]), "sanity: later scores should change"
    print(f"[5] perturbing events in batches >= {cut} leaves all earlier scores unchanged")

    # ---- 6. gradients reach memory updater + attention ----
    model.train()
    st = MemoryState(g.num_nodes, model.d, g.device)
    for b in range(4):
        ev, _ = stream.batch(b)
        logits, aux, commit = step_batch(model, g, st, ev, train=True, aux_weight=1.0)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, g.y[ev].float()) + aux.mean()
        model.zero_grad()
        loss.backward()
        # b=1: the first pending batch is applied (memory all zero, and the first
        # batch has no earlier neighbours), so only the input side of the updater
        # can get gradient. From b=2 on, every module must.
        names = []
        if b >= 1:
            names = ["rnn.weight_ih", "omega.0.weight", "classifier.0.weight"]
        if b >= 2:
            names += ["rnn.weight_hh", "q_lin.weight", "k_lin.weight", "v_lin.weight",
                      "attn.in_proj_weight", "merge.0.weight"]
        for name in names:
            p = dict(model.named_parameters())[name]
            assert p.grad is not None and p.grad.abs().sum() > 0, f"no gradient for {name} (batch {b})"
        commit()
    print("[6] gradients reach RNN memory updater, DyRep attention, intensity and classifier")

    # ---- 7. determinism ----
    model.eval()
    d1 = eval_scores(model, g, stream, nb)
    d2 = eval_scores(model, g, stream, nb)
    assert torch.equal(d1, d2)
    print("[7] eval pass deterministic")

    # ---- 8. snapshot ----
    if stream.first[SPLIT_VAL] <= stream.last[SPLIT_TRAIN] + 1 and stream.n_batches < 400:
        opt = torch.optim.SGD(model.parameters(), lr=0.0)    # lr 0: weights fixed
        lf = nn.BCEWithLogitsLoss()
        model.train()
        _, _, snap = train_epoch(model, g, stream, opt, lf, 0.0, 1.0)
        model.eval()
        ref = MemoryState(g.num_nodes, model.d, g.device)
        with torch.no_grad():
            for bb in range(stream.first[SPLIT_VAL]):
                ev, _ = stream.batch(bb)
                _, _, cm = step_batch(model, g, ref, ev, train=False)
                cm()
        assert torch.allclose(snap.M, ref.M, atol=1e-5) and torch.equal(snap.pending, ref.pending)
        print("[8] train-epoch snapshot == memory just before the first val batch")
    else:
        print("[8] skipped (stream too long for the full-epoch snapshot check; use a smaller slice)")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
