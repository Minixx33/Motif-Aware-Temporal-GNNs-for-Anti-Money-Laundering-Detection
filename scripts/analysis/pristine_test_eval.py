"""
pristine_test_eval.py
----------------------------------------------------------------------
Data-leakage robustness check for RAT/SLT theory injection.

BACKGROUND
Theory injection (rat_injector.py / slt_injector.py) selects a fraction of
laundering-labeled rows (by pristine RAT_score/SLT_score quantile) and
boosts their underlying RAT_*/SLT_*/motif_* feature values BEFORE the
train/val/test split is made. That means some test-set laundering edges
carry artificially amplified theory features, so reported test AUPR/
ROC-AUC partly reflects "how much easier the test set was made" rather
than pure detection performance on realistic, un-doctored data.

WHY THIS CHECK IS CLEAN (no retraining needed)
GraphSAGE-T's architecture makes this check exact without touching the
trained weights: node embeddings h = encoder(x, edge_index) depend ONLY
on structural node features and graph connectivity (see GraphSAGEEncoder
in train_graphsage_t.py -- SAGEConv never sees edge_attr). edge_attr/feat
is consumed ONLY by the final edge classifier MLP:
    classify(h_src, h_dst, feat) = MLP(concat(h_src, h_dst, feat))
So swapping a test edge's feat vector for its un-boosted ("pristine")
equivalent has ZERO effect on any other edge's node embeddings -- it
isolates exactly the marginal effect of that edge's OWN (possibly
boosted) features on the model's decision. h is identical in both runs
below (same x, same edge_index either way).

WHAT THIS SCRIPT DOES
  1. Loads the existing (boosted) graph + the already-trained best_model.pt
     checkpoint for one seed. No retraining.
  2. Loads a "pristine" graph built from a CSV snapshot the injector saves
     BEFORE any intensity boosting is applied (see the --dump_pristine
     flag added to rat_injector.py / slt_injector.py).
  3. Sanity-checks edge_index / y_edge / edge_attr_cols match between the
     two graphs (same rows, same order, same schema). If this fails, the
     pristine graph was built from a misaligned CSV and everything below
     is meaningless -- the script aborts rather than printing bad numbers.
  4. Evaluates the SAME frozen model twice on the test split:
       (a) original boosted feat              -> should reproduce metrics.json
       (b) feat w/ test rows swapped to pristine values -> the actual check
  5. Prints both metric sets side by side and saves them to JSON next to
     the checkpoint.

USAGE (run on the remote desktop, after training as usual)
    python scripts/analysis/pristine_test_eval.py \
        --theory RAT \
        --model_path results/HI-Small_Trans_RAT_medium/seed1_.../graphsage-t/best_model.pt

Repeat with --theory SLT and for each seed you want to check.
----------------------------------------------------------------------
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from scripts.utils.evaluation_utils import evaluate_binary_classifier, print_metrics
from scripts.training.train_graphsage_t import GraphSAGE_T, build_sinusoidal_time_encoding


def load_graph(graph_dir):
    edge_index = torch.load(os.path.join(graph_dir, "edge_index.pt"))
    edge_attr = torch.load(os.path.join(graph_dir, "edge_attr.pt"))
    x = torch.load(os.path.join(graph_dir, "x.pt"))
    y_edge = torch.load(os.path.join(graph_dir, "y_edge.pt"))
    timestamps = torch.load(os.path.join(graph_dir, "timestamps.pt"))
    with open(os.path.join(graph_dir, "edge_attr_cols.json")) as f:
        cols = json.load(f)
    return edge_index, edge_attr, x, y_edge, timestamps, cols


def main():
    ap = argparse.ArgumentParser(description="Pristine-test-set robustness check (no retraining)")
    ap.add_argument("--theory", choices=["RAT", "SLT"], required=True)
    ap.add_argument("--intensity", default="medium",
                     help="Which boosted graph to check against (default: medium, "
                          "the one actually used for the paper's headline results)")
    ap.add_argument("--boosted_graph_dir", default=None,
                     help="Default: graphs/HI-Small_Trans_<theory>_<intensity>")
    ap.add_argument("--pristine_graph_dir", default=None,
                     help="Default: graphs/HI-Small_Trans_<theory>_pristine")
    ap.add_argument("--split_dir", default=None,
                     help="Default: splits/HI-Small_Trans_<theory>_<intensity>")
    ap.add_argument("--model_path", required=True,
                     help="Path to the trained best_model.pt for this theory/seed")

    # Must match the trained model's config exactly (see experiment_config.json
    # for the run you're checking -- model_config.model.*)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--aggregator", default="mean")
    ap.add_argument("--time_dim", type=int, default=32)

    ap.add_argument("--eval_batch_size", type=int, default=16384)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dataset_name = f"HI-Small_Trans_{args.theory}_{args.intensity}"
    boosted_dir = args.boosted_graph_dir or os.path.join(PROJECT_ROOT, "graphs", dataset_name)
    pristine_dir = args.pristine_graph_dir or os.path.join(
        PROJECT_ROOT, "graphs", f"HI-Small_Trans_{args.theory}_pristine"
    )
    split_dir = args.split_dir or os.path.join(PROJECT_ROOT, "splits", dataset_name)

    device = torch.device(args.device)

    print(f"[LOAD] boosted graph:  {boosted_dir}")
    b_edge_index, b_edge_attr, x, y_edge, timestamps, b_cols = load_graph(boosted_dir)
    print(f"[LOAD] pristine graph: {pristine_dir}")
    p_edge_index, p_edge_attr, p_x, p_y_edge, p_timestamps, p_cols = load_graph(pristine_dir)

    # ------------------------------------------------------------------
    # Sanity checks -- this whole result is meaningless if these fail.
    # ------------------------------------------------------------------
    assert b_cols == p_cols, (
        "Column schema mismatch between boosted and pristine graphs.\n"
        f"  boosted:  {b_cols}\n  pristine: {p_cols}\n"
        "Did you build the pristine graph from a CSV produced by the SAME "
        "injector run (--dump_pristine) as the boosted one?"
    )
    assert torch.equal(b_edge_index, p_edge_index), (
        "edge_index differs between boosted and pristine graphs -- row "
        "order / account mapping does not match. Do not trust any numbers "
        "below until this is fixed. Most likely cause: the pristine CSV "
        "was generated by a different injector invocation than the "
        "boosted CSV (e.g. re-run with different weights), so row order "
        "or the account set differs."
    )
    assert torch.equal(y_edge, p_y_edge), "y_edge differs -- row alignment is broken."
    print("[OK] boosted and pristine graphs are row-aligned and schema-matched.\n")

    x = x.to(device)
    edge_index = b_edge_index.to(device)
    y_edge = y_edge.to(device)
    timestamps = timestamps.to(device)
    b_edge_attr = b_edge_attr.to(device)
    p_edge_attr = p_edge_attr.to(device)

    time_enc = build_sinusoidal_time_encoding(timestamps, args.time_dim)
    feat_boosted = torch.cat([b_edge_attr, time_enc], dim=1)

    test_idx = torch.load(os.path.join(split_dir, "test_edge_idx.pt")).to(device)

    # Identical to the boosted feat matrix everywhere EXCEPT the test-split
    # rows, where the theory/edge_attr portion is swapped for its
    # never-boosted value. time_enc is untouched (Timestamp isn't injected).
    feat_pristine_test = feat_boosted.clone()
    feat_pristine_test[test_idx] = torch.cat(
        [p_edge_attr[test_idx], time_enc[test_idx]], dim=1
    )

    n_pos = (y_edge[test_idx] == 1).sum().item()
    print(f"[INFO] test edges: {test_idx.numel():,} ({n_pos:,} positive / laundering)\n")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_cfg = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "aggregator": args.aggregator,
        "time_dim": args.time_dim,
    }
    model = GraphSAGE_T(node_dim=x.size(1), feat_dim=feat_boosted.size(1), cfg=model_cfg).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    @torch.no_grad()
    def eval_on(feat, idx, batch_size):
        h = model.encode(x, edge_index)
        probs = []
        for start in range(0, idx.numel(), batch_size):
            end = min(start + batch_size, idx.numel())
            b = idx[start:end]
            logits = model.classify(h, edge_index[:, b], feat[b])
            probs.append(torch.sigmoid(logits).cpu().numpy())
        probs = np.concatenate(probs)
        labels = y_edge[idx].cpu().numpy()
        return evaluate_binary_classifier(
            labels, probs,
            threshold=0.5, auto_threshold=True,
            compute_top_k=True, k_values=[100, 500, 1000],
            verbose=False,
        )

    print("[1/2] Reproducing reported test metrics (original boosted feat) ...")
    m_boosted = eval_on(feat_boosted, test_idx, args.eval_batch_size)
    print_metrics(m_boosted, f"TEST {args.theory} ({args.intensity}) -- boosted, as trained/reported")

    print("\n[2/2] Pristine-test robustness check (test rows de-boosted) ...")
    m_pristine = eval_on(feat_pristine_test, test_idx, args.eval_batch_size)
    print_metrics(m_pristine, f"TEST {args.theory} ({args.intensity}) -- pristine (de-boosted)")

    print("\n" + "=" * 70)
    print(f"{'metric':<12}{'boosted (reported)':>22}{'pristine (robustness)':>24}{'delta':>10}")
    for k in ["aupr", "roc_auc", "precision", "recall", "f1"]:
        bv = float(m_boosted.get(k, float("nan")))
        pv = float(m_pristine.get(k, float("nan")))
        print(f"{k:<12}{bv:>22.4f}{pv:>24.4f}{pv - bv:>10.4f}")
    print("=" * 70)
    print(
        "\nInterpretation: if AUPR/ROC-AUC on the pristine test set stay close "
        "to the reported (boosted) test numbers, that's evidence the model's "
        "test-time performance reflects real structural/behavioral signal, not "
        "just detecting the synthetic boost applied to test-set laundering rows."
    )

    out = {
        "theory": args.theory,
        "intensity": args.intensity,
        "model_path": args.model_path,
        "n_test_edges": int(test_idx.numel()),
        "n_test_positive": int(n_pos),
        "boosted": m_boosted,
        "pristine": m_pristine,
    }
    out_dir = os.path.dirname(os.path.abspath(args.model_path))
    out_path = os.path.join(out_dir, f"pristine_test_check_{args.theory}_{args.intensity}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
