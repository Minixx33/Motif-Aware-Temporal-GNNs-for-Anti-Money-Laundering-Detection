"""
slt_causal_baselines.py
---------------------------------------------------------------------------
Feature-only (NON-graph) diagnostic for the SLT study. A GraphSAGE encoder
already aggregates neighbour information, so an SLT feature can look useless
there simply because the GNN re-derives it. A tabular model has no neighbour
aggregation at all, so if a feature group carries real signal it MUST show up
here. Comparing the two answers "is SLT redundant with message passing, or
just weak?".

Same rows, same chronological split, same feature definitions as the GNN
conditions (features come from the SAME compute_features() used to build the
graphs). Every feature set is baseline(12) + ONE added group, so each delta vs
`base` is attributable to that group only:

  base                       12 baseline transaction features
  base+SLT_natural           + the existing UNSUPERVISED slt_natural features
                             (graphs/HI-Small_Trans_SLT_pristine, row-aligned
                             and verified identical in edges/labels)
  base+SELF                  + account-reputation control (NOT SLT)
  base+PEER1                 + direct causal peer exposure
  base+PEER1+MULTI           + decayed / 2-hop exposure
  base+SELF+PEER1+MULTI      peer exposure BEYOND reputation
                             (compare against base+SELF)
  base+PLACEBO               PEER1+MULTI computed from permuted train labels
  [lag sensitivity]          SELF, PEER1+MULTI, SELF+PEER1+MULTI recomputed at
                             other reporting lags (--sens_lags_hours)

Models: logistic regression and sklearn HistGradientBoosting (no external
dependency). Fit on train only (features standardised on train), evaluated on
val and test with threshold-free AUPR / ROC-AUC. GBM is run for several seeds.

Usage (cluster, CPU):
    python scripts/analysis/slt_causal_baselines.py \\
        --output_json results_baselines/slt_causal_baselines.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

from build_slt_causal_graphs import (  # noqa: E402
    compute_features, SELF_COLS, PEER1_COLS, MULTI_COLS,
)


def _load(d, name):
    import torch
    return torch.load(os.path.join(d, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_graph_dir", default=str(PROJECT_ROOT / "graphs" / "HI-Small_Trans"))
    ap.add_argument("--baseline_split_dir", default=str(PROJECT_ROOT / "splits" / "HI-Small_Trans"))
    ap.add_argument("--natural_slt_graph_dir", default=str(PROJECT_ROOT / "graphs" / "HI-Small_Trans_SLT_pristine"))
    ap.add_argument("--lag_hours", type=float, default=24.0)
    ap.add_argument("--sens_lags_hours", type=str, default="0,72")
    ap.add_argument("--halflife_days", type=float, default=3.0)
    ap.add_argument("--seeds", type=str, default="1 2 3 4 5", help="seeds for the GBM")
    ap.add_argument("--placebo_seed", type=int, default=123)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    bdir, sdir = args.baseline_graph_dir, args.baseline_split_dir
    base_cols = json.load(open(os.path.join(bdir, "edge_attr_cols.json")))
    ei = _load(bdir, "edge_index.pt").numpy()
    base_X = _load(bdir, "edge_attr.pt").numpy().astype(np.float32)
    t_sec = _load(bdir, "timestamps.pt").numpy().astype(np.int64)
    y = _load(bdir, "y_edge.pt").numpy().astype(np.int64)
    n_nodes = int(_load(bdir, "x.pt").shape[0])
    tr = _load(sdir, "train_edge_idx.pt").numpy()
    va = _load(sdir, "val_edge_idx.pt").numpy()
    te = _load(sdir, "test_edge_idx.pt").numpy()
    assert t_sec[tr].max() <= t_sec[va].min() and t_sec[tr].max() <= t_sec[te].min(), \
        "split is not chronological -- rebuild with --split_mode chronological"
    E = len(y)
    src, dst = ei[0].astype(np.int64), ei[1].astype(np.int64)
    w_amt = base_X[:, base_cols.index("log_amt_paid")].astype(np.float64)
    train_mask = np.zeros(E, dtype=bool)
    train_mask[tr] = True
    print(f"E={E:,} train/val/test={len(tr):,}/{len(va):,}/{len(te):,}")

    # natural (unsupervised) SLT block, verified row-aligned with the baseline
    nat_block, nat_cols = None, []
    ndir = args.natural_slt_graph_dir
    if os.path.isdir(ndir):
        n_ei = _load(ndir, "edge_index.pt")
        n_y = _load(ndir, "y_edge.pt")
        if (n_ei.numpy() == ei).all() and (n_y.numpy() == y).all():
            n_cols = json.load(open(os.path.join(ndir, "edge_attr_cols.json")))
            n_X = _load(ndir, "edge_attr.pt").numpy().astype(np.float32)
            keep = [i for i, c in enumerate(n_cols) if c not in base_cols]
            nat_cols = [n_cols[i] for i in keep]
            nat_block = n_X[:, keep]
            print(f"natural SLT block: {len(nat_cols)} columns")
        else:
            print("[WARN] natural SLT graph not row-aligned with baseline -- skipping that feature set")
    else:
        print(f"[WARN] {ndir} not found -- skipping natural SLT feature set")

    def feats_for(lag_h, labels):
        return compute_features(src, dst, t_sec, w_amt, labels, train_mask, n_nodes,
                                lag_sec=int(lag_h * 3600), halflife_days=args.halflife_days)

    def block(f, cols):
        return np.stack([f[c] for c in cols], axis=1).astype(np.float32)

    # Each feature set is a LIST of column blocks, concatenated only when fitted
    # (keeps peak memory low -- ~13 sets x 5M rows would not fit in 14 GB).
    sets = {"base": [base_X]}
    if nat_block is not None:
        sets["base+SLT_natural"] = [base_X, nat_block]

    lag = args.lag_hours
    print(f"\nComputing causal features (lag={lag}h) ...")
    f_true = feats_for(lag, y)
    rng = np.random.default_rng(args.placebo_seed)
    y_plc = y.copy()
    y_plc[tr] = rng.permutation(y[tr])
    f_plc = feats_for(lag, y_plc)

    ALL = PEER1_COLS + MULTI_COLS
    sets["base+SELF"] = [base_X, block(f_true, SELF_COLS)]
    sets["base+PEER1"] = [base_X, block(f_true, PEER1_COLS)]
    sets["base+PEER1+MULTI"] = [base_X, block(f_true, ALL)]
    sets["base+SELF+PEER1+MULTI"] = [base_X, block(f_true, SELF_COLS + ALL)]
    sets["base+PLACEBO(PEER1+MULTI)"] = [base_X, block(f_plc, ALL)]

    # feature density by split (documents the cold-start train/test shift)
    density = {}
    for c in SELF_COLS + ALL:
        v = f_true[c]
        density[c] = {"train": float((v[tr] > 0).mean()),
                      "val": float((v[va] > 0).mean()),
                      "test": float((v[te] > 0).mean())}

    for lh in [float(x) for x in args.sens_lags_hours.split(",") if x.strip()]:
        if lh == lag:
            continue
        print(f"Computing sensitivity features (lag={lh}h) ...")
        fl = feats_for(lh, y)
        sets[f"[lag={lh:g}h] base+SELF"] = [base_X, block(fl, SELF_COLS)]
        sets[f"[lag={lh:g}h] base+PEER1+MULTI"] = [base_X, block(fl, ALL)]
        sets[f"[lag={lh:g}h] base+SELF+PEER1+MULTI"] = [base_X, block(fl, SELF_COLS + ALL)]

    seeds = [int(s) for s in args.seeds.split()]
    results = {"lag_hours": lag, "halflife_days": args.halflife_days,
               "feature_density_by_split": density, "sets": {}}
    ytr, yva, yte = y[tr], y[va], y[te]

    def metrics(yy, p):
        return {"aupr": float(average_precision_score(yy, p)),
                "roc_auc": float(roc_auc_score(yy, p))}

    for name, blocks in sets.items():
        t0 = time.time()
        X = np.concatenate(blocks, axis=1)
        sc = StandardScaler().fit(X[tr])
        Xtr, Xva, Xte = sc.transform(X[tr]), sc.transform(X[va]), sc.transform(X[te])
        entry = {"num_features": int(X.shape[1])}

        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
        entry["logistic_regression"] = {
            "val": metrics(yva, lr.predict_proba(Xva)[:, 1]),
            "test": metrics(yte, lr.predict_proba(Xte)[:, 1]),
        }

        runs = []
        for sd in seeds:
            gb = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                class_weight="balanced", early_stopping=False, random_state=sd,
            ).fit(Xtr, ytr)
            runs.append({"seed": sd,
                         "val": metrics(yva, gb.predict_proba(Xva)[:, 1]),
                         "test": metrics(yte, gb.predict_proba(Xte)[:, 1])})
        ta = [r["test"]["aupr"] for r in runs]
        entry["gbm"] = {
            "runs": runs,
            "test_aupr_mean": float(np.mean(ta)), "test_aupr_std": float(np.std(ta, ddof=1)) if len(ta) > 1 else 0.0,
            "test_auc_mean": float(np.mean([r["test"]["roc_auc"] for r in runs])),
        }
        results["sets"][name] = entry
        del X, Xtr, Xva, Xte
        print(f"[{name:38s}] LR test AUPR={entry['logistic_regression']['test']['aupr']:.4f}  "
              f"GBM test AUPR={entry['gbm']['test_aupr_mean']:.4f}±{entry['gbm']['test_aupr_std']:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.output_json}")


if __name__ == "__main__":
    main()
