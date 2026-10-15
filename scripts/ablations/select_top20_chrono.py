"""
select_top20_chrono.py -- leak-free replacement for the old "top20_features" RAT ablation.

The old TOP20_FEATURES list (rat_ablation_groups.py) came from a random forest
trained on the label-boosted RAT_medium CSV with a RANDOM 70/15/15 split, so
labels from our chronological test period influenced which features were kept.

This script re-derives the top-20 using ONLY the chronological TRAIN split of
the rat_natural graph (graphs/HI-Small_Trans_RAT_pristine), in graph-column
space (no one-hot translation needed). Val is used only for a sanity AUPR;
the test split is never loaded. It then builds the ablation graph
graphs/HI-Small_Trans_RAT_pristine__top20_chrono (edge columns subset; every
other file copied unchanged) via run_ablation_static.

  python scripts/ablations/select_top20_chrono.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_ablation_static import run_ablation_static  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--graph", default="graphs/HI-Small_Trans_RAT_pristine")
ap.add_argument("--splits", default="splits/HI-Small_Trans_RAT_pristine")
ap.add_argument("--out_graph", default="graphs/HI-Small_Trans_RAT_pristine__top20_chrono")
ap.add_argument("--out_dir", default="results/feature_importance/rat_natural_chrono")
ap.add_argument("--k", type=int, default=20)
ap.add_argument("--neg_sample", type=int, default=400_000,
                help="random train negatives kept for the RF (all train positives are kept)")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

# ts_normalized is absolute time fit on the full date range; the v2 models drop
# it, so it is not a candidate.
EXCLUDE = {"ts_normalized"}

edge_attr = torch.load(os.path.join(args.graph, "edge_attr.pt")).float().numpy()
cols = json.load(open(os.path.join(args.graph, "edge_attr_cols.json")))
y = torch.load(os.path.join(args.graph, "y_edge.pt")).long().numpy()
ts = torch.load(os.path.join(args.graph, "timestamps.pt")).numpy()
tr = torch.load(os.path.join(args.splits, "train_edge_idx.pt")).numpy()
va = torch.load(os.path.join(args.splits, "val_edge_idx.pt")).numpy()
assert ts[tr].max() <= ts[va].min(), "split is not chronological -- refusing to run"

cand = [c for c in cols if c not in EXCLUDE]
ci = [cols.index(c) for c in cand]
print(f"[top20] {len(cand)} candidate columns (excluded {sorted(EXCLUDE & set(cols))})")

rng = np.random.default_rng(args.seed)
pos = tr[y[tr] == 1]
neg = tr[y[tr] == 0]
neg = rng.choice(neg, size=min(args.neg_sample, len(neg)), replace=False)
idx = np.sort(np.concatenate([pos, neg]))
print(f"[top20] RF on TRAIN only: {len(pos)} positives + {len(neg)} sampled negatives")

rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                            min_samples_leaf=5, n_jobs=-1, random_state=args.seed)
rf.fit(edge_attr[idx][:, ci], y[idx])
val_aupr = average_precision_score(y[va], rf.predict_proba(edge_attr[va][:, ci])[:, 1])
print(f"[top20] sanity: val AUPR = {val_aupr:.4f} (test split never loaded)")

imp = pd.DataFrame({"feature": cand, "importance": rf.feature_importances_})
imp = imp.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
top = imp["feature"].head(args.k).tolist()

os.makedirs(args.out_dir, exist_ok=True)
imp.to_csv(os.path.join(args.out_dir, "feature_importances_train_only.csv"), index=False)
json.dump({"source_graph": args.graph, "selection": "RF impurity importance, chronological TRAIN split only",
           "train_pos": int(len(pos)), "train_neg_sampled": int(len(neg)), "val_aupr": val_aupr,
           "top_k": top}, open(os.path.join(args.out_dir, "top20.json"), "w"), indent=2)
print("[top20] selected:\n  " + "\n  ".join(f"{i+1:2d}. {f}" for i, f in enumerate(top)))

run_ablation_static(args.graph, args.out_graph, top)
print("[top20] done")
