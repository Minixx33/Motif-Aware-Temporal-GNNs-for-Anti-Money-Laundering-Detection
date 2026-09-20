"""
stratified_theory_eval.py
----------------------------------------------------------------------
Standard stratified/sliced evaluation, promoted from the one-off check run
during the Sept 21 2026 review ("does RAT/SLT's signal concentrate where
the theory predicts?") into a reusable metric computed for every experiment.

WHAT IT DOES
For a trained run's test-set predictions, in addition to the usual
aggregate AUPR/ROC-AUC, this computes the SAME metrics restricted to
theory-motivated subgroups of the test set:

  RAT slices (opportunity / guardianship, from RAT_pristine's edge_attr):
    - RAT_is_off_hours (0/1)
    - RAT_is_cross_bank (0/1)
    - RAT_is_weekend (0/1)
    - RAT_mutual_flag (0/1)
    - joint: {daytime,off-hours} x {same-bank,cross-bank}

  SLT slices (peer exposure / social influence, from SLT_pristine's edge_attr):
    - SLT_same_entity (0/1)
    - src_peer_risk_score / dst_peer_risk_score (median split)
    - SLT_src_strong_tie_susp_ratio / SLT_dst_strong_tie_susp_ratio (top decile)

Both slice families are computed for EVERY run regardless of which
theory's edge_attr that run's own graph has -- so you can see, e.g.,
whether rat_natural's predictions improve over baseline's specifically in
the cross-bank/weekend subgroups, not just in aggregate. Slice membership
comes from a fixed "donor" graph (RAT_pristine / SLT_pristine, whichever
build has the theory's natural features) and is applied to whatever run
you're evaluating; the script verifies edge_index/y_edge (or src/dst/
labels for DyRep) match between the donor and the target graph before
using the donor's flags on the target's predictions, and aborts/skips
rather than silently mixing up rows if they don't.

TWO SCORE SOURCES
  1. Trained-model predictions (preferred): reads test_pred_probs.pt from
     the run's results directory.
  2. Raw-feature fallback (--allow_raw_feature_fallback): when a run has
     no saved predictions yet (e.g. results synced elsewhere), scores
     each edge with the donor graph's own RAT_score/SLT_score directly.
     This is NOT a model evaluation -- it's the same raw-feature check
     from the review, and output is tagged raw_feature_fallback: true so
     it's never confused with a real model result.

USAGE
  Single run:
    python scripts/analysis/stratified_theory_eval.py \\
        --graph_dir graphs/HI-Small_Trans_RAT_pristine \\
        --split_dir splits/HI-Small_Trans_RAT_pristine \\
        --results_dir results/HI-Small_Trans_RAT_pristine/seed1_causal_leakfix_v1/graphsage-t

  All experiments (batch), falling back to raw features where predictions
  aren't available locally yet:
    python scripts/analysis/stratified_theory_eval.py \\
        --results_root results --name_filter causal_leakfix_v1 \\
        --allow_raw_feature_fallback
----------------------------------------------------------------------
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.evaluation_utils import evaluate_binary_classifier  # noqa: E402
from scripts.analysis.recompute_test_metrics_val_threshold import (  # noqa: E402
    resolve_graph_and_split_dirs,
)

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

DEFAULT_DONORS = {
    "RAT": {
        "static": ("graphs/HI-Small_Trans_RAT_pristine", "splits/HI-Small_Trans_RAT_pristine"),
        "dyrep": ("graphs_dyrep/HI-Small_Trans_RAT_pristine", "splits_dyrep/HI-Small_Trans_RAT_pristine"),
    },
    "SLT": {
        "static": ("graphs/HI-Small_Trans_SLT_pristine", "splits/HI-Small_Trans_SLT_pristine"),
        "dyrep": ("graphs_dyrep/HI-Small_Trans_SLT_pristine", "splits_dyrep/HI-Small_Trans_SLT_pristine"),
    },
}

# (column, slice_type, extra) -- extra is the percentile for top_pct slices.
RAT_SLICE_SPECS = [
    ("RAT_is_off_hours", "binary", None),
    ("RAT_is_cross_bank", "binary", None),
    ("RAT_is_weekend", "binary", None),
    ("RAT_mutual_flag", "binary", None),
]
SLT_SLICE_SPECS = [
    ("SLT_same_entity", "binary", None),
    ("src_peer_risk_score", "median", None),
    ("dst_peer_risk_score", "median", None),
    ("SLT_src_strong_tie_susp_ratio", "top_pct", 90),
    ("SLT_dst_strong_tie_susp_ratio", "top_pct", 90),
]
RAT_SCORE_COL = "RAT_score"
SLT_SCORE_COL = "SLT_score"


# ------------------------------------------------------------------
# Torch-free-safe helpers (work whether torch is real or a stub; only
# need .numpy()-style access, which real torch tensors support directly)
# ------------------------------------------------------------------

def _to_numpy(t):
    return t.numpy() if hasattr(t, "numpy") else np.asarray(t)


def _binary_mask(vals):
    uniq = np.unique(vals)
    if len(uniq) != 2:
        return None
    lo, hi = uniq
    return vals >= (lo + hi) / 2.0


def _median_mask(vals):
    med = np.median(vals)
    return vals > med


def _top_pct_mask(vals, pct):
    thresh = np.percentile(vals, pct)
    return vals >= thresh


def _slice_metrics(y, scores, mask, label, min_n=20, min_pos=2):
    n = int(mask.sum())
    n_pos = int(y[mask].sum()) if n > 0 else 0
    row = {"label": label, "n": n, "n_pos": n_pos}
    if n < min_n or n_pos < min_pos:
        row["note"] = "too few positives to score reliably"
        return row
    base_rate = float(y[mask].mean())
    aupr = float(average_precision_score(y[mask], scores[mask]))
    auc = float(roc_auc_score(y[mask], scores[mask]))
    row.update({
        "base_rate": base_rate,
        "aupr": aupr,
        "roc_auc": auc,
        "lift": (aupr / base_rate) if base_rate > 0 else float("nan"),
    })
    return row


def compute_stratified(y, scores, edge_attr, cols, specs, joint_rat=False):
    """y, scores: 1D arrays for the test rows being evaluated.
    edge_attr/cols: donor graph's edge_attr (same row set/order as y/scores)
    and its column name list."""
    out = {"overall": _slice_metrics(y, scores, np.ones_like(y, dtype=bool), "overall")}

    def col(name):
        return edge_attr[:, cols.index(name)]

    for name, kind, extra in specs:
        if name not in cols:
            out[name] = {"note": f"column '{name}' not present in donor graph"}
            continue
        vals = col(name)
        if kind == "binary":
            mask = _binary_mask(vals)
            if mask is None:
                out[name] = {"note": "expected a binary flag but found >2 unique values"}
                continue
            out[f"{name}=0"] = _slice_metrics(y, scores, ~mask, f"{name}=0")
            out[f"{name}=1"] = _slice_metrics(y, scores, mask, f"{name}=1")
        elif kind == "median":
            mask = _median_mask(vals)
            out[f"{name}<=median"] = _slice_metrics(y, scores, ~mask, f"{name}<=median")
            out[f"{name}>median"] = _slice_metrics(y, scores, mask, f"{name}>median")
        elif kind == "top_pct":
            mask = _top_pct_mask(vals, extra)
            out[f"{name}<p{extra}"] = _slice_metrics(y, scores, ~mask, f"{name}<p{extra}")
            out[f"{name}>=p{extra}"] = _slice_metrics(y, scores, mask, f"{name}>=p{extra}")

    if joint_rat and "RAT_is_off_hours" in cols and "RAT_is_cross_bank" in cols:
        oh = _binary_mask(col("RAT_is_off_hours"))
        cb = _binary_mask(col("RAT_is_cross_bank"))
        if oh is not None and cb is not None:
            out["joint_daytime_samebank"] = _slice_metrics(y, scores, ~oh & ~cb, "daytime+same-bank")
            out["joint_daytime_crossbank"] = _slice_metrics(y, scores, ~oh & cb, "daytime+cross-bank")
            out["joint_offhours_samebank"] = _slice_metrics(y, scores, oh & ~cb, "off-hours+same-bank")
            out["joint_offhours_crossbank"] = _slice_metrics(y, scores, oh & cb, "off-hours+cross-bank")

    return out


def _load_labels(graph_dir, label_file):
    return _to_numpy(torch.load(os.path.join(graph_dir, label_file), weights_only=False))


def _load_structure(graph_dir):
    if os.path.exists(os.path.join(graph_dir, "edge_index.pt")):
        return torch.load(os.path.join(graph_dir, "edge_index.pt"), weights_only=False)
    return torch.stack([
        torch.load(os.path.join(graph_dir, "src.pt"), weights_only=False),
        torch.load(os.path.join(graph_dir, "dst.pt"), weights_only=False),
    ])


class DonorCache:
    """Loads a donor graph's structure/labels/edge_attr ONCE and reuses it
    across every run in a batch -- edge_attr.pt for these graphs is
    hundreds of MB, and reloading it per run (dozens of times in a batch)
    is wasteful and can exhaust memory. Restricts edge_attr to the test
    split immediately so only the ~20% test-row slice is kept resident,
    not the full graph."""

    def __init__(self, donor_graph, donor_split, label_file):
        self.graph_dir = donor_graph
        self.label_file = label_file
        self.structure = _to_numpy(_load_structure(donor_graph))
        self.y_full = _load_labels(donor_graph, label_file)
        self.test_idx = _to_numpy(torch.load(os.path.join(donor_split, "test_edge_idx.pt"), weights_only=False))
        self.cols = json.load(open(os.path.join(donor_graph, "edge_attr_cols.json")))
        full_edge_attr = _to_numpy(torch.load(os.path.join(donor_graph, "edge_attr.pt"), weights_only=False))
        self.edge_attr_test = full_edge_attr[self.test_idx]
        del full_edge_attr  # only the test-row slice needs to stay resident
        self.y_test = self.y_full[self.test_idx]


def _assert_aligned_to_donor(target_graph_dir, target_label_file, donor: DonorCache, donor_label):
    """Verify the donor graph's rows line up with the target graph's rows
    (same underlying transactions/labels), so donor flags can be applied
    to the target's predictions."""
    t_struct = _to_numpy(_load_structure(target_graph_dir))
    if t_struct.shape != donor.structure.shape or not np.array_equal(t_struct, donor.structure):
        return False, f"structure (edge_index/src+dst) mismatch vs. {donor_label} donor"

    t_y = _load_labels(target_graph_dir, target_label_file)
    if not np.array_equal(t_y, donor.y_full):
        return False, f"labels mismatch vs. {donor_label} donor"
    return True, None


def run_one(graph_dir, split_dir, label_file, donors, results_dir=None, raw_feature_fallback=False):
    """donors: dict {"RAT": DonorCache, "SLT": DonorCache}, pre-loaded once
    per batch (see DonorCache). Returns a dict with 'RAT'/'SLT' stratified
    breakdowns for one run, or None if it couldn't be evaluated."""
    test_idx = _to_numpy(torch.load(os.path.join(split_dir, "test_edge_idx.pt"), weights_only=False))
    y_full = _load_labels(graph_dir, label_file)
    y_test = y_full[test_idx]

    used_raw_feature = False
    scores = None
    if results_dir is not None:
        probs_path = os.path.join(results_dir, "test_pred_probs.pt")
        if os.path.exists(probs_path):
            scores = _to_numpy(torch.load(probs_path, weights_only=False))
            if len(scores) != len(test_idx):
                print(f"  [SKIP] {results_dir}: test_pred_probs.pt length {len(scores)} "
                      f"!= test split size {len(test_idx)}")
                return None

    if scores is None:
        if not raw_feature_fallback:
            print(f"  [SKIP] {graph_dir}: no test_pred_probs.pt found and "
                  f"raw-feature fallback not enabled")
            return None
        used_raw_feature = True

    out = {"used_raw_feature_fallback": used_raw_feature, "theories": {}}

    for theory, score_col in [("RAT", RAT_SCORE_COL), ("SLT", SLT_SCORE_COL)]:
        donor = donors.get(theory)
        if donor is None:
            out["theories"][theory] = {"note": "donor graph not available"}
            continue

        aligned, why = _assert_aligned_to_donor(graph_dir, label_file, donor, theory)
        if not aligned:
            out["theories"][theory] = {"note": f"skipped -- {why}"}
            continue
        if not np.array_equal(donor.test_idx, test_idx):
            out["theories"][theory] = {"note": "skipped -- donor's test split indices differ from target's"}
            continue

        this_scores = scores
        if used_raw_feature:
            if score_col not in donor.cols:
                out["theories"][theory] = {"note": f"raw fallback: '{score_col}' not in donor columns"}
                continue
            this_scores = donor.edge_attr_test[:, donor.cols.index(score_col)]

        specs = RAT_SLICE_SPECS if theory == "RAT" else SLT_SLICE_SPECS
        out["theories"][theory] = compute_stratified(
            y_test, this_scores, donor.edge_attr_test, donor.cols, specs,
            joint_rat=(theory == "RAT"),
        )

    return out


def print_summary(theory_name, breakdown):
    print(f"  -- {theory_name} --")
    for key, row in breakdown.items():
        if "note" in row and "aupr" not in row:
            print(f"    {key:<28} {row['note']}")
            continue
        if "note" in row:
            print(f"    {key:<28} n={row['n']:>9,}  {row['note']}")
            continue
        print(f"    {key:<28} n={row['n']:>9,} pos={row['n_pos']:<5} "
              f"base_rate={row['base_rate']*100:>7.4f}% AUPR={row['aupr']:.4f} "
              f"lift={row['lift']:.2f}x ROC-AUC={row['roc_auc']:.4f}")


_donor_cache_by_format = {}


def _get_donors(format_kind):
    """Lazily build (and cache for the rest of the process) the RAT/SLT
    DonorCache objects for a given graph format (static/dyrep). Reused
    across every run in a batch instead of reloading each donor's
    edge_attr.pt (hundreds of MB) per run."""
    if format_kind in _donor_cache_by_format:
        return _donor_cache_by_format[format_kind]

    donors = {}
    for theory in ("RAT", "SLT"):
        donor_graph_rel, donor_split_rel = DEFAULT_DONORS[theory][format_kind]
        donor_graph = str(PROJECT_ROOT / donor_graph_rel)
        donor_split = str(PROJECT_ROOT / donor_split_rel)
        if not os.path.isdir(donor_graph):
            print(f"  [WARN] {theory} donor graph not found: {donor_graph} -- "
                  f"{theory} slices will be skipped for {format_kind} runs")
            continue
        label_file = "labels.pt" if format_kind == "dyrep" else "y_edge.pt"
        donors[theory] = DonorCache(donor_graph, donor_split, label_file)

    _donor_cache_by_format[format_kind] = donors
    return donors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph_dir", type=str, default=None)
    ap.add_argument("--split_dir", type=str, default=None)
    ap.add_argument("--results_dir", type=str, default=None,
                     help="Single-run mode: directory containing test_pred_probs.pt")
    ap.add_argument("--results_root", type=str, default=None,
                     help="Batch mode: scan results/<dataset>/<seed_exp>/<model>/ under this root")
    ap.add_argument("--name_filter", type=str, default=None,
                     help="Batch mode: only process seed{N}_<exp_name> folders containing this substring")
    ap.add_argument("--allow_raw_feature_fallback", action="store_true",
                     help="If a run has no test_pred_probs.pt, score edges with the donor's "
                          "own RAT_score/SLT_score instead of a trained model's predictions "
                          "(clearly tagged in the output as used_raw_feature_fallback: true)")
    args = ap.parse_args()

    if args.results_root:
        pattern = str(Path(args.results_root).resolve() / "*" / "*" / "*" / "metrics.json")
        all_paths = sorted(Path(p) for p in glob.glob(pattern))
        if args.name_filter:
            all_paths = [p for p in all_paths if args.name_filter in p.parent.parent.name]
        print(f"Found {len(all_paths)} run(s) to process.\n")

        for metrics_path in all_paths:
            model_dirname = metrics_path.parent.name
            dataset_name = metrics_path.parent.parent.parent.name
            format_kind = "dyrep" if model_dirname == "dyrep" else "static"
            graph_dir, split_dir, label_file = resolve_graph_and_split_dirs(
                PROJECT_ROOT, dataset_name, model_dirname
            )
            print(f"Processing: {metrics_path}")
            if not graph_dir.is_dir() or not split_dir.is_dir():
                print(f"  [SKIP] graph/split dir not found ({graph_dir} / {split_dir})")
                continue

            donors = _get_donors(format_kind)
            result = run_one(
                str(graph_dir), str(split_dir), label_file, donors,
                results_dir=str(metrics_path.parent),
                raw_feature_fallback=args.allow_raw_feature_fallback,
            )
            if result is None:
                continue

            out_path = metrics_path.parent / "stratified_eval.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            for theory, breakdown in result["theories"].items():
                if "note" in breakdown:
                    print(f"  {theory}: {breakdown['note']}")
                else:
                    print_summary(theory, breakdown)
            print(f"  Saved: {out_path}\n")

    else:
        if not (args.graph_dir and args.split_dir):
            ap.error("--graph_dir and --split_dir are required in single-run mode")
        label_file = "labels.pt" if os.path.exists(os.path.join(args.graph_dir, "labels.pt")) else "y_edge.pt"
        format_kind = "dyrep" if label_file == "labels.pt" else "static"
        donors = _get_donors(format_kind)
        result = run_one(
            args.graph_dir, args.split_dir, label_file, donors,
            results_dir=args.results_dir,
            raw_feature_fallback=args.allow_raw_feature_fallback,
        )
        if result is None:
            return
        for theory, breakdown in result["theories"].items():
            if "note" in breakdown:
                print(f"{theory}: {breakdown['note']}")
            else:
                print_summary(theory, breakdown)
        if args.results_dir:
            out_path = Path(args.results_dir) / "stratified_eval.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
