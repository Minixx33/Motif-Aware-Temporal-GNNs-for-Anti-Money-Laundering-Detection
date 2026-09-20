"""
recompute_test_metrics_val_threshold.py
----------------------------------------------------------------------
Post-hoc correction for the test-threshold-leakage bug found in the
Sept 21 2026 pre-cluster logic audit.

BACKGROUND
train_graphsage.py / train_graphsage_t.py / train_dyrep.py all called
evaluate_binary_classifier(..., auto_threshold=True) independently on
train, val, AND test. auto_threshold picks whatever decision threshold
maximizes F1 by scanning the SPLIT'S OWN true labels -- so the test-set
precision/recall/f1/balanced_accuracy/mcc/kappa/specificity/confusion_matrix
in every existing metrics.json were selected with oracle access to test
labels. Standard practice is to pick the threshold on val only, then
apply that same fixed threshold to test. AUPR and ROC-AUC are threshold-
independent and are NOT affected by this bug -- the primary 4-condition
comparison (which used AUPR) is not invalidated. This script only
corrects the threshold-dependent numbers, and needs no retraining: every
run already saved val_pred_probs.pt / test_pred_probs.pt, so the correct
metrics can be recomputed directly from those.

The three training scripts have been fixed going forward (they now pass
val's selected threshold into the test evaluation). This script back-
fills corrected numbers for runs that already finished under the old
(buggy) evaluation code.

WHAT IT DOES
For every metrics.json found under --results_root (optionally filtered
by --name_filter, a substring match on the "seed{N}_<exp_name>" folder):
  1. Parses dataset_name / model_name straight from the path
     (results/<dataset_name>/seed{N}_<exp_name>/<model_name>/metrics.json).
  2. Loads y_edge (or labels for DyRep) from the matching graph directory
     and train/val/test_edge_idx.pt from the matching split directory.
  3. Loads val_pred_probs.pt / test_pred_probs.pt from the same directory
     as metrics.json.
  4. Recomputes val metrics with auto_threshold=True (reproduces the
     original val block/threshold as a sanity check) and test metrics
     with that SAME threshold fixed (auto_threshold=False).
  5. Writes metrics_corrected.json next to the original metrics.json
     (original file is left untouched) and prints a before/after table.

USAGE
    python scripts/analysis/recompute_test_metrics_val_threshold.py \\
        --results_root results \\
        --name_filter causal_leakfix_v1
----------------------------------------------------------------------
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.evaluation_utils import evaluate_binary_classifier  # noqa: E402

DYREP_MODEL_DIRNAMES = {"dyrep"}


def resolve_graph_and_split_dirs(project_root, dataset_name, model_dirname):
    if model_dirname in DYREP_MODEL_DIRNAMES:
        graph_dir = project_root / "graphs_dyrep" / dataset_name
        split_dir = project_root / "splits_dyrep" / dataset_name
        label_file = "labels.pt"
    else:
        graph_dir = project_root / "graphs" / dataset_name
        split_dir = project_root / "splits" / dataset_name
        label_file = "y_edge.pt"
    return graph_dir, split_dir, label_file


def recompute_one(metrics_path: Path, project_root: Path):
    # results/<dataset_name>/seed{N}_<exp_name>/<model_name>/metrics.json
    model_dirname = metrics_path.parent.name
    dataset_name = metrics_path.parent.parent.parent.name

    graph_dir, split_dir, label_file = resolve_graph_and_split_dirs(
        project_root, dataset_name, model_dirname
    )

    if not graph_dir.is_dir() or not split_dir.is_dir():
        print(f"  [SKIP] {metrics_path}: graph/split dir not found "
              f"({graph_dir} / {split_dir})")
        return None

    val_probs_path = metrics_path.parent / "val_pred_probs.pt"
    test_probs_path = metrics_path.parent / "test_pred_probs.pt"
    if not val_probs_path.exists() or not test_probs_path.exists():
        print(f"  [SKIP] {metrics_path}: missing val/test_pred_probs.pt")
        return None

    with open(metrics_path) as f:
        original = json.load(f)

    labels_all = torch.load(graph_dir / label_file, weights_only=False).numpy()
    val_idx = torch.load(split_dir / "val_edge_idx.pt", weights_only=False).numpy()
    test_idx = torch.load(split_dir / "test_edge_idx.pt", weights_only=False).numpy()

    val_probs = torch.load(val_probs_path, weights_only=False).numpy()
    test_probs = torch.load(test_probs_path, weights_only=False).numpy()

    y_val = labels_all[val_idx]
    y_test = labels_all[test_idx]

    if len(y_val) != len(val_probs) or len(y_test) != len(test_probs):
        print(f"  [SKIP] {metrics_path}: length mismatch "
              f"(val {len(y_val)} vs {len(val_probs)}, "
              f"test {len(y_test)} vs {len(test_probs)})")
        return None

    # Recompute val the same way it was originally scored (auto-threshold
    # on val's own labels -- that's legitimate, val is what you're allowed
    # to tune on). This should reproduce the original val block closely as
    # a sanity check.
    val_metrics = evaluate_binary_classifier(
        y_val, val_probs, auto_threshold=True,
        compute_top_k=True, k_values=[100, 500, 1000], verbose=False,
    )

    # Score test with val's threshold fixed, instead of test auto-picking
    # its own (the actual fix).
    test_metrics_corrected = evaluate_binary_classifier(
        y_test, test_probs,
        threshold=val_metrics["threshold"], auto_threshold=False,
        compute_top_k=True, k_values=[100, 500, 1000], verbose=False,
    )

    corrected = dict(original)
    corrected["val"] = val_metrics
    corrected["test"] = test_metrics_corrected
    corrected["_correction_note"] = (
        "test metrics recomputed with the threshold selected on val "
        "(auto_threshold on val), instead of test auto-selecting its own "
        "threshold from its own labels. See recompute_test_metrics_val_threshold.py. "
        "AUPR/ROC-AUC are threshold-independent and identical to the original "
        "metrics.json; only threshold-dependent fields below differ."
    )

    out_path = metrics_path.parent / "metrics_corrected.json"
    with open(out_path, "w") as f:
        json.dump(corrected, f, indent=2)

    orig_test = original.get("test", {})
    return {
        "path": str(metrics_path),
        "dataset": dataset_name,
        "model": model_dirname,
        "orig_test_threshold": orig_test.get("threshold"),
        "new_test_threshold": test_metrics_corrected["threshold"],
        "orig_test_f1": orig_test.get("f1"),
        "new_test_f1": test_metrics_corrected["f1"],
        "orig_test_precision": orig_test.get("precision"),
        "new_test_precision": test_metrics_corrected["precision"],
        "orig_test_recall": orig_test.get("recall"),
        "new_test_recall": test_metrics_corrected["recall"],
        "test_aupr_unchanged": abs(
            orig_test.get("aupr", float("nan")) - test_metrics_corrected["aupr"]
        ) < 1e-9,
        "out_path": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Recompute test precision/recall/F1/etc using val's "
                     "selected threshold, for runs already trained under "
                     "the old (test-auto-thresholds-itself) evaluation code."
    )
    ap.add_argument("--results_root", type=str, default="results")
    ap.add_argument("--name_filter", type=str, default=None,
                     help="Only process runs whose seed{N}_<exp_name> folder "
                          "name contains this substring, e.g. causal_leakfix_v1")
    args = ap.parse_args()

    project_root = PROJECT_ROOT
    results_root = (project_root / args.results_root).resolve()

    pattern = str(results_root / "*" / "*" / "*" / "metrics.json")
    all_paths = sorted(Path(p) for p in glob.glob(pattern))

    if args.name_filter:
        all_paths = [p for p in all_paths if args.name_filter in p.parent.parent.name]

    print(f"Found {len(all_paths)} metrics.json file(s) to process"
          f"{f' (filtered by {args.name_filter!r})' if args.name_filter else ''}.\n")

    rows = []
    for p in all_paths:
        print(f"Processing: {p}")
        row = recompute_one(p, project_root)
        if row:
            rows.append(row)

    if not rows:
        print("\nNothing recomputed.")
        return

    print("\n" + "=" * 100)
    print(f"{'dataset':<32}{'model':<12}{'orig_thr':>9}{'new_thr':>9}"
          f"{'orig_F1':>9}{'new_F1':>9}{'orig_P':>8}{'new_P':>8}{'orig_R':>8}{'new_R':>8}{'AUPR OK':>9}")
    for r in rows:
        print(
            f"{r['dataset']:<32}{r['model']:<12}"
            f"{r['orig_test_threshold']:>9.4f}{r['new_test_threshold']:>9.4f}"
            f"{r['orig_test_f1']:>9.4f}{r['new_test_f1']:>9.4f}"
            f"{r['orig_test_precision']:>8.4f}{r['new_test_precision']:>8.4f}"
            f"{r['orig_test_recall']:>8.4f}{r['new_test_recall']:>8.4f}"
            f"{'yes' if r['test_aupr_unchanged'] else 'NO':>9}"
        )
    print("=" * 100)
    print(f"\nWrote {len(rows)} metrics_corrected.json file(s), one next to each original metrics.json.")
    print("Original metrics.json files were left untouched.")


if __name__ == "__main__":
    main()
