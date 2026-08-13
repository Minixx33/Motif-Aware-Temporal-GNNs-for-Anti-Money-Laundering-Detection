"""
train_baselines.py
---------------------------------------------------------------------------
Non-graph-based baselines (review feedback Aug 4 2026): logistic regression,
Random Forest, XGBoost/LightGBM, and an MLP, all trained on the existing
tabular edge-feature matrix -- to test whether the GNNs add value beyond
engineered features alone.

Loads edge_attr.pt / edge_attr_cols.json / y_edge.pt directly from a graph
directory, and train/val/test edge-index tensors from the matching split
directory -- the same tensors GraphSAGE/GraphSAGE-T train on -- so results
are computed on IDENTICAL rows/splits and are directly comparable to the GNN
metrics.json outputs. Evaluation goes through the same
evaluate_binary_classifier() used everywhere else in the pipeline.

Models are fit on the train split only (features standardized using train-set
statistics), evaluated on val and test. Class imbalance is handled with
class_weight="balanced" (LR, RF) / scale_pos_weight (XGBoost) / a balanced
resample for MLP's early-stopping validation set -- no SMOTE or synthetic
resampling of the training data itself, to keep this comparable to how the
GNNs are trained (real transactions only).

Usage:
    python scripts/analysis/train_baselines.py \\
        --graph_dir graphs/HI-Small_Trans_RAT_pristine \\
        --split_dir splits/HI-Small_Trans_RAT_pristine \\
        --output_json results_baselines/HI-Small_Trans_RAT_pristine_baselines.json \\
        --seed 42
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.evaluation_utils import evaluate_binary_classifier  # noqa: E402


def load_split_data(graph_dir, split_dir):
    edge_attr = torch.load(os.path.join(graph_dir, "edge_attr.pt")).numpy()
    y_edge = torch.load(os.path.join(graph_dir, "y_edge.pt")).numpy().astype(int)
    with open(os.path.join(graph_dir, "edge_attr_cols.json")) as f:
        cols = json.load(f)

    train_idx = torch.load(os.path.join(split_dir, "train_edge_idx.pt")).numpy()
    val_idx = torch.load(os.path.join(split_dir, "val_edge_idx.pt")).numpy()
    test_idx = torch.load(os.path.join(split_dir, "test_edge_idx.pt")).numpy()

    if edge_attr.shape[0] != y_edge.shape[0]:
        raise ValueError(
            f"edge_attr has {edge_attr.shape[0]} rows but y_edge has "
            f"{y_edge.shape[0]} -- graph directory is inconsistent."
        )

    return edge_attr, y_edge, cols, train_idx, val_idx, test_idx


def fit_scaler(X_train):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def run_logistic_regression(X_train, y_train, X_val, X_test, seed):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=seed
    )
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1]


def run_random_forest(X_train, y_train, X_val, X_test, seed):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1]


def run_xgboost(X_train, y_train, X_val, X_test, seed):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("[SKIP] xgboost not installed -- trying lightgbm instead.")
        return run_lightgbm(X_train, y_train, X_val, X_test, seed)

    n_pos = max(int(y_train.sum()), 1)
    n_neg = max(int(len(y_train) - n_pos), 1)
    scale_pos_weight = n_neg / n_pos

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1]


def run_lightgbm(X_train, y_train, X_val, X_test, seed):
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print("[SKIP] Neither xgboost nor lightgbm is installed -- skipping this baseline.")
        return None, None

    n_pos = max(int(y_train.sum()), 1)
    n_neg = max(int(len(y_train) - n_pos), 1)
    scale_pos_weight = n_neg / n_pos

    clf = LGBMClassifier(
        n_estimators=400,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1]


def run_mlp(X_train, y_train, X_val, X_test, seed):
    from sklearn.neural_network import MLPClassifier
    # sklearn's MLPClassifier has no native class_weight support, so
    # oversample the minority class in the TRAINING data only (val/test are
    # left untouched/imbalanced, matching how the GNNs are evaluated).
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise RuntimeError("Training split has only one class -- cannot fit MLP.")
    target_pos = len(neg_idx)  # balance 1:1
    if len(pos_idx) < target_pos:
        extra = rng.choice(pos_idx, size=target_pos - len(pos_idx), replace=True)
        bal_idx = np.concatenate([np.arange(len(y_train)), extra])
    else:
        bal_idx = np.arange(len(y_train))
    rng.shuffle(bal_idx)

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        max_iter=200,
        early_stopping=True,
        random_state=seed,
    )
    clf.fit(X_train[bal_idx], y_train[bal_idx])
    return clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1]


BASELINES = {
    "logistic_regression": run_logistic_regression,
    "random_forest": run_random_forest,
    "xgboost": run_xgboost,
    "mlp": run_mlp,
}


def main():
    parser = argparse.ArgumentParser(
        description="Non-graph tabular baselines (LR / RF / XGBoost / MLP) "
                     "on the same edge-feature matrix and splits the GNNs use."
    )
    parser.add_argument("--graph_dir", type=str, required=True)
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", type=str, default="logistic_regression,random_forest,xgboost,mlp",
                        help="Comma-separated subset of: " + ",".join(BASELINES.keys()))
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Loading graph:  {args.graph_dir}")
    print(f"Loading splits: {args.split_dir}")
    X, y, cols, train_idx, val_idx, test_idx = load_split_data(args.graph_dir, args.split_dir)
    print(f"X: {X.shape}, y positives: {int(y.sum())}/{len(y)} "
          f"({100 * y.sum() / len(y):.3f}%)")
    print(f"train/val/test sizes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_val_raw, y_val = X[val_idx], y[val_idx]
    X_test_raw, y_test = X[test_idx], y[test_idx]

    scaler = fit_scaler(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    results = {
        "graph_dir": args.graph_dir,
        "split_dir": args.split_dir,
        "num_features": len(cols),
        "feature_cols": cols,
        "seed": args.seed,
        "models": {},
    }

    for name in requested:
        if name not in BASELINES:
            print(f"[WARN] Unknown model '{name}' -- skipping. "
                  f"Valid: {list(BASELINES.keys())}")
            continue

        print("\n" + "=" * 70)
        print(f"Training: {name}")
        print("=" * 70)
        t0 = time.time()
        try:
            val_probs, test_probs = BASELINES[name](X_train, y_train, X_val, X_test, args.seed)
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            results["models"][name] = {"error": str(e)}
            continue
        elapsed = time.time() - t0

        if val_probs is None:
            results["models"][name] = {"error": "dependency not installed"}
            continue

        val_metrics = evaluate_binary_classifier(y_val, val_probs, verbose=False)
        test_metrics = evaluate_binary_classifier(y_test, test_probs, verbose=False)

        print(f"[{name}] val AUPR={val_metrics['aupr']:.4f}  "
              f"test AUPR={test_metrics['aupr']:.4f}  "
              f"test ROC-AUC={test_metrics['roc_auc']:.4f}  "
              f"({elapsed:.1f}s)")

        results["models"][name] = {
            "train_time_sec": elapsed,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
