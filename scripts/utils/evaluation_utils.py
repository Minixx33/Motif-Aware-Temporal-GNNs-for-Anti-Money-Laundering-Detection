# evaluation_utils.py
# -----------------------------------------------------------
# Enhanced evaluation module for GraphSAGE, GraphSAGE-T, TGAT
# Works for node or edge-level binary classification.
# Optimized for highly imbalanced datasets (e.g., 0.10% positive).
# ASCII-safe version (no emojis, no non-ASCII characters).
# -----------------------------------------------------------

import numpy as np
import warnings
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score
)


def evaluate_binary_classifier(
    y_true,
    y_pred_probs,
    threshold=0.5,
    auto_threshold=False,
    compute_top_k=True,
    k_values=[100, 500, 1000],
    verbose=True
):
    """
    Unified evaluation utility for:
      - GraphSAGE (static)
      - GraphSAGE-T (temporal)
      - TGAT (temporal event model)

    Supports node or edge classification with severe class imbalance.
    """

    # ------------------------------------------------------------
    # INPUT VALIDATION
    # ------------------------------------------------------------

    y_true = np.asarray(y_true).flatten()

    if len(y_true) == 0:
        raise ValueError("y_true is empty")

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        if verbose:
            warnings.warn(
                "Only one class present in y_true. "
                "Some metrics will be undefined."
            )

    # ------------------------------------------------------------
    # SANITIZE PREDICTION SHAPES
    # ------------------------------------------------------------

    if y_pred_probs.ndim == 2 and y_pred_probs.shape[1] == 2:
        y_pred_probs = y_pred_probs[:, 1]
    elif y_pred_probs.ndim == 2 and y_pred_probs.shape[1] == 1:
        y_pred_probs = y_pred_probs[:, 0]

    y_pred_probs = np.asarray(y_pred_probs).flatten()

    if y_true.shape[0] != y_pred_probs.shape[0]:
        raise ValueError(
            "Shape mismatch: y_true {} vs y_pred_probs {}".format(
                y_true.shape, y_pred_probs.shape
            )
        )

    y_pred_probs = np.clip(y_pred_probs, 0.0, 1.0)

    # ------------------------------------------------------------
    # AUTO THRESHOLD (optional)
    # ------------------------------------------------------------

    if auto_threshold:
        best_thr = 0.5
        best_f1 = -1
        thr_grid = np.linspace(0.01, 0.99, 200)

        for thr in thr_grid:
            preds = (y_pred_probs >= thr).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr

        threshold = best_thr
        if verbose:
            print("[Auto-threshold] Best threshold = {:.3f} (F1={:.4f})".format(
                threshold, best_f1
            ))

    # Hard predictions
    y_pred = (y_pred_probs >= threshold).astype(int)

    # ------------------------------------------------------------
    # STANDARD METRICS
    # ------------------------------------------------------------

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        rocauc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        rocauc = float("nan")
        if verbose:
            warnings.warn("ROC-AUC could not be computed.")

    aupr = average_precision_score(y_true, y_pred_probs)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # ------------------------------------------------------------
    # IMBALANCED-AWARE METRICS
    # ------------------------------------------------------------

    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        tn = fp = fn = tp = 0
        specificity = 0.0
        if verbose:
            warnings.warn("Confusion matrix is not 2x2. Setting specificity to 0.")

    # ------------------------------------------------------------
    # TOP-K METRICS
    # ------------------------------------------------------------

    top_k_metrics = {}
    if compute_top_k:
        for k in k_values:
            if k <= len(y_true):
                top_k_idx = np.argsort(y_pred_probs)[-k:]
                precision_at_k = y_true[top_k_idx].sum() / k
                top_k_metrics["precision_at_{}".format(k)] = float(precision_at_k)

    # ------------------------------------------------------------
    # ASSEMBLE RESULTS
    # ------------------------------------------------------------

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(rocauc),
        "aupr": float(aupr),
        "balanced_accuracy": float(bal_acc),

        "mcc": float(mcc),
        "kappa": float(kappa),
        "specificity": float(specificity),

        "threshold": float(threshold),

        "confusion_matrix": cm.tolist(),
        "confusion_matrix_dict": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        },

        "top_k_metrics": top_k_metrics
    }


# ------------------------------------------------------------
# ASCII SAFE PRINTING
# ------------------------------------------------------------

def print_metrics(metrics, model_name="Model"):
    """
    ASCII-safe printing (no emojis, no unicode).
    """

    print("=" * 70)
    print("EVALUATION RESULTS: {}".format(model_name))
    print("=" * 70)

    print("\nSTANDARD METRICS:")
    print("  Precision:        {:.4f}".format(metrics["precision"]))
    print("  Recall:           {:.4f}".format(metrics["recall"]))
    print("  F1-Score:         {:.4f}".format(metrics["f1"]))
    print("  ROC-AUC:          {:.4f}".format(metrics["roc_auc"]))
    print("  AUPR:             {:.4f}".format(metrics["aupr"]))
    print("  Balanced Acc:     {:.4f}".format(metrics["balanced_accuracy"]))

    print("\nIMBALANCED-AWARE METRICS:")
    print("  MCC:              {:.4f}".format(metrics["mcc"]))
    print("  Cohen Kappa:      {:.4f}".format(metrics["kappa"]))
    print("  Specificity:      {:.4f}".format(metrics["specificity"]))

    print("\nTHRESHOLD: {:.3f}".format(metrics["threshold"]))

    cm = metrics["confusion_matrix_dict"]
    print("\nCONFUSION MATRIX:")
    print("  True Negatives:   {}".format(cm["TN"]))
    print("  False Positives:  {}".format(cm["FP"]))
    print("  False Negatives:  {}".format(cm["FN"]))
    print("  True Positives:   {}".format(cm["TP"]))

    if metrics["top_k_metrics"]:
        print("\nTOP-K PRECISION:")
        for k, value in metrics["top_k_metrics"].items():
            print("  {}: {:.4f}".format(k, value))

    print("=" * 70)
