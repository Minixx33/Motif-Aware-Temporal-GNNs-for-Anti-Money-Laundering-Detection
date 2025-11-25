# evaluation_utils.py
# -----------------------------------------------------------
# Enhanced evaluation module for GraphSAGE, GraphSAGE-T, TGAT
# Works for node or edge-level binary classification.
# Optimized for highly imbalanced datasets (e.g., 0.10% positive).
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


def evaluate_binary_classifier(y_true, y_pred_probs, threshold=0.5, auto_threshold=False,
                                compute_top_k=True, k_values=[100, 500, 1000], verbose=True):
    """
    Unified evaluation utility for:
      - GraphSAGE (static)
      - GraphSAGE-T (temporal)
      - TGAT (temporal event model)
    
    Supports node or edge classification with severe class imbalance.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (0/1). Shape: (N,)

    y_pred_probs : np.ndarray
        Predicted probabilities for class 1.
        Shape can be (N,), (N, 1), or (N, 2) for softmax outputs.

    threshold : float, default=0.5
        Classification threshold for converting probabilities to binary predictions.

    auto_threshold : bool, default=False
        If True, searches for the best F1 threshold in range [0.01, 0.99].

    compute_top_k : bool, default=True
        If True, compute precision at top-K predictions (useful for imbalanced data).

    k_values : list, default=[100, 500, 1000]
        K values for top-K precision metrics.

    verbose : bool, default=True
        If True, print warnings and threshold info.

    Returns
    -------
    dict
        Comprehensive metrics including:
        - Standard: precision, recall, f1, roc_auc, aupr, balanced_accuracy
        - Imbalanced-aware: mcc, kappa, specificity
        - Top-K: precision_at_k for specified k values
        - Confusion matrix (as list and dict)
    """

    # ============================================================
    # INPUT VALIDATION
    # ============================================================
    
    y_true = np.asarray(y_true).flatten()
    
    if len(y_true) == 0:
        raise ValueError("y_true is empty")
    
    # Check for single class
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        if verbose:
            warnings.warn(
                f"Only one class present in y_true: {unique_classes}. "
                "Some metrics will be undefined."
            )
    
    # ============================================================
    # SANITIZE PREDICTION SHAPES
    # ============================================================
    
    if y_pred_probs.ndim == 2 and y_pred_probs.shape[1] == 2:
        # Softmax outputs → extract probability of class 1
        y_pred_probs = y_pred_probs[:, 1]
    elif y_pred_probs.ndim == 2 and y_pred_probs.shape[1] == 1:
        # Squeeze (N, 1) → (N,)
        y_pred_probs = y_pred_probs[:, 0]
    
    y_pred_probs = np.asarray(y_pred_probs).flatten()
    
    # Validate shapes match
    if y_true.shape[0] != y_pred_probs.shape[0]:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred_probs {y_pred_probs.shape}"
        )
    
    # Clip probabilities to valid range [0, 1]
    y_pred_probs = np.clip(y_pred_probs, 0.0, 1.0)
    
    # ============================================================
    # OPTIONAL THRESHOLD OPTIMIZATION (AUTO-THRESHOLD)
    # ============================================================
    
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
            print(f"[Auto-threshold] Best threshold for F1 = {threshold:.3f} (F1={best_f1:.4f})")

    # Final hard predictions
    y_pred = (y_pred_probs >= threshold).astype(int)

    # ============================================================
    # COMPUTE STANDARD METRICS
    # ============================================================
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        rocauc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        rocauc = float("nan")
        if verbose:
            warnings.warn("ROC-AUC could not be computed (only one class in y_true?)")

    aupr = average_precision_score(y_true, y_pred_probs)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # ============================================================
    # COMPUTE IMBALANCED-AWARE METRICS
    # ============================================================
    
    mcc = matthews_corrcoef(y_true, y_pred)  # Matthews Correlation Coefficient
    kappa = cohen_kappa_score(y_true, y_pred)  # Cohen's Kappa

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Extract confusion matrix components
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # Handle edge case where only one class predicted
        tn = fp = fn = tp = 0
        specificity = 0.0
        if verbose:
            warnings.warn("Confusion matrix is not 2x2. Setting specificity to 0.")

    # ============================================================
    # COMPUTE TOP-K PRECISION (FOR IMBALANCED DATA)
    # ============================================================
    
    top_k_metrics = {}
    if compute_top_k:
        for k in k_values:
            if k <= len(y_true):
                # Get indices of top-k predictions
                top_k_idx = np.argsort(y_pred_probs)[-k:]
                precision_at_k = y_true[top_k_idx].sum() / k
                top_k_metrics[f"precision_at_{k}"] = float(precision_at_k)

    # ============================================================
    # ASSEMBLE RESULTS
    # ============================================================
    
    results = {
        # Standard metrics
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(rocauc),
        "aupr": float(aupr),
        "balanced_accuracy": float(bal_acc),
        
        # Imbalanced-aware metrics
        "mcc": float(mcc),
        "kappa": float(kappa),
        "specificity": float(specificity),
        
        # Threshold
        "threshold": float(threshold),
        
        # Confusion matrix
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_dict": {
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp)
        },
        
        # Top-K metrics
        "top_k_metrics": top_k_metrics
    }
    
    return results


def print_metrics(metrics, model_name="Model"):
    """
    Pretty-print evaluation metrics.
    
    Parameters
    ----------
    metrics : dict
        Output from evaluate_binary_classifier()
    model_name : str
        Name of the model being evaluated
    """
    print("=" * 70)
    print(f"EVALUATION RESULTS: {model_name}")
    print("=" * 70)
    
    print("\n📊 STANDARD METRICS:")
    print(f"  Precision:        {metrics['precision']:.4f}")
    print(f"  Recall:           {metrics['recall']:.4f}")
    print(f"  F1-Score:         {metrics['f1']:.4f}")
    print(f"  ROC-AUC:          {metrics['roc_auc']:.4f}")
    print(f"  AUPR:             {metrics['aupr']:.4f}")
    print(f"  Balanced Acc:     {metrics['balanced_accuracy']:.4f}")
    
    print("\n🎯 IMBALANCED-AWARE METRICS:")
    print(f"  MCC:              {metrics['mcc']:.4f}")
    print(f"  Cohen's Kappa:    {metrics['kappa']:.4f}")
    print(f"  Specificity:      {metrics['specificity']:.4f}")
    
    print(f"\n⚙️  THRESHOLD: {metrics['threshold']:.3f}")
    
    cm_dict = metrics['confusion_matrix_dict']
    print("\n📋 CONFUSION MATRIX:")
    print(f"  True Negatives:   {cm_dict['TN']:,}")
    print(f"  False Positives:  {cm_dict['FP']:,}")
    print(f"  False Negatives:  {cm_dict['FN']:,}")
    print(f"  True Positives:   {cm_dict['TP']:,}")
    
    if metrics['top_k_metrics']:
        print("\n🔝 TOP-K PRECISION:")
        for k, prec in metrics['top_k_metrics'].items():
            k_val = k.split('_')[-1]
            print(f"  Precision @ {k_val:>4}: {prec:.4f}")
    
    print("=" * 70)


def compare_models(results_dict, save_path=None):
    """
    Compare multiple model results side-by-side.
    
    Parameters
    ----------
    results_dict : dict
        {"model_name": metrics_dict, ...}
    save_path : str, optional
        Path to save comparison table as CSV
    
    Returns
    -------
    pd.DataFrame
        Comparison table with all metrics
    """
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas required for compare_models(). Install with: pip install pandas")
        return None
    
    comparison = {}
    for model_name, metrics in results_dict.items():
        comparison[model_name] = {
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'ROC-AUC': metrics['roc_auc'],
            'AUPR': metrics['aupr'],
            'Balanced Acc': metrics['balanced_accuracy'],
            'MCC': metrics['mcc'],
            'Kappa': metrics['kappa'],
            'Specificity': metrics['specificity'],
            'Threshold': metrics['threshold'],
            'TP': metrics['confusion_matrix_dict']['TP'],
            'FP': metrics['confusion_matrix_dict']['FP'],
            'FN': metrics['confusion_matrix_dict']['FN'],
            'TN': metrics['confusion_matrix_dict']['TN'],
        }
    
    df = pd.DataFrame(comparison).T
    
    if save_path:
        df.to_csv(save_path)
        print(f"✓ Comparison table saved to: {save_path}")
    
    return df


def plot_metrics(metrics, model_name="Model", save_path=None):
    """
    Plot evaluation metrics for easy interpretation.
    
    Parameters
    ----------
    metrics : dict
        Output from evaluate_binary_classifier()
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib required for plot_metrics(). Install with: pip install matplotlib")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Evaluation: {model_name}", fontsize=16, fontweight='bold')
    
    # Plot 1: Main metrics
    metric_names = ['Precision', 'Recall', 'F1', 'ROC-AUC', 'AUPR', 'Bal. Acc', 'MCC']
    metric_keys = ['precision', 'recall', 'f1', 'roc_auc', 'aupr', 'balanced_accuracy', 'mcc']
    metric_values = [metrics[k] for k in metric_keys]
    
    colors = ['steelblue' if v >= 0 else 'coral' for v in metric_values]
    
    axes[0].bar(range(len(metric_names)), metric_values, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_xticks(range(len(metric_names)))
    axes[0].set_xticklabels(metric_names, rotation=45, ha='right')
    axes[0].set_ylim(-1, 1)
    axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].set_ylabel('Score')
    axes[0].set_title('Evaluation Metrics')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(metric_values):
        axes[0].text(i, v + 0.02 if v >= 0 else v - 0.05, f'{v:.3f}',
                    ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    
    # Plot 2: Confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    im = axes[1].imshow(cm, cmap='Blues', aspect='auto')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['Predicted 0', 'Predicted 1'])
    axes[1].set_yticklabels(['Actual 0', 'Actual 1'])
    axes[1].set_title('Confusion Matrix')
    
    # Add text annotations
    for i in range(min(2, cm.shape[0])):
        for j in range(min(2, cm.shape[1])):
            text = axes[1].text(j, i, f'{cm[i, j]:,}',
                              ha="center", va="center", 
                              color="white" if cm[i, j] > cm.max()/2 else "black",
                              fontweight='bold')
    
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Metrics plot saved to: {save_path}")
    
    plt.show()


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Example with synthetic imbalanced data
    np.random.seed(42)
    
    # Simulate 0.10% positive class (like your dataset)
    n_samples = 10000
    n_positive = int(0.001 * n_samples)  # 10 positive samples
    
    y_true = np.array([1] * n_positive + [0] * (n_samples - n_positive))
    
    # Simulate predictions (model struggling with imbalance)
    y_pred_probs = np.random.beta(0.5, 10, n_samples)  # Skewed toward 0
    y_pred_probs[y_true == 1] += 0.3  # Give true positives higher scores
    y_pred_probs = np.clip(y_pred_probs, 0, 1)
    
    # Evaluate
    metrics = evaluate_binary_classifier(
        y_true, 
        y_pred_probs, 
        threshold=0.5,
        auto_threshold=True,
        compute_top_k=True,
        k_values=[10, 50, 100]
    )
    
    # Print results
    print_metrics(metrics, model_name="Example Model")
    
    # Plot (requires matplotlib)
    try:
        plot_metrics(metrics, model_name="Example Model")
    except:
        print("\nmatplotlib not available - skipping plot")