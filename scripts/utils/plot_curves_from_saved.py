import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score
)
import os

# ============================================================
# CONFIGURE PATHS (edit only this part)
# ============================================================
RESULTS_DIR = r"results/HI-Small_Trans/seed2025_experiment_3/graphsage"   # <-- change per run
GRAPH_DIR   = r"graphs/HI-Small_Trans"                   # static/dataset folder
SPLIT_DIR   = r"splits/HI-Small_Trans"

# ============================================================
# LOAD TRUE LABELS
# ============================================================
y_edge = torch.load(os.path.join(GRAPH_DIR, "y_edge.pt")).numpy()
test_idx = torch.load(os.path.join(SPLIT_DIR, "test_edge_idx.pt")).numpy()

y_true = y_edge[test_idx]

# ============================================================
# LOAD PREDICTED PROBABILITIES
# ============================================================
test_probs = torch.load(os.path.join(RESULTS_DIR, "test_pred_probs.pt")).numpy()

# Ensure shape is (N,)
test_probs = test_probs.reshape(-1)

print("Loaded:")
print("  y_true =", y_true.shape)
print("  test_probs =", test_probs.shape)

# ============================================================
# ROC CURVE
# ============================================================
fpr, tpr, _ = roc_curve(y_true, test_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=300)

# ============================================================
# PRECISION–RECALL CURVE (AUPR)
# ============================================================
precision, recall, _ = precision_recall_curve(y_true, test_probs)
ap = average_precision_score(y_true, test_probs)

plt.figure(figsize=(6,5))
plt.plot(recall, precision, label=f"AUPR = {ap:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pr_curve.png"), dpi=300)

print("\nSaved:")
print(" -", os.path.join(RESULTS_DIR, "roc_curve.png"))
print(" -", os.path.join(RESULTS_DIR, "pr_curve.png"))
