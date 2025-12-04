import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
import matplotlib.pyplot as plt
import os

# -------------------------------------------------------
# DyRep Paths — CHANGE THESE
# -------------------------------------------------------

results_dir = r"results/HI-Small_Trans/seed1337_experiment2/dyrep"   # <-- change per run
graph_dir   = r"graphs_dyrep/HI-Small_Trans"                   # static/dataset folder
split_dir   = r"splits_dyrep/HI-Small_Trans"

# -------------------------------------------------------
# Load predicted probabilities (TEST only)
# -------------------------------------------------------

probs = torch.load(os.path.join(results_dir, "test_pred_probs.pt")).cpu().numpy().ravel()

# -------------------------------------------------------
# Load ground truth DYREP labels (labels.pt)
# -------------------------------------------------------

y_all = torch.load(os.path.join(graph_dir, "labels.pt")).cpu().numpy().astype(int)

# -------------------------------------------------------
# Load test split indices
# -------------------------------------------------------

# Try the two common naming schemes
if os.path.exists(os.path.join(split_dir, "test_edge_idx.pt")):
    test_idx = torch.load(os.path.join(split_dir, "test_edge_idx.pt")).cpu().numpy()
elif os.path.exists(os.path.join(split_dir, "test_events_idx.pt")):
    test_idx = torch.load(os.path.join(split_dir, "test_events_idx.pt")).cpu().numpy()
else:
    raise FileNotFoundError("Could not find test split file.")

y_true = y_all[test_idx]

print("Shapes:", probs.shape, y_true.shape)
print("Positives in test:", y_true.sum(), "/", len(y_true))

# -------------------------------------------------------
# Compute PR curve
# -------------------------------------------------------

precision, recall, _ = precision_recall_curve(y_true, probs)
aupr = average_precision_score(y_true, probs)

plt.figure(figsize=(7,5))
plt.plot(recall, precision, label=f"AUPR = {aupr:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("DyRep Precision–Recall Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "pr_curve.png"), dpi=300)

# -------------------------------------------------------
# Compute ROC curve
# -------------------------------------------------------

fpr, tpr, _ = roc_curve(y_true, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("DyRep ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "roc_curve.png"), dpi=300)
