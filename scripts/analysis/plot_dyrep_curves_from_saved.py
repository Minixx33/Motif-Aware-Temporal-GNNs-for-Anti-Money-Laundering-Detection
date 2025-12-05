import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
import matplotlib.pyplot as plt
import os

# -------------------------------------------------------
# DyRep Paths — CHANGE THESE
# -------------------------------------------------------

results_dir = r"results/HI-Small_Trans_RAT_high/seed2025_experiment_3/dyrep"
graph_dir   = r"graphs_dyrep/HI-Small_Trans_RAT_high"
split_dir   = r"splits_dyrep/HI-Small_Trans_RAT_high"

# -------------------------------------------------------
# Load predicted probabilities (TEST only)
# -------------------------------------------------------

probs = torch.load(os.path.join(results_dir, "test_pred_probs.pt")).cpu().numpy().ravel()

# Replace NaNs/Infs safely
if not np.isfinite(probs).all():
    print("⚠ WARNING: probs contained NaN/Inf. Fixing...")
    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

# -------------------------------------------------------
# Load ground truth DYREP labels
# -------------------------------------------------------

y_all = torch.load(os.path.join(graph_dir, "labels.pt")).cpu().numpy().astype(int)

# -------------------------------------------------------
# Load test split indices
# -------------------------------------------------------

test_idx = None

candidate_split_files = [
    "test_edge_idx.pt",
    "test_edges_idx.pt",
    "test_event_idx.pt",
    "test_events_idx.pt",
]

for fname in candidate_split_files:
    fpath = os.path.join(split_dir, fname)
    if os.path.exists(fpath):
        print(f"✓ Found split file: {fname}")
        test_idx = torch.load(fpath).cpu().numpy()
        break

if test_idx is None:
    raise FileNotFoundError("No test split index file found in split_dir.")

y_true = y_all[test_idx]

print("Loaded shapes (probs, y_true):", probs.shape, y_true.shape)
print("Positives in test:", y_true.sum(), "/", len(y_true))

# -------------------------------------------------------
# FIX LENGTH MISMATCH (DyRep commonly off by small margin)
# -------------------------------------------------------

if len(probs) != len(y_true):
    print(f"⚠ WARNING: Length mismatch! probs={len(probs)}, labels={len(y_true)}")
    min_len = min(len(probs), len(y_true))
    print(f"→ Trimming both to {min_len}")
    probs = probs[:min_len]
    y_true = y_true[:min_len]

# Final safety check
assert len(probs) == len(y_true), "Lengths still do not match after trimming!"

# -------------------------------------------------------
# Compute PR curve
# -------------------------------------------------------

precision, recall, _ = precision_recall_curve(y_true, probs)
aupr = average_precision_score(y_true, probs)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f"AUPR = {aupr:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("DyRep Precision–Recall Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "pr_curve.png"), dpi=300)

print(f"✓ Saved PR curve → {os.path.join(results_dir, 'pr_curve.png')}")

# -------------------------------------------------------
# Compute ROC curve
# -------------------------------------------------------

fpr, tpr, _ = roc_curve(y_true, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("DyRep ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "roc_curve.png"), dpi=300)

print(f"✓ Saved ROC curve → {os.path.join(results_dir, 'roc_curve.png')}")
