"""
dataset_diagnostics_report.py
-----------------------------
Generates high-quality diagnostic plots comparing the baseline and RAT-injected AML datasets.
All figures are exported in both PNG for inclusion in reports/papers.

Outputs (saved under Datasets/baseline_RAT/plots/):
  - transaction_amount_distribution.png
  - motif_participation_distribution.png
  - motif_ratio_SAR_vs_Normal.png
  - motif_ratio_SAR_vs_Normal_log_scaled.png
  - dataset_overview.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# === STYLE CONFIGURATION ===
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
})

# === PATHS ===
BASELINE_DIR = r"Datasets\\baseline"
RAT_DIR = r"Datasets\\baseline_RAT"
OUTPUT_DIR = os.path.join(RAT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
print("Loading baseline and RAT-injected data...")
tx_base = pd.read_csv(os.path.join(BASELINE_DIR, "transactions.csv"), low_memory=False)
tx_rat = pd.read_csv(os.path.join(RAT_DIR, "transactions.csv"), low_memory=False)
feat = pd.read_csv(os.path.join(RAT_DIR, "motif_features_labeled.csv"))

# === COLORS ===
BLUE = "#4B8BBE"
ORANGE = "#FF9900"
GREY = "#666666"

def save_plot(name):
    """Save figure as PNG"""
    out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_path}")


# === 1. TRANSACTION AMOUNT DISTRIBUTION COMPARISON ===
plt.figure(figsize=(7,5))
plt.hist(tx_base["base_amt"], bins=100, alpha=0.7, label="Baseline", color=BLUE)
plt.hist(tx_rat["base_amt"], bins=100, alpha=0.7, label="RAT-injected", color=ORANGE)
plt.title("Transaction Amount Distribution (Baseline vs RAT-injected)")
plt.xlabel("Transaction Amount")
plt.ylabel("Number of Transactions")
plt.legend()
plt.figtext(0.5, -0.05,
            "Figure 1. Distribution of transaction amounts across baseline and RAT-injected datasets. "
            "The RAT-injected set preserves natural skew while adding structured patterns.",
            wrap=True, ha="center", fontsize=10, color=GREY)
save_plot("transaction_amount_distribution")

# === 2. MOTIF PARTICIPATION DISTRIBUTION ===
plt.figure(figsize=(7,5))
plt.hist(feat["motif_participation"], bins=50, alpha=0.8, color=BLUE)
plt.title("Motif Participation per Account (RAT-injected Dataset)")
plt.xlabel("Number of Motif Edges")
plt.ylabel("Count of Accounts")
plt.figtext(0.5, -0.05,
            "Figure 2. Distribution of motif participation per account. "
            "Only a small subset of accounts engage in motif structures (RAT-related activity).",
            wrap=True, ha="center", fontsize=10, color=GREY)
save_plot("motif_participation_distribution")

# === 3. MOTIF RATIO: SAR VS NORMAL ===
plt.figure(figsize=(7,5))
plt.hist(feat[feat["is_sar_account"]==0]["motif_ratio"],
         bins=np.linspace(0,0.02,100), alpha=0.7, label="Normal", color=BLUE)
plt.hist(feat[feat["is_sar_account"]==1]["motif_ratio"],
         bins=np.linspace(0,0.02,100), alpha=0.7, label="SAR", color=ORANGE)
plt.legend()
plt.title("Motif Ratio: SAR vs Normal (zoomed)")
plt.xlabel("Motif Ratio")
plt.ylabel("Number of Accounts")
plt.figtext(0.5, -0.05,
            "Figure 3. Comparison of motif ratios between SAR and normal accounts. "
            "SAR accounts show slightly higher motif involvement.",
            wrap=True, ha="center", fontsize=10, color=GREY)
save_plot("motif_ratio_SAR_vs_Normal")

# === 4. MOTIF RATIO: SAR VS NORMAL (LOG-SCALED) ===
plt.figure(figsize=(7,5))
plt.hist(feat[feat["is_sar_account"]==0]["motif_ratio"],
         bins=np.linspace(0,0.005,200), alpha=0.6, label="Normal", color=BLUE)
plt.hist(feat[feat["is_sar_account"]==1]["motif_ratio"],
         bins=np.linspace(0,0.005,200), alpha=0.6, label="SAR", color=ORANGE)
plt.yscale("log")
plt.legend()
plt.title("Motif Ratio: SAR vs Normal (log-scaled)")
plt.xlabel("Motif Ratio")
plt.ylabel("Number of Accounts (log scale)")
plt.figtext(0.5, -0.05,
            "Figure 4. Log-scaled version revealing rare high-motif SAR accounts in the tail.",
            wrap=True, ha="center", fontsize=10, color=GREY)
save_plot("motif_ratio_SAR_vs_Normal_log_scaled")

# Plot interpretation
# 1. The log scale reveals that while nearly all accounts have a motif ratio near zero,
# there’s a clear sparse tail of SAR (orange) accounts extending up to ~0.005.
# 2. Those are your RAT-injected laundering entities — rare, but structurally distinct.
# 3. The Normal (blue) distribution drops off almost immediately, confirming that motif participation is a strong differentiator.

# This shows that your motif features (especially motif_ratio, fanin_as_dst, etc.) 
# actually encode criminologically meaningful patterns — a great validation before 
# you move on to your Temporal GNN training.

# === 5. SUMMARY DASHBOARD (3×1 GRID) ===
fig, axes = plt.subplots(3, 1, figsize=(7, 12))
axes = axes.flatten()

# (a) Transaction amounts
axes[0].hist(tx_base["base_amt"], bins=100, alpha=0.7, label="Baseline", color=BLUE)
axes[0].hist(tx_rat["base_amt"], bins=100, alpha=0.7, label="RAT-injected", color=ORANGE)
axes[0].set_title("(a) Transaction Amount Distribution")
axes[0].set_xlabel("Transaction Amount")
axes[0].set_ylabel("Count")
axes[0].legend()

# (b) Motif participation
axes[1].hist(feat["motif_participation"], bins=50, alpha=0.8, color=BLUE)
axes[1].set_title("(b) Motif Participation per Account")
axes[1].set_xlabel("Number of Motif Edges")
axes[1].set_ylabel("Accounts")

# (c) Motif ratio comparison (log)
axes[2].hist(feat[feat["is_sar_account"]==0]["motif_ratio"],
             bins=np.linspace(0,0.005,200), alpha=0.6, label="Normal", color=BLUE)
axes[2].hist(feat[feat["is_sar_account"]==1]["motif_ratio"],
             bins=np.linspace(0,0.005,200), alpha=0.6, label="SAR", color=ORANGE)
axes[2].set_yscale("log")
axes[2].set_title("(c) Motif Ratio: SAR vs Normal (log-scaled)")
axes[2].set_xlabel("Motif Ratio")
axes[2].set_ylabel("Accounts (log scale)")
axes[2].legend()

fig.tight_layout(pad=3.0)
fig.subplots_adjust(bottom=0.05)
fig.text(0.5, 0.01,
         "Figure 5. Summary dashboard combining key diagnostics of the baseline and RAT-injected datasets.",
         ha="center", fontsize=11, color=GREY)
save_plot("dataset_overview")

print(f"\n✅ All figures saved under: {OUTPUT_DIR}")


