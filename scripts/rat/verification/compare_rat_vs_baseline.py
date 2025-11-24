"""
plot_rat_distributions.py
Usage:
    python plot_rat_distributions.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets"
RAT_DIR  = os.path.join(BASE_DIR, "RAT")
PLOT_DIR = os.path.join(RAT_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

LABEL_COL = "Is Laundering"
INJECT_COL = "RAT_injected"

RAT_FILES = {
    "low":    "HI-Small_Trans_RAT_low.csv",
    "medium": "HI-Small_Trans_RAT_medium.csv",
    "high":   "HI-Small_Trans_RAT_high.csv",
}

# =====================================================================
#   Histogram for **Injected Rows Only**
# =====================================================================
def plot_histograms():
    plt.figure(figsize=(10, 6))
    
    for name, fname in RAT_FILES.items():
        path = os.path.join(RAT_DIR, fname)
        df = pd.read_csv(path, usecols=[LABEL_COL, INJECT_COL, "RAT_score"])
        
        # Only analyze laundering rows that are actually injected
        injected_scores = df.loc[
            (df[LABEL_COL] == 1) & (df[INJECT_COL] == 1),
            "RAT_score"
        ].dropna()

        if injected_scores.empty:
            print(f"[WARN] No injected rows found in {name}. Skipping.")
            continue

        plt.hist(injected_scores, bins=40, alpha=0.5, density=True, label=name)

    plt.xlabel("RAT_score (Injected Laundering Rows Only)")
    plt.ylabel("Density")
    plt.title("RAT_score Distribution by Intensity (Injected Rows Only)")
    plt.legend()
    
    out_path = os.path.join(PLOT_DIR, "rat_score_hist_injected_only.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved histogram to {out_path}")

# =====================================================================
#   CDF for **Injected Rows Only**
# =====================================================================
def plot_cdf():
    plt.figure(figsize=(10, 6))

    for name, fname in RAT_FILES.items():
        path = os.path.join(RAT_DIR, fname)
        df = pd.read_csv(path, usecols=[LABEL_COL, INJECT_COL, "RAT_score"])
        
        injected_scores = df.loc[
            (df[LABEL_COL] == 1) & (df[INJECT_COL] == 1),
            "RAT_score"
        ].dropna()

        if injected_scores.empty:
            print(f"[WARN] No injected rows found in {name}. Skipping.")
            continue

        ls_sorted = np.sort(injected_scores.values)
        y = np.linspace(0, 1, len(ls_sorted))

        plt.plot(ls_sorted, y, label=name)

    plt.xlabel("RAT_score (Injected Laundering Rows Only)")
    plt.ylabel("CDF")
    plt.title("RAT_score CDF by Intensity (Injected Rows Only)")
    plt.legend()

    out_path = os.path.join(PLOT_DIR, "rat_score_cdf_injected_only.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved CDF plot to {out_path}")

# =====================================================================

def main():
    plot_histograms()
    plot_cdf()

if __name__ == "__main__":
    main()
