"""
generate_all_plots.py
Generates:
1) RAT score plots (hist + CDF)
2) Motif feature distributions
3) RAT feature distributions
4) Injected vs non-injected comparisons
5) Correlation heatmaps
6) Intensity drift plots

USAGE:
    python generate_all_plots.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
BASE_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets"
RAT_DIR  = os.path.join(BASE_DIR, "RAT")
PLOT_DIR = os.path.join(RAT_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

RAT_FILES = {
    "low":    "HI-Small_Trans_RAT_low.csv",
    "medium": "HI-Small_Trans_RAT_medium.csv",
    "high":   "HI-Small_Trans_RAT_high.csv",
}

LABEL = "Is Laundering"
INJECT = "RAT_injected"

# RAT-related feature names
RAT_FEATURES = [
    "RAT_offender_score", "RAT_target_score", "RAT_guardian_weakness_score",
    "RAT_score", "RAT_is_off_hours", "RAT_is_weekend", "RAT_is_cross_bank",
    "RAT_src_amount_z_pos", "RAT_dst_amount_z_pos",
    "RAT_src_out_deg_norm", "RAT_dst_in_deg_norm",
    "RAT_src_burst_norm", "RAT_dst_burst_norm", "RAT_combined_burst",
    "RAT_same_entity", "RAT_mutual_flag"
]

MOTIF_FEATURES = [
    "motif_fanin", "motif_fanout", "motif_chain", "motif_cycle"
]

def load_df(name, columns):
    """Load only necessary columns, reducing memory footprint."""
    path = os.path.join(RAT_DIR, RAT_FILES[name])
    return pd.read_csv(path, usecols=columns)

# ------------------------------------------------------------
# 1. RAT SCORE HISTOGRAM + CDF
# ------------------------------------------------------------
def plot_rat_scores():
    outdir = os.path.join(PLOT_DIR, "rat_score_distributions")
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(10,6))
    for name in RAT_FILES:
        df = load_df(name, [LABEL, INJECT, "RAT_score"])
        scores = df.loc[(df[LABEL]==1)&(df[INJECT]==1), "RAT_score"]

        plt.hist(scores, bins=40, alpha=0.5, density=True, label=name)

    plt.xlabel("RAT_score (Injected Only)")
    plt.ylabel("Density")
    plt.legend()
    plt.title("RAT_score Histogram")
    plt.savefig(os.path.join(outdir, "hist_rat_score.png"), dpi=200)
    plt.close()

    # ----- CDF -----
    plt.figure(figsize=(10,6))
    for name in RAT_FILES:
        df = load_df(name, [LABEL, INJECT, "RAT_score"])
        scores = df.loc[(df[LABEL]==1)&(df[INJECT]==1), "RAT_score"]
        s = np.sort(scores)
        y = np.linspace(0,1,len(s))
        plt.plot(s, y, label=name)

    plt.xlabel("RAT_score")
    plt.ylabel("CDF")
    plt.legend()
    plt.title("RAT_score CDF")
    plt.savefig(os.path.join(outdir, "cdf_rat_score.png"), dpi=200)
    plt.close()


# ------------------------------------------------------------
# 2. FEATURE DISTRIBUTIONS (RAT + MOTIF)
# ------------------------------------------------------------
def plot_feature_distributions():
    outdir = os.path.join(PLOT_DIR, "feature_distributions")
    os.makedirs(outdir, exist_ok=True)

    # Only load from HIGH intensity (largest variation)
    df = load_df("high", [LABEL, INJECT] + RAT_FEATURES + MOTIF_FEATURES)
    df = df.loc[(df[LABEL]==1) & (df[INJECT]==1)]

    for feat in RAT_FEATURES + MOTIF_FEATURES:
        plt.figure(figsize=(8,5))
        plt.hist(df[feat].dropna(), bins=40, density=True, alpha=0.7)
        plt.title(f"Distribution of {feat}")
        plt.xlabel(feat)
        plt.ylabel("Density")
        plt.savefig(os.path.join(outdir, f"{feat}.png"), dpi=200)
        plt.close()


# ------------------------------------------------------------
# 3. CORRELATION HEATMAP
# ------------------------------------------------------------
def plot_correlation_heatmap():
    outdir = os.path.join(PLOT_DIR, "feature_correlations")
    os.makedirs(outdir, exist_ok=True)

    df = load_df("high", [LABEL, INJECT] + RAT_FEATURES + MOTIF_FEATURES)
    df = df.loc[(df[LABEL]==1)&(df[INJECT]==1)]

    corr = df[RAT_FEATURES + MOTIF_FEATURES].corr()

    plt.figure(figsize=(14,12))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap (Injected Laundering Rows)")
    plt.savefig(os.path.join(outdir, "correlations.png"), dpi=200)
    plt.close()


# ------------------------------------------------------------
# 4. INTENSITY DRIFT PLOTS
#     This shows how each feature increases from low→medium→high
# ------------------------------------------------------------
def plot_intensity_drift():
    outdir = os.path.join(PLOT_DIR, "intensity_comparisons")
    os.makedirs(outdir, exist_ok=True)

    stats = {}

    for name in RAT_FILES:
        df = load_df(name, [LABEL, INJECT] + RAT_FEATURES + MOTIF_FEATURES)
        df = df.loc[(df[LABEL]==1)&(df[INJECT]==1)]
        stats[name] = df.mean()

    stats_df = pd.DataFrame(stats)

    # RAT features drift
    for feat in RAT_FEATURES + MOTIF_FEATURES:
        plt.figure(figsize=(6,4))
        vals = [stats_df.loc[feat, "low"],
                stats_df.loc[feat, "medium"],
                stats_df.loc[feat, "high"]]

        plt.plot(["low","medium","high"], vals, marker="o")
        plt.title(f"Intensity Drift: {feat}")
        plt.ylabel(f"{feat} (mean)")
        plt.savefig(os.path.join(outdir, f"drift_{feat}.png"), dpi=200)
        plt.close()


# ------------------------------------------------------------
# RUN EVERYTHING
# ------------------------------------------------------------
def main():
    print("Generating all plots...")
    plot_rat_scores()
    plot_feature_distributions()
    plot_correlation_heatmap()
    plot_intensity_drift()
    print("\nAll plots saved to:", PLOT_DIR)

if __name__ == "__main__":
    main()
