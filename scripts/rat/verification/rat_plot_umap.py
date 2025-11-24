"""
umap_per_intensity.py

Generates multiple UMAP visualizations for RAT+motif-enriched IBM HI-Small:

1) UMAP per intensity (low / medium / high)
   - Only injected laundering rows
   - Uses a HYBRID feature set: all RAT_* and motif_* columns
     (except RAT_injected and RAT_intensity_level)

2) UMAP baseline vs injected per intensity
   - Laundering rows only
   - Colors: injected vs non-injected

3) UMAPs per motif type (using HIGH intensity injected rows)
   - Features: motif_fanin, motif_fanout, motif_chain, motif_cycle
   - Colored by each motif value

4) Optional cluster metrics (if scikit-learn is available)
   - Silhouette score
   - Calinski–Harabasz index
   - Davies–Bouldin index

Outputs are saved under:
    ibm_transcations_datasets/RAT/plots_umap/
"""

import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt

# Optional: clustering quality metrics
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
BASE_DIR = r"C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\ibm_transcations_datasets"
RAT_DIR = os.path.join(BASE_DIR, "RAT")
PLOT_DIR = os.path.join(RAT_DIR, "plots_umap")

os.makedirs(PLOT_DIR, exist_ok=True)

RAT_FILES = {
    "low": "HI-Small_Trans_RAT_low.csv",
    "medium": "HI-Small_Trans_RAT_medium.csv",
    "high": "HI-Small_Trans_RAT_high.csv",
}

LABEL_COL = "Is Laundering"
INJECT_COL = "RAT_injected"
INTENSITY_COL = "RAT_intensity_level"

SAMPLES = 2000  # adaptive sampling will apply

CLUSTER_SCORES_PATH = os.path.join(PLOT_DIR, "cluster_scores.txt")
if os.path.exists(CLUSTER_SCORES_PATH):
    os.remove(CLUSTER_SCORES_PATH)


# ---------------------------------------------------------------------
# FEATURE SELECTION HELPERS (HYBRID = all RAT_* and motif_* columns)
# ---------------------------------------------------------------------
def discover_feature_cols(sample_df: pd.DataFrame):
    """
    Discover hybrid feature set:
        - all columns starting with "RAT_" or "motif_"
        - excluding label-like columns (RAT_injected, RAT_intensity_level)
        - numeric only
    """
    exclude = {INJECT_COL, INTENSITY_COL}
    feature_cols = []
    for col in sample_df.columns:
        if col in exclude:
            continue
        if col.startswith("RAT_") or col.startswith("motif_"):
            if np.issubdtype(sample_df[col].dtype, np.number):
                feature_cols.append(col)
    feature_cols = sorted(feature_cols)
    print(f"Discovered {len(feature_cols)} hybrid feature columns.")
    return feature_cols


# ---------------------------------------------------------------------
# CLUSTER METRICS
# ---------------------------------------------------------------------
def compute_cluster_metrics(X: np.ndarray, name: str):
    """Compute cluster quality metrics (if sklearn is available)."""
    if not SKLEARN_AVAILABLE:
        return

    n_samples = X.shape[0]
    if n_samples < 10:
        return

    # Choose up to 4 clusters but not more than n_samples-1
    k = min(4, n_samples - 1)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)

    with open(CLUSTER_SCORES_PATH, "a", encoding="utf-8") as f:
        f.write(f"{name}:\n")
        f.write(f"  samples={n_samples}, k={k}\n")
        f.write(f"  silhouette_score={sil:.4f}\n")
        f.write(f"  calinski_harabasz_score={ch:.4f}\n")
        f.write(f"  davies_bouldin_score={db:.4f}\n")
        f.write("\n")


# ---------------------------------------------------------------------
# UMAP HELPERS
# ---------------------------------------------------------------------
def run_umap(X: np.ndarray, n_neighbors=25, min_dist=0.3):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=42,
    )
    emb = reducer.fit_transform(X)
    return emb


def plot_scatter(
    x,
    y,
    title: str,
    out_path: str,
    color="royalblue",
    labels=None,
    cmap=None,
    colorbar_label=None,
):
    plt.figure(figsize=(12, 8))

    if labels is None:
        plt.scatter(x, y, s=10, alpha=0.7, color=color)
    else:
        sc = plt.scatter(x, y, s=10, alpha=0.8, c=labels, cmap=cmap)
        if colorbar_label is not None:
            cb = plt.colorbar(sc)
            cb.set_label(colorbar_label)

    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out_path}")


# ---------------------------------------------------------------------
# DATA LOADING HELPERS
# ---------------------------------------------------------------------
def load_intensity_df(intensity: str, injected_only=True, laundering_only=True):
    """Load one intensity dataset with optional laundering/injected filters."""
    path = os.path.join(RAT_DIR, RAT_FILES[intensity])
    df = pd.read_csv(path)

    if laundering_only:
        df = df[df[LABEL_COL] == 1]
    if injected_only:
        df = df[df[INJECT_COL] == 1]

    return df


# ---------------------------------------------------------------------
# 1) UMAP per intensity (hybrid feature set, injected laundering only)
# ---------------------------------------------------------------------
def umap_per_intensity(feature_cols):
    for intensity in ["low", "medium", "high"]:
        print(f"\n=== UMAP per intensity: {intensity} ===")
        df = load_intensity_df(intensity, injected_only=True, laundering_only=True)

        if df.empty:
            print(f"[WARN] No injected laundering rows in {intensity}.")
            continue

        # Adaptive sampling
        n = min(SAMPLES, len(df))
        df = df.sample(n, random_state=42)

        X = df[feature_cols].fillna(0.0).values

        # Cluster metrics in original feature space
        compute_cluster_metrics(X, f"UMAP_per_intensity_{intensity}")

        emb = run_umap(X)
        df["umap_x"] = emb[:, 0]
        df["umap_y"] = emb[:, 1]

        out_path = os.path.join(PLOT_DIR, f"umap_intensity_{intensity}.png")
        plot_scatter(
            df["umap_x"],
            df["umap_y"],
            f"UMAP Projection — {intensity.capitalize()} Intensity (Injected Laundering)",
            out_path,
        )


# ---------------------------------------------------------------------
# 2) UMAP baseline vs injected per intensity
# ---------------------------------------------------------------------
def umap_baseline_vs_injected(feature_cols):
    for intensity in ["low", "medium", "high"]:
        print(f"\n=== Baseline vs Injected UMAP: {intensity} ===")
        # Laundering rows only, both injected AND non-injected
        path = os.path.join(RAT_DIR, RAT_FILES[intensity])
        df = pd.read_csv(path)
        df = df[df[LABEL_COL] == 1]

        if df.empty:
            print(f"[WARN] No laundering rows in {intensity}.")
            continue

        # Injected label
        df["is_injected"] = (df[INJECT_COL] == 1).astype(int)

        # Balanced sampling: up to SAMPLES/2 from each class if possible
        dfs = []
        for val in [0, 1]:
            sub = df[df["is_injected"] == val]
            if sub.empty:
                continue
            n = min(SAMPLES // 2, len(sub))
            dfs.append(sub.sample(n, random_state=42))
        if not dfs:
            print(f"[WARN] No data for baseline vs injected in {intensity}.")
            continue

        df_balanced = pd.concat(dfs, ignore_index=True)

        X = df_balanced[feature_cols].fillna(0.0).values
        y = df_balanced["is_injected"].values

        compute_cluster_metrics(X, f"Baseline_vs_Injected_{intensity}")

        emb = run_umap(X)
        df_balanced["umap_x"] = emb[:, 0]
        df_balanced["umap_y"] = emb[:, 1]

        # Color by injected vs baseline
        colors = np.where(df_balanced["is_injected"] == 1, "Injected", "Baseline")

        plt.figure(figsize=(12, 8))
        for label, color in [("Baseline", "orange"), ("Injected", "royalblue")]:
            mask = (colors == label)
            plt.scatter(
                df_balanced.loc[mask, "umap_x"],
                df_balanced.loc[mask, "umap_y"],
                s=10,
                alpha=0.7,
                label=label,
                color=color,
            )

        plt.title(f"UMAP — {intensity.capitalize()} Intensity (Baseline vs Injected)")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend()
        out_path = os.path.join(PLOT_DIR, f"umap_baseline_vs_injected_{intensity}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved {out_path}")


# ---------------------------------------------------------------------
# 3) UMAP per motif type (using HIGH intensity injected laundering)
# ---------------------------------------------------------------------
def umap_motif_types():
    print("\n=== UMAP per motif type (high intensity, injected laundering) ===")
    df = load_intensity_df("high", injected_only=True, laundering_only=True)

    if df.empty:
        print("[WARN] No high-intensity injected laundering rows found.")
        return

    # Adaptive sampling
    n = min(SAMPLES, len(df))
    df = df.sample(n, random_state=42)

    motif_cols = ["motif_fanin", "motif_fanout", "motif_chain", "motif_cycle"]
    for col in motif_cols:
        if col not in df.columns:
            print(f"[WARN] Motif column {col} missing, skipping.")
            continue

    # Use motif-only features for these plots
    valid_motifs = [c for c in motif_cols if c in df.columns]
    if not valid_motifs:
        print("[WARN] No motif_* columns found.")
        return

    X = df[valid_motifs].fillna(0.0).values
    compute_cluster_metrics(X, "Motif_only_high_intensity")

    emb = run_umap(X)
    df["umap_x"] = emb[:, 0]
    df["umap_y"] = emb[:, 1]

    for motif_col in valid_motifs:
        out_path = os.path.join(PLOT_DIR, f"umap_motif_{motif_col}.png")
        plot_scatter(
            df["umap_x"],
            df["umap_y"],
            f"UMAP — High Intensity (Colored by {motif_col})",
            out_path,
            labels=df[motif_col].values,
            cmap="viridis",
            colorbar_label=motif_col,
        )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    # Discover hybrid feature columns from HIGH dataset (all should share schema)
    sample_path = os.path.join(RAT_DIR, RAT_FILES["high"])
    sample_df = pd.read_csv(sample_path, nrows=500)  # small sample is enough
    feature_cols = discover_feature_cols(sample_df)

    # 1) UMAP per intensity (injected laundering only)
    umap_per_intensity(feature_cols)

    # 2) UMAP baseline vs injected per intensity
    umap_baseline_vs_injected(feature_cols)

    # 3) UMAP per motif type (high intensity)
    umap_motif_types()

    if SKLEARN_AVAILABLE:
        print(f"\nCluster metrics written to: {CLUSTER_SCORES_PATH}")
    else:
        print("\n[INFO] scikit-learn not available; no cluster metrics were computed.")


if __name__ == "__main__":
    main()
