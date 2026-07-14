"""
plot_topk_curve.py

Real, directly-computed Top-K alert retrieval precision curve for a
selected set of Model x Setting combinations -- not an interpolation
through the three table values (P@100/P@500/P@1000), an actual curve:
for each seed, predictions are ranked by score and precision is computed
at every K in K_GRID (cumulative true positives / K), then curves are
averaged across all available seeds for that condition.

Default series (edit the SERIES list below to add/remove any model-setting
combination):
    Baseline  GraphSAGE-T
    RAT-Med   GraphSAGE-T
    SLT-High  GraphSAGE-T
    RAT-Med   DyRep-Lite
    SLT-High  DyRep-Lite

Data source: per-seed test-set predictions and ground-truth labels, loaded
directly from the results tree (NOT the aggregated metrics.json / paper
table values used in the other figures):
  - GraphSAGE-T : results/<dataset>/<seed>/graphsage-t/test_pred_probs.pt
                  graphs/<dataset>/y_edge.pt + splits/<dataset>/test_edge_idx.pt
  - DyRep-Lite  : results/<dataset>/<seed>/dyrep/test_pred_probs.pt
                  graphs_dyrep/<dataset>/labels.pt + splits_dyrep/<dataset>/test_edge_idx.pt
                  (DyRep-Lite uses its own graph/split build, not the shared
                  static one GraphSAGE/GraphSAGE-T use.)

Edit CONDITIONS to add/remove seeds as more runs complete; edit SERIES to
change which Model x Setting curves are drawn.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# DATA -- edit paths/seeds here if the results tree changes.
# ---------------------------------------------------------------------------

CONDITIONS = {
    "Baseline": {
        "results_base": "../../results/HI-Small_Trans",
        "seeds": ["seed1337_experiment2", "seed2025_experiment_3",
                  "seed42_default_experiment", "seed4_seed4_experiment"],
    },
    "RAT-Med": {
        "results_base": "../../results/HI-Small_Trans_RAT_medium",
        "seeds": ["seed1337_experiment2", "seed2025_experiment_3",
                  "seed42_default_experiment", "seed4_seed4_experiment",
                  "seed4_seed5_experiment"],
    },
    "SLT-High": {
        "results_base": "../../results/HI-Small_Trans_SLT_high",
        "seeds": [f"seed{i}_seed1_experiment" for i in range(1, 6)],
    },
}

MODEL_SPECS = {
    "GraphSAGE-T": {
        "result_dir_variants": ["graphsage-t", "graphsage_t"],
        "graph_dir": {
            "Baseline": "../../graphs/HI-Small_Trans",
            "RAT-Med":  "../../graphs/HI-Small_Trans_RAT_medium",
            "SLT-High": "../../graphs/HI-Small_Trans_SLT_high",
        },
        "split_dir": {
            "Baseline": "../../splits/HI-Small_Trans",
            "RAT-Med":  "../../splits/HI-Small_Trans_RAT_medium",
            "SLT-High": "../../splits/HI-Small_Trans_SLT_high",
        },
        "label_file": "y_edge.pt",
    },
    "DyRep-Lite": {
        "result_dir_variants": ["dyrep"],
        "graph_dir": {
            "Baseline": "../../graphs_dyrep/HI-Small_Trans",
            "RAT-Med":  "../../graphs_dyrep/HI-Small_Trans_RAT_medium",
            "SLT-High": "../../graphs_dyrep/HI-Small_Trans_SLT_high",
        },
        "split_dir": {
            "Baseline": "../../splits_dyrep/HI-Small_Trans",
            "RAT-Med":  "../../splits_dyrep/HI-Small_Trans_RAT_medium",
            "SLT-High": "../../splits_dyrep/HI-Small_Trans_SLT_high",
        },
        "label_file": "labels.pt",
    },
}

# (setting, model) pairs to plot -- must exist in CONDITIONS / MODEL_SPECS.
SERIES = [
    ("Baseline", "GraphSAGE-T"),
    ("RAT-Med",  "GraphSAGE-T"),
    ("SLT-High", "GraphSAGE-T"),
    ("RAT-Med",  "DyRep-Lite"),
    ("SLT-High", "DyRep-Lite"),
]

# K values the curve is evaluated at -- log-spaced, with 100/500/1000 forced
# in explicitly so those specific (paper-reported) points land exactly.
K_GRID = sorted(set(np.unique(np.logspace(np.log10(10), np.log10(2000), 50)).astype(int)) | {100, 500, 1000})
K_GRID = np.array([k for k in K_GRID if k >= 10])

OUTPUT_PNG = "../../paper/figures/topk_retrieval_curve.png"
OUTPUT_PDF = "../../paper/figures/topk_retrieval_curve.pdf"

# ---------------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------------

COLORS = {"GraphSAGE-T": "#ff7f0e", "DyRep-Lite": "#2ca02c"}
LINESTYLES = {"Baseline": ":", "RAT-Med": "-", "SLT-High": "--"}
MARKERS = {"Baseline": "o", "RAT-Med": "s", "SLT-High": "^"}

# ---------------------------------------------------------------------------
# COMPUTATION
# ---------------------------------------------------------------------------

def precision_at_k(probs, y_true, k_grid):
    order = np.argsort(-probs)
    y_sorted = y_true[order]
    cum_tp = np.cumsum(y_sorted)
    return cum_tp[k_grid - 1] / k_grid


def compute_curve(setting, model):
    spec = MODEL_SPECS[model]
    cond = CONDITIONS[setting]
    y_all = torch.load(os.path.join(spec["graph_dir"][setting], spec["label_file"])).numpy()
    test_idx = torch.load(os.path.join(spec["split_dir"][setting], "test_edge_idx.pt")).numpy()
    y_true = y_all[test_idx]

    curves = []
    for seed in cond["seeds"]:
        for variant in spec["result_dir_variants"]:
            probs_path = os.path.join(cond["results_base"], seed, variant, "test_pred_probs.pt")
            if os.path.exists(probs_path):
                probs = torch.load(probs_path).numpy().reshape(-1)
                curves.append(precision_at_k(probs, y_true, K_GRID))
                break
    if not curves:
        raise FileNotFoundError(f"No predictions found for {setting} {model}")
    return np.mean(curves, axis=0), len(curves)


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def main():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for setting, model in SERIES:
        curve, n_seeds = compute_curve(setting, model)
        ax.plot(
            K_GRID, curve,
            color=COLORS[model], linestyle=LINESTYLES[setting], linewidth=2,
            label=f"{setting} {model}  (n={n_seeds} seeds)",
        )
        for k_mark in (100, 500, 1000):
            idx = np.where(K_GRID == k_mark)[0]
            if len(idx):
                ax.plot(k_mark, curve[idx[0]], color=COLORS[model],
                         marker=MARKERS[setting], markersize=7, linestyle="none")

    ax.set_xscale("log")
    ax.set_xticks([10, 100, 500, 1000, 2000])
    ax.set_xticklabels(["10", "100", "500", "1000", "2000"])
    ax.minorticks_off()

    ax.set_xlabel("K (number of top-ranked alerts reviewed)")
    ax.set_ylabel("Precision@K")
    ax.set_title("Top-K Alert Retrieval Precision (computed directly from test predictions)")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
