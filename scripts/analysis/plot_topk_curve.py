"""
plot_topk_curve.py

Real, directly-computed Top-K alert retrieval precision curve for a
selected set of Model x Setting combinations -- not an interpolation
through the three table values (P@100/P@500/P@1000), an actual curve:
for each seed, predictions are ranked by score and precision is computed
at every K in K_GRID (cumulative true positives / K), then curves are
averaged across all available seeds for that condition.

Default series (edit the SERIES list below to add/remove any model-setting
combination), grouped by dataset:
    Baseline  GraphSAGE-T
    Baseline  DyRep-Lite
    RAT-Med   GraphSAGE-T
    RAT-Med   DyRep-Lite
    SLT-Med  GraphSAGE-T
    SLT-Med  DyRep-Lite

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
        # baseline was not rerun -- still the original seed folders
        "results_base": "../../results/HI-Small_Trans",
        "seeds": ["seed1337_experiment2", "seed2025_experiment_3",
                  "seed42_default_experiment", "seed4_seed4_experiment"],
    },
    "RAT-Med": {
        # new injection runs (missing model/seed combinations are skipped)
        "results_base": "../../results/HI-Small_Trans_RAT_medium",
        "seeds": [f"seed{i}_new_experiment" for i in range(1, 6)],
    },
    "SLT-Med": {
        "results_base": "../../results/HI-Small_Trans_SLT_medium",
        "seeds": [f"seed{i}_new_experiment" for i in range(1, 6)],
    },
}

MODEL_SPECS = {
    "GraphSAGE-T": {
        "result_dir_variants": ["graphsage-t", "graphsage_t"],
        "graph_dir": {
            "Baseline": "../../graphs/HI-Small_Trans",
            "RAT-Med":  "../../graphs/HI-Small_Trans_RAT_medium",
            "SLT-Med": "../../graphs/HI-Small_Trans_SLT_medium",
        },
        "split_dir": {
            "Baseline": "../../splits/HI-Small_Trans",
            "RAT-Med":  "../../splits/HI-Small_Trans_RAT_medium",
            "SLT-Med": "../../splits/HI-Small_Trans_SLT_medium",
        },
        "label_file": "y_edge.pt",
    },
    "DyRep-Lite": {
        "result_dir_variants": ["dyrep", "dyrep/Dyrep"],
        "graph_dir": {
            "Baseline": "../../graphs_dyrep/HI-Small_Trans",
            "RAT-Med":  "../../graphs_dyrep/HI-Small_Trans_RAT_medium",
            "SLT-Med": "../../graphs_dyrep/HI-Small_Trans_SLT_medium",
        },
        "split_dir": {
            "Baseline": "../../splits_dyrep/HI-Small_Trans",
            "RAT-Med":  "../../splits_dyrep/HI-Small_Trans_RAT_medium",
            "SLT-Med": "../../splits_dyrep/HI-Small_Trans_SLT_medium",
        },
        "label_file": "labels.pt",
    },
}

# (setting, model) pairs to plot -- must exist in CONDITIONS / MODEL_SPECS.
# Grouped by dataset (Baseline, then RAT-Med, then SLT-Med) so the legend
# reads top-to-bottom by dataset rather than by model.
SERIES = [
    ("Baseline", "GraphSAGE-T"),
    ("Baseline", "DyRep-Lite"),
    ("RAT-Med",  "GraphSAGE-T"),
    ("RAT-Med",  "DyRep-Lite"),
    ("SLT-Med", "GraphSAGE-T"),
    ("SLT-Med", "DyRep-Lite"),
]

# K values the curve is evaluated at -- log-spaced, with 100/500/1000 forced
# in explicitly so those specific (paper-reported) points land exactly.
K_GRID = sorted(set(np.unique(np.logspace(np.log10(10), np.log10(2000), 50)).astype(int)) | {100, 500, 1000})
K_GRID = np.array([k for k in K_GRID if k >= 10])

OUTPUT_PNG = "../../paper/figures/topk_retrieval_curve_v2.png"
OUTPUT_PDF = "../../paper/figures/topk_retrieval_curve.pdf"

# ---------------------------------------------------------------------------
# STYLE -- color encodes dataset (a shade of purple per Baseline/RAT/SLT,
# spread wide for clear distinction), linestyle encodes model (solid =
# GraphSAGE-T, dashed = DyRep-Lite), marker shape also encodes model
# (circle = GraphSAGE-T, star = DyRep-Lite). Times New Roman throughout,
# bold titles.
# ---------------------------------------------------------------------------

COLORS = {
    "Baseline": "#e0aaff",   # light violet
    "RAT-Med":  "#9d4edd",   # medium violet
    "SLT-Med": "#4c1d73",   # dark violet
}
LINESTYLES = {"GraphSAGE-T": "-", "DyRep-Lite": "--"}
# Custom (longer, cleaner) dash pattern instead of matplotlib's default tight
# dashes, which looked busy/cluttered on a log-x axis.
DASHES = {"GraphSAGE-T": (1, 0), "DyRep-Lite": (5, 2)}
MARKERS = {"GraphSAGE-T": "o", "DyRep-Lite": "*"}
MARKERSIZES = {"GraphSAGE-T": 7, "DyRep-Lite": 11}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8,
})

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
    skipped = []
    for seed in cond["seeds"]:
        for variant in spec["result_dir_variants"]:
            probs_path = os.path.join(cond["results_base"], seed, variant, "test_pred_probs.pt")
            if os.path.exists(probs_path):
                probs = torch.load(probs_path).numpy().reshape(-1)
                if len(probs) != len(y_true):
                    # Some runs' saved prediction arrays don't line up with the
                    # current test split size (e.g. seed2025_experiment_3's
                    # DyRep-Lite baseline run: 1,015,683 preds vs a 1,015,669-edge
                    # split). Skip rather than silently misaligning predictions
                    # with labels.
                    skipped.append((seed, len(probs), len(y_true)))
                    break
                curves.append(precision_at_k(probs, y_true, K_GRID))
                break
    if skipped:
        for seed, n_probs, n_labels in skipped:
            print(f"  [skip] {setting} {model} {seed}: {n_probs} preds != {n_labels} labels")
    if not curves:
        raise FileNotFoundError(f"No predictions found for {setting} {model}")
    return np.mean(curves, axis=0), len(curves)


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def main():
    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    all_curves = []
    for setting, model in SERIES:
        curve, n_seeds = compute_curve(setting, model)
        all_curves.append(curve)
        line, = ax.plot(
            K_GRID, curve,
            color=COLORS[setting], linestyle=LINESTYLES[model], linewidth=1.8,
            label=f"{setting} {model}",
        )
        line.set_dashes(DASHES[model])
        for k_mark in (100, 500, 1000):
            idx = np.where(K_GRID == k_mark)[0]
            if len(idx):
                ax.plot(k_mark, curve[idx[0]], color=COLORS[setting],
                         marker=MARKERS[model], markersize=MARKERSIZES[model],
                         markeredgecolor="white", markeredgewidth=0.9,
                         linestyle="none")

    ax.set_xscale("log")
    ax.set_xticks([10, 100, 500, 1000, 2000])
    ax.set_xticklabels(["10", "100", "500", "1000", "2000"])
    ax.minorticks_off()

    ax.set_xlabel("K (number of top-ranked alerts reviewed)")
    ax.set_ylabel("Precision@K")
    ax.set_title("Top-K Alert Retrieval Precision", fontweight="bold")
    ax.grid(alpha=0.25, linewidth=0.6)

    # Trim the dead space below the lowest curve instead of always starting
    # the y-axis at 0 -- keep a small pad above the true minimum.
    y_min_data = min(c.min() for c in all_curves)
    y_floor = max(0.0, np.floor(y_min_data * 50) / 50 - 0.005)
    ax.set_ylim(y_floor, 1.01)
    ax.set_xlim(10, 2000)

    # Legend order follows SERIES (already grouped by dataset). Compact.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels, loc="lower left", frameon=True, fontsize=8,
        handlelength=2.2, handletextpad=0.5, borderpad=0.35,
        labelspacing=0.25, framealpha=0.9, borderaxespad=0.15,
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
