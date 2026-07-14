"""
plot_score_distribution.py

Compares the distribution of GraphSAGE-T's predicted probability for actual
laundering (positive-class) test transactions across three conditions:
Baseline, RAT-Medium, and SLT-Medium. This is the "why AUPR improves" figure:
it shows the classifier's confidence on true launderers shifting toward 1.0
as theory-guided features are injected, complementing the AUPR trajectory
plot (plot_aupr_trajectory.py).

For each condition, positive-class test predictions are pooled across every
available seed (not just one run), then smoothed with a boundary-reflected
Gaussian KDE (reflection at 0 and 1, since predicted probability is bounded)
implemented in plain numpy -- no scipy dependency required.

Data source: results/<dataset>/<seed>/graphsage-t/test_pred_probs.pt (model
output) matched against graphs/<dataset>/y_edge.pt + splits/<dataset>/test_edge_idx.pt
(ground-truth labels), i.e. real per-transaction predictions, not the
aggregated metrics.json summary values used in the other paper figures.

Edit SEEDS / DATASET paths below if new seeds are added; edit COLORS/labels
in the STYLE section to restyle without touching the data-loading logic.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# DATA -- edit paths/seeds here if the results tree changes.
# ---------------------------------------------------------------------------

CONDITIONS = [
    {
        "label": "Baseline",
        "results_base": "../../results/HI-Small_Trans",
        "graph_dir": "../../graphs/HI-Small_Trans",
        "split_dir": "../../splits/HI-Small_Trans",
        "seeds": [
            "seed1337_experiment2", "seed2025_experiment_3",
            "seed42_default_experiment", "seed4_seed4_experiment",
        ],
    },
    {
        "label": "RAT-Medium",
        "results_base": "../../results/HI-Small_Trans_RAT_medium",
        "graph_dir": "../../graphs/HI-Small_Trans_RAT_medium",
        "split_dir": "../../splits/HI-Small_Trans_RAT_medium",
        "seeds": [
            "seed1337_experiment2", "seed2025_experiment_3",
            "seed42_default_experiment", "seed4_seed4_experiment",
            "seed4_seed5_experiment",
        ],
    },
    {
        "label": "SLT-Medium",
        "results_base": "../../results/HI-Small_Trans_SLT_medium",
        "graph_dir": "../../graphs/HI-Small_Trans_SLT_medium",
        "split_dir": "../../splits/HI-Small_Trans_SLT_medium",
        "seeds": [f"seed{i}_seed1_experiment" for i in range(1, 6)],
    },
]

# GraphSAGE-T's results directory is named inconsistently across runs
# ("graphsage-t" vs "graphsage_t") -- try both.
MODEL_DIR_VARIANTS = ["graphsage-t", "graphsage_t"]

OUTPUT_PNG = "../../paper/figures/score_distribution.png"
OUTPUT_PDF = "../../paper/figures/score_distribution.pdf"

# ---------------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------------

FRAME_COLOR = "#141a35"
CARD_COLOR = "#ffffff"
TITLE_COLOR = "#1a1a2e"
GRID_COLOR = "#e3e6ee"

COLORS = {
    "Baseline":   "#6FA8DC",
    "RAT-Medium": "#4B2E83",
    "SLT-Medium": "#9B72CF",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 14.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 12.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
})

# ---------------------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------------------

def load_positive_scores(condition):
    """Pool GraphSAGE-T's predicted probability for true-positive (laundering)
    test transactions across every available seed for one condition."""
    y_edge = torch.load(os.path.join(condition["graph_dir"], "y_edge.pt")).numpy()
    test_idx = torch.load(os.path.join(condition["split_dir"], "test_edge_idx.pt")).numpy()
    y_true = y_edge[test_idx]

    pooled = []
    for seed in condition["seeds"]:
        for variant in MODEL_DIR_VARIANTS:
            probs_path = os.path.join(condition["results_base"], seed, variant, "test_pred_probs.pt")
            if os.path.exists(probs_path):
                probs = torch.load(probs_path).numpy().reshape(-1)
                pooled.append(probs[y_true == 1])
                break
    if not pooled:
        raise FileNotFoundError(f"No test_pred_probs.pt found for condition '{condition['label']}'")
    return np.concatenate(pooled)


# ---------------------------------------------------------------------------
# KDE -- boundary-reflected Gaussian KDE, numpy only (no scipy dependency).
# ---------------------------------------------------------------------------

def reflected_kde(data, grid, bw=None):
    """Gaussian KDE with reflection at 0 and 1 so density doesn't leak
    outside the valid probability range or fall off artificially at the
    boundary (most mass sits near 1.0)."""
    data = np.asarray(data)
    n = len(data)
    if bw is None:
        bw = max(1.06 * np.std(data) * n ** (-1 / 5), 0.01)
    augmented = np.concatenate([data, -data, 2 - data])
    diffs = (grid[:, None] - augmented[None, :]) / bw
    density = np.exp(-0.5 * diffs ** 2).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))
    return density


def style_panel(ax):
    ax.set_facecolor("none")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#9aa0b4")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=1.1, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, colors="#4d4d4d")


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def main():
    grid = np.linspace(0.0, 1.0, 400)

    fig = plt.figure(figsize=(9, 5.6), facecolor=FRAME_COLOR)
    card = FancyBboxPatch(
        (0.03, 0.06), 0.94, 0.85,
        transform=fig.transFigure,
        boxstyle="round,pad=0.0,rounding_size=0.035",
        linewidth=0, facecolor=CARD_COLOR, zorder=0,
    )
    fig.add_artist(card)

    ax = fig.add_axes([0.10, 0.14, 0.85, 0.60])
    style_panel(ax)

    for condition in CONDITIONS:
        label = condition["label"]
        scores = load_positive_scores(condition)
        density = reflected_kde(scores, grid)
        color = COLORS[label]
        ax.plot(grid, density, color=color, linewidth=2.5, zorder=3, label=f"{label}  (n={len(scores)})")
        ax.fill_between(grid, density, 0, color=color, alpha=0.15, zorder=2, linewidth=0)
        median = np.median(scores)
        ax.axvline(median, color=color, linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Predicted probability of laundering", fontweight="bold", color=TITLE_COLOR)
    ax.set_ylabel("Density", fontweight="bold", color=TITLE_COLOR)

    fig.text(
        0.5, 0.88,
        "GraphSAGE-T Score Distribution on True Laundering Transactions",
        ha="center", va="center", fontsize=15.5, fontweight="bold", color=TITLE_COLOR,
    )
    fig.text(
        0.5, 0.82,
        "Dashed lines mark the median predicted probability per condition",
        ha="center", va="center", fontsize=10.5, color="#5a5f73",
    )

    legend = ax.legend(
        loc="upper left", frameon=True, fancybox=True,
        edgecolor="#dcdfe8", facecolor="white", handlelength=1.8,
    )
    legend.get_frame().set_linewidth(0.9)

    fig.savefig(OUTPUT_PNG, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(OUTPUT_PDF, facecolor=fig.get_facecolor())
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
