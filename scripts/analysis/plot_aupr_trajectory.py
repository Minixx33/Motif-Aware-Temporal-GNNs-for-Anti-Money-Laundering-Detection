"""
plot_aupr_trajectory.py

Plots mean test-set AUPR for GraphSAGE, GraphSAGE-T, and DyRep-Lite across:
  - the RAT trajectory: Baseline -> RAT-Low -> RAT-Medium -> RAT-High
  - the SLT trajectory: Baseline -> SLT-Low -> SLT-Medium -> SLT-High

Two side-by-side panels (shared y-axis) so the RAT and SLT gains are directly
comparable in scale. Both panels start from the same Baseline point.

Data source: hand-verified values transcribed from the paper's own tables
(cross-checked against results/*/seed*/*/metrics.json earlier in the project):
  - Baseline / RAT values : paper/sections/results_baseline.tex (Table "tab:baseline")
                             paper/sections/results_rat.tex      (Table "tab:rat_results")
  - SLT values             : Results and Discussion draft, Table "tab:slt_results"
                             (SLT table has NOT yet been independently re-verified
                             against metrics.json -- do that before final submission).

Edit the AUPR dict below to update numbers, colors, or labels; everything else
(axis limits, legend, annotations) will adjust automatically.
"""

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# DATA -- edit these values directly if the tables change.
# ---------------------------------------------------------------------------

MODELS = ["GraphSAGE", "GraphSAGE-T", "DyRep-Lite"]

COLORS = {
    "GraphSAGE":   "#1f77b4",
    "GraphSAGE-T": "#ff7f0e",
    "DyRep-Lite":  "#2ca02c",
}

MARKERS = {
    "GraphSAGE":   "o",
    "GraphSAGE-T": "s",
    "DyRep-Lite":  "^",
}

# Baseline AUPR (shared starting point for both panels)
BASELINE_AUPR = {
    "GraphSAGE":   0.3207,
    "GraphSAGE-T": 0.3266,
    "DyRep-Lite":  0.2929,
}

# RAT trajectory (Low, Medium, High)
RAT_AUPR = {
    "GraphSAGE":   [0.4208, 0.4243, 0.4216],
    "GraphSAGE-T": [0.4626, 0.4671, 0.4686],
    "DyRep-Lite":  [0.4592, 0.4568, 0.4568],
}

# SLT trajectory (Low, Medium, High)
SLT_AUPR = {
    "GraphSAGE":   [0.3254, 0.3251, 0.3193],
    "GraphSAGE-T": [0.3943, 0.3957, 0.4115],
    "DyRep-Lite":  [0.3067, 0.3068, 0.3068],
}

RAT_LABELS = ["Baseline", "RAT-Low", "RAT-Medium", "RAT-High"]
SLT_LABELS = ["Baseline", "SLT-Low", "SLT-Medium", "SLT-High"]

OUTPUT_PNG = "../../paper/figures/aupr_trajectory.png"
OUTPUT_PDF = "../../paper/figures/aupr_trajectory.pdf"

# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def main():
    fig, (ax_rat, ax_slt) = plt.subplots(
        1, 2, figsize=(10, 4.5), sharey=True
    )

    x = [0, 1, 2, 3]

    for model in MODELS:
        y_rat = [BASELINE_AUPR[model]] + RAT_AUPR[model]
        ax_rat.plot(
            x, y_rat,
            color=COLORS[model], marker=MARKERS[model],
            linewidth=2, markersize=7, label=model,
        )

        y_slt = [BASELINE_AUPR[model]] + SLT_AUPR[model]
        ax_slt.plot(
            x, y_slt,
            color=COLORS[model], marker=MARKERS[model],
            linewidth=2, markersize=7, label=model,
        )

    ax_rat.set_xticks(x)
    ax_rat.set_xticklabels(RAT_LABELS, rotation=20, ha="right")
    ax_rat.set_ylabel("AUPR")
    ax_rat.set_title("RAT Injection")
    ax_rat.grid(alpha=0.3)

    ax_slt.set_xticks(x)
    ax_slt.set_xticklabels(SLT_LABELS, rotation=20, ha="right")
    ax_slt.set_title("SLT Injection")
    ax_slt.grid(alpha=0.3)

    handles, labels = ax_rat.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3,
        bbox_to_anchor=(0.5, 1.06), frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
