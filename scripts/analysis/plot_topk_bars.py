"""
plot_topk_bars.py

Grouped bar chart version of Top-K alert retrieval precision: one group per
K (100 / 500 / 1000), with one bar per Model x Setting combination inside
each group. Unlike a curve, this never implies a continuous function between
K values -- it's a precise, discrete readout, which is the more conservative
choice if you don't want the "is the line between points real data" question
to come up at all.

Default series (edit SERIES below to add/remove any model-setting
combination):
    Baseline  GraphSAGE-T
    RAT-Med   GraphSAGE-T
    SLT-High  GraphSAGE-T
    RAT-Med   DyRep-Lite
    SLT-High  DyRep-Lite

Data source: hand-verified values transcribed from the paper's own tables:
  - Baseline        : paper/sections/results_baseline.tex (Table "tab:baseline")
  - RAT-Medium      : paper/sections/results_rat.tex       (Table "tab:rat_results")
  - SLT-High        : Results and Discussion draft, Table "tab:slt_results"
                       (SLT table has NOT yet been independently re-verified
                       against metrics.json -- do that before final submission).
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# DATA -- edit this list to add/remove series.
# ---------------------------------------------------------------------------

K_VALUES = [100, 500, 1000]

COLORS = {
    "GraphSAGE-T": "#ff7f0e",
    "DyRep-Lite":  "#2ca02c",
}

HATCHES = {
    "Baseline": "...",
    "RAT-Med":  None,
    "SLT-High": "///",
}

ALPHAS = {
    "Baseline": 0.55,
    "RAT-Med":  1.0,
    "SLT-High": 0.85,
}

# (setting, model, [P@100, P@500, P@1000])
SERIES = [
    ("Baseline", "GraphSAGE-T", [0.880,  0.551,  0.382]),
    ("RAT-Med",  "GraphSAGE-T", [0.9575, 0.7270, 0.4898]),
    ("SLT-High", "GraphSAGE-T", [0.9300, 0.6708, 0.4464]),
    ("RAT-Med",  "DyRep-Lite",  [0.9625, 0.8210, 0.6330]),
    ("SLT-High", "DyRep-Lite",  [0.8920, 0.6260, 0.4692]),
]

OUTPUT_PNG = "../../paper/figures/topk_retrieval_bars.png"
OUTPUT_PDF = "../../paper/figures/topk_retrieval_bars.pdf"

# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def main():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    n_series = len(SERIES)
    n_groups = len(K_VALUES)
    group_width = 0.8
    bar_width = group_width / n_series
    group_centers = np.arange(n_groups)

    for i, (setting, model, values) in enumerate(SERIES):
        offset = (i - (n_series - 1) / 2) * bar_width
        positions = group_centers + offset
        ax.bar(
            positions, values, width=bar_width * 0.92,
            color=COLORS[model], alpha=ALPHAS[setting], hatch=HATCHES[setting],
            edgecolor="white", linewidth=0.8,
            label=f"{setting} {model}",
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"K = {k}" for k in K_VALUES])
    ax.set_ylabel("Precision@K")
    ax.set_title("Top-K Alert Retrieval Precision")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.05)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
