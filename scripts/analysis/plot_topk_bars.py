"""
plot_topk_bars.py

Grouped bar chart version of Top-K alert retrieval precision: one group per
K (100 / 500 / 1000). Within each K group, bars are arranged as three
sub-blocks with a small visual gap between them: (1) Baseline, (2) RAT
GraphSAGE-T + RAT DyRep-Lite, (3) SLT GraphSAGE-T + SLT DyRep-Lite. Each
setting gets one hue family (gray for Baseline, orange for RAT, purple for
SLT); within a setting, the two models are the light/dark shade of that
same hue, with a thin border in the darker shade of the family (style
reference: NeRV/HNeRV PTQ-vs-QAT bar charts).

Default blocks (edit BLOCKS below to add/remove any model-setting
combination or change block membership -- order in each block list
determines left-to-right bar order within that block):
    Block 1: Baseline  GraphSAGE-T
    Block 2: RAT-Med   GraphSAGE-T,  RAT-Med   DyRep-Lite
    Block 3: SLT-High  GraphSAGE-T,  SLT-High  DyRep-Lite

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
# DATA -- edit BLOCKS to add/remove series or change grouping/order. Each
# block is a list of (setting, model, [P@100, P@500, P@1000]) drawn with no
# gap between its own bars; GAP_FRACTION controls the space between blocks
# WITHIN a K group, and GROUP_SPAN controls the space BETWEEN K groups (must
# be well below 1.0, or adjacent K groups visually fuse together).
# ---------------------------------------------------------------------------

K_VALUES = [100, 500, 1000]

BLOCKS = [
    [("Baseline", "GraphSAGE-T", [0.880,  0.551,  0.382])],
    [("RAT", "GraphSAGE-T", [0.9575, 0.7270, 0.4898]),
     ("RAT", "DyRep-Lite",  [0.9625, 0.8210, 0.6330])],
    [("SLT", "GraphSAGE-T", [0.9300, 0.6708, 0.4464]),
     ("SLT", "DyRep-Lite",  [0.8920, 0.6260, 0.4692])],
]

GAP_FRACTION = 0.25   # gap between blocks within a K group, as a fraction of one bar's width
GROUP_SPAN = 0.78    # total width used by all bars+internal gaps within a K group (<1.0 leaves room between K groups)

# One hue family per setting; within a setting, model picks the light/dark
# shade of that same hue.
FACE_COLORS = {
    ("Baseline", "GraphSAGE-T"): "#ecafe8",
    ("RAT", "GraphSAGE-T"):      "#aab8ff",
    ("RAT", "DyRep-Lite"):       "#404f9d",
    ("SLT", "GraphSAGE-T"):      "#dcb6ff",
    ("SLT", "DyRep-Lite"):       "#6a3d9a",
}

EDGE_COLORS = {
    "Baseline": "#8b0081",
    "RAT":      "#011785",
    "SLT":      "#4b2172",
}

OUTPUT_PNG = "../../paper/figures/topk_retrieval_bars.png"
OUTPUT_PDF = "../../paper/figures/topk_retrieval_bars.pdf"

# ---------------------------------------------------------------------------
# LAYOUT -- compute bar x-positions: bars within a block are flush against
# each other, blocks are separated by GAP_FRACTION extra spacing, and the
# whole K-group is scaled to GROUP_SPAN so neighboring K groups don't touch.
# ---------------------------------------------------------------------------

def compute_offsets():
    n_bars = sum(len(block) for block in BLOCKS)
    n_gaps = len(BLOCKS) - 1
    unit = GROUP_SPAN / (n_bars + n_gaps * GAP_FRACTION)

    offsets = []
    cursor = 0.0
    for bi, block in enumerate(BLOCKS):
        for entry in block:
            offsets.append((entry, cursor + unit / 2))
            cursor += unit
        if bi < len(BLOCKS) - 1:
            cursor += unit * GAP_FRACTION

    offsets = [(entry, pos - cursor / 2) for entry, pos in offsets]
    return offsets, unit


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def main():
    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    offsets, bar_width = compute_offsets()
    group_centers = np.arange(len(K_VALUES))

    plotted_handles = []
    plotted_labels = []
    for (setting, model, values), offset in offsets:
        positions = group_centers + offset
        bar = ax.bar(
            positions, values, width=bar_width * 0.95,
            color=FACE_COLORS[(setting, model)],
            edgecolor=EDGE_COLORS[setting], linewidth=1.1,
        )
        plotted_handles.append(bar)
        plotted_labels.append(f"{setting} {model}")

    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"K = {k}" for k in K_VALUES])
    ax.set_ylabel("Precision@K")
    ax.set_title("Top-K Alert Retrieval Precision")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.05)

    # Single-row legend, in the same left-to-right order as the bars
    # themselves (matplotlib's default multi-column legend fills
    # column-first, which scrambles this order if ncol < number of items).
    ax.legend(
        plotted_handles, plotted_labels,
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=len(plotted_labels), frameon=True, fontsize=9.5,
        columnspacing=1.2, handlelength=1.6,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
