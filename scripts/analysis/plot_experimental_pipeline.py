import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

COLORS = {
    "raw":      "#e0aaff",
    "process":  "#c77dff",
    "model":    "#9d4edd",
    "analysis": "#4c1d73",
}
TEXT = {"raw": "black", "process": "black", "model": "white", "analysis": "white"}
EDGE_COLOR = "#5a189a"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 9.5,
})

fig, ax = plt.subplots(figsize=(7.2, 8.0))
ax.set_xlim(0, 10)
ax.set_ylim(-0.15, 9.85)
ax.axis("off")

boxes = {}

def add_box(name, x, y, w, h, label, kind, fontsize=9.0):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.0, edgecolor="#2b0a45",
        facecolor=COLORS[kind], zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
             color=TEXT[kind], zorder=4, linespacing=1.35)
    boxes[name] = (x, y, w, h)

def pt(name, side, frac=0.0):
    x, y, w, h = boxes[name]
    if side == "top":
        return (x + frac * w, y + h / 2)
    if side == "bottom":
        return (x + frac * w, y - h / 2)
    if side == "left":
        return (x - w / 2, y)
    if side == "right":
        return (x + w / 2, y)

def straight(src, srcside, dst, dstside):
    p1, p2 = pt(src, srcside), pt(dst, dstside)
    patch = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                             linewidth=1.3, color=EDGE_COLOR, zorder=2,
                             shrinkA=0, shrinkB=0)
    ax.add_patch(patch)

def elbow(src, srcside, dst, dstside, drop=0.3, dst_frac=0.0):
    p1 = pt(src, srcside)
    p2 = pt(dst, dstside, frac=dst_frac)
    ymid = p1[1] - drop
    verts = [p1, (p1[0], ymid), (p2[0], ymid), p2]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
    path = Path(verts, codes)
    patch = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=13,
                             linewidth=1.3, color=EDGE_COLOR, zorder=2,
                             shrinkA=0, shrinkB=0)
    ax.add_patch(patch)

# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
add_box("raw", 1.75, 9.1, 2.6, 0.95, "IBM HI-Small\n+ Theory Injections", "raw")
add_box("construct", 5.8, 9.1, 3.6, 0.95, "Graph Construction &\nFeature Engineering", "process")

add_box("static", 2.7, 7.55, 3.1, 1.05, "Static Graph Tensors\n$G=(V,E)$", "process")
add_box("temporal", 7.7, 7.55, 3.2, 1.05, "Temporal Event Tensors\n(src, dst, t, k)", "process")

add_box("split_static", 2.7, 6.15, 3.1, 0.95, "Static Splits\n60/20/20 Stratified", "process")
add_box("split_temp", 7.7, 6.15, 3.2, 0.95, "Temporal Splits\n60/20/20 Chronological", "process")

add_box("sage", 1.55, 4.65, 2.3, 0.95, "GraphSAGE\n(Static GNN)", "model")
add_box("sage_t", 4.15, 4.65, 2.5, 0.95, "GraphSAGE-T\n(Temporal GNN)", "model")
add_box("dyrep", 7.7, 4.65, 2.7, 0.95, "DyRep-Lite\n(Event Model)", "model")

add_box("metrics", 4.9, 3.05, 4.6, 1.15,
        "Evaluation Metrics\nAUPR, ROC-AUC, F1, MCC,\nBal. Acc., Top-$k$", "analysis", fontsize=8.6)

add_box("ablations", 3.0, 1.35, 3.3, 1.05, "Ablation Studies\n(No Temporal, No Theory,\nTop-20 Features)", "analysis", fontsize=8.4)
add_box("featimp", 6.9, 1.35, 3.5, 1.05, "Feature Importance\n(RF Gini Importance on\nEdge Features)", "analysis", fontsize=8.4)

# ---------------------------------------------------------------------------
# EDGES
# ---------------------------------------------------------------------------
straight("raw", "right", "construct", "left")

elbow("construct", "bottom", "static", "top", drop=0.3)
elbow("construct", "bottom", "temporal", "top", drop=0.3)

straight("static", "bottom", "split_static", "top")
straight("temporal", "bottom", "split_temp", "top")

elbow("split_static", "bottom", "sage", "top", drop=0.25)
elbow("split_static", "bottom", "sage_t", "top", drop=0.25)
straight("split_temp", "bottom", "dyrep", "top")

elbow("sage", "bottom", "metrics", "top", drop=0.35, dst_frac=-0.35)
elbow("sage_t", "bottom", "metrics", "top", drop=0.35)
elbow("dyrep", "bottom", "metrics", "top", drop=0.35, dst_frac=0.35)

elbow("metrics", "bottom", "ablations", "top", drop=0.3)
elbow("metrics", "bottom", "featimp", "top", drop=0.3)

# ---------------------------------------------------------------------------
# LEGEND
# ---------------------------------------------------------------------------
legend_elems = [
    Patch(facecolor=COLORS["raw"], edgecolor="#2b0a45", label="Raw data"),
    Patch(facecolor=COLORS["process"], edgecolor="#2b0a45", label="Process / tensors"),
    Patch(facecolor=COLORS["model"], edgecolor="#2b0a45", label="Model"),
    Patch(facecolor=COLORS["analysis"], edgecolor="#2b0a45", label="Analysis"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, 0.015), frameon=True, fontsize=8.5,
           handlelength=1.3, handletextpad=0.5, columnspacing=1.2, framealpha=0.9)

fig.tight_layout(pad=0.4, rect=[0, 0.035, 1, 1])
fig.savefig("/tmp/pipeline_fig/experimental_pipeline.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
fig.savefig("/tmp/pipeline_fig/experimental_pipeline.pdf", bbox_inches="tight", pad_inches=0.05)
print("done")
