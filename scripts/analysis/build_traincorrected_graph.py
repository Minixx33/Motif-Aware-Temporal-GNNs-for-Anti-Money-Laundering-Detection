"""
build_traincorrected_graph.py
----------------------------------------------------------------------
Builds a "train-corrected" graph: theory features are boosted ONLY on
train-split edges; val and test edges keep their pristine (never-boosted)
values. This eliminates the evaluation-validity leakage measured by
pristine_test_eval.py (RAT: AUPR -16.0%, SLT: AUPR -22.4% on average
across seeds when test rows are de-boosted, ROC-AUC roughly unchanged)
by construction: the model is trained, early-stopped on val, and
evaluated on test using data where val/test never contained an
artificially amplified row in the first place.

WHY SPLICING (NOT RETRAINING-TIME SURGERY) IS SAFE HERE
Both target architectures score each edge independently of every other
edge's own features:
  - GraphSAGE-T: SAGEConv message passing uses only x/edge_index, never
    edge_attr (see GraphSAGEEncoder.forward in train_graphsage_t.py).
    edge_attr is consumed ONLY by the final per-edge classifier MLP.
  - DyRep-Lite: DyRepEventModel.forward has no recurrent/temporal memory
    that carries across events -- node_emb is a plain learned embedding
    table, and each event's logit is MLP([h_src, h_dst, edge_feat,
    time_enc, type_emb]) computed independently (see train_dyrep.py).
So an edge's own boosted-vs-pristine feature value affects only that
edge's own label during training; splicing at the row level and then
training normally on the result is exact, not an approximation.

WHAT GETS COPIED UNCHANGED
Everything except edge_attr.pt is untouched by theory injection (node
degrees, event types, timestamps, graph structure), so those files are
copied verbatim from the boosted graph. The split membership itself
(train/val/test) also doesn't change -- only which feature values live
at each row -- so the split directory is copied as-is under the new
dataset name.

Usage:
    python scripts/analysis/build_traincorrected_graph.py \
        --graph_type static \
        --boosted_dir   graphs/HI-Small_Trans_RAT_medium \
        --pristine_dir  graphs/HI-Small_Trans_RAT_pristine \
        --split_dir     splits/HI-Small_Trans_RAT_medium \
        --out_graph_dir graphs/HI-Small_Trans_RAT_traincorrected_medium \
        --out_split_dir splits/HI-Small_Trans_RAT_traincorrected_medium

    python scripts/analysis/build_traincorrected_graph.py \
        --graph_type dyrep \
        --boosted_dir   graphs_dyrep/HI-Small_Trans_RAT_medium \
        --pristine_dir  graphs_dyrep/HI-Small_Trans_RAT_pristine \
        --split_dir     splits_dyrep/HI-Small_Trans_RAT_medium \
        --out_graph_dir graphs_dyrep/HI-Small_Trans_RAT_traincorrected_medium \
        --out_split_dir splits_dyrep/HI-Small_Trans_RAT_traincorrected_medium
----------------------------------------------------------------------
"""

import os
import json
import argparse
import shutil

import torch

STATIC_COPY_FILES = [
    "edge_index.pt", "x.pt", "timestamps.pt", "y_edge.pt", "y_node.pt",
    "node_mapping.json",
]
DYREP_COPY_FILES = [
    "src.pt", "dst.pt", "ts.pt", "event_type.pt", "node_features.pt",
    "labels.pt", "y_node.pt", "node_mapping.json",
]


def load_cols(d):
    with open(os.path.join(d, "edge_attr_cols.json")) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(
        description="Splice boosted(train)/pristine(val+test) edge features into a corrected graph"
    )
    ap.add_argument("--graph_type", choices=["static", "dyrep"], required=True)
    ap.add_argument("--boosted_dir", required=True)
    ap.add_argument("--pristine_dir", required=True)
    ap.add_argument("--split_dir", required=True,
                     help="Existing split dir for the BOOSTED graph (train/val/test_edge_idx.pt)")
    ap.add_argument("--out_graph_dir", required=True)
    ap.add_argument("--out_split_dir", required=True,
                     help="Where to copy the split files so the new dataset name resolves correctly")
    args = ap.parse_args()

    os.makedirs(args.out_graph_dir, exist_ok=True)
    os.makedirs(args.out_split_dir, exist_ok=True)

    copy_files = STATIC_COPY_FILES if args.graph_type == "static" else DYREP_COPY_FILES

    # ------------------------------------------------------------------
    # Sanity checks -- abort rather than silently building a misaligned
    # corrected graph.
    # ------------------------------------------------------------------
    b_cols = load_cols(args.boosted_dir)
    p_cols = load_cols(args.pristine_dir)
    assert b_cols == p_cols, (
        f"Column schema mismatch.\n  boosted:  {b_cols}\n  pristine: {p_cols}\n"
        "The pristine graph must be built from the --dump_pristine CSV of the "
        "SAME injector run as the boosted graph."
    )

    if args.graph_type == "static":
        b_struct = torch.load(os.path.join(args.boosted_dir, "edge_index.pt"))
        p_struct = torch.load(os.path.join(args.pristine_dir, "edge_index.pt"))
    else:
        b_struct = torch.stack([
            torch.load(os.path.join(args.boosted_dir, "src.pt")),
            torch.load(os.path.join(args.boosted_dir, "dst.pt")),
        ])
        p_struct = torch.stack([
            torch.load(os.path.join(args.pristine_dir, "src.pt")),
            torch.load(os.path.join(args.pristine_dir, "dst.pt")),
        ])
    assert torch.equal(b_struct, p_struct), (
        "Graph structure (src/dst) differs between boosted and pristine graphs -- "
        "row order or account mapping does not match. Aborting rather than "
        "producing a silently-misaligned corrected graph."
    )
    print("[OK] boosted and pristine graphs are row-aligned and schema-matched.")

    b_edge_attr = torch.load(os.path.join(args.boosted_dir, "edge_attr.pt"))
    p_edge_attr = torch.load(os.path.join(args.pristine_dir, "edge_attr.pt"))
    assert b_edge_attr.shape == p_edge_attr.shape, (
        f"edge_attr shape mismatch: boosted {tuple(b_edge_attr.shape)} vs "
        f"pristine {tuple(p_edge_attr.shape)}"
    )

    # ------------------------------------------------------------------
    # Splice: start from pristine everywhere, restore the boost only for
    # train rows. val/test rows stay pristine by construction.
    # ------------------------------------------------------------------
    train_idx = torch.load(os.path.join(args.split_dir, "train_edge_idx.pt"))
    val_idx = torch.load(os.path.join(args.split_dir, "val_edge_idx.pt"))
    test_idx = torch.load(os.path.join(args.split_dir, "test_edge_idx.pt"))

    corrected = p_edge_attr.clone()
    corrected[train_idx] = b_edge_attr[train_idx]

    print(f"[INFO] train edges (boosted retained): {train_idx.numel():,}")
    print(f"[INFO] val+test edges (forced pristine): {val_idx.numel() + test_idx.numel():,}")

    torch.save(corrected, os.path.join(args.out_graph_dir, "edge_attr.pt"))
    with open(os.path.join(args.out_graph_dir, "edge_attr_cols.json"), "w") as f:
        json.dump(b_cols, f, indent=2)

    for fname in copy_files:
        src = os.path.join(args.boosted_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out_graph_dir, fname))
        else:
            print(f"  [WARN] {fname} not found in boosted graph dir -- skipping")

    # graph_stats.json: carry through + annotate the correction
    src_stats = os.path.join(args.boosted_dir, "graph_stats.json")
    if os.path.exists(src_stats):
        with open(src_stats) as f:
            stats = json.load(f)
        stats["train_corrected"] = True
        stats["note"] = (
            "edge_attr for val/test rows forced to pristine (never-boosted) "
            "values; only train rows retain the theory-injection boost. "
            "See pristine_test_eval.py for the robustness check that motivated "
            "this correction."
        )
        with open(os.path.join(args.out_graph_dir, "graph_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
    else:
        print("  [WARN] graph_stats.json not found in boosted graph dir -- skipping")

    # Split membership is unchanged -- copy the split dir as-is under the
    # new dataset name so build_paths()/setup_experiment() resolve to it.
    for fname in os.listdir(args.split_dir):
        shutil.copy(os.path.join(args.split_dir, fname), os.path.join(args.out_split_dir, fname))

    print(f"\n[DONE] Corrected graph: {args.out_graph_dir}")
    print(f"[DONE] Corrected split: {args.out_split_dir}")


if __name__ == "__main__":
    main()
