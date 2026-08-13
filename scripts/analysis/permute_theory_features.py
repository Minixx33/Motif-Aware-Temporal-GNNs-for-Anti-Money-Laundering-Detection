"""
permute_theory_features.py
---------------------------------------------------------------------------
Falsification / placebo control (review feedback Aug 4 2026, sec 4.3):
"permutation of RAT/SLT features across transactions."

Takes an already-built injected CSV (e.g. HI-Small_Trans_RAT_medium.csv or
HI-Small_Trans_SLT_pristine.csv) and randomly shuffles the *theory feature
columns only* across rows, while every other column (labels, account IDs,
amounts, timestamps, banks, metadata flags) stays in its original row order.

This severs any real association between the RAT/SLT/motif feature values
and the specific transaction (and its label) they're attached to, while
preserving the marginal distribution of each theory feature exactly. If a
model trained on the permuted file still performs well, the theory features
were not actually carrying transaction-specific signal -- the model would be
learning from something else (e.g. residual leakage elsewhere, or the
base/structural features already present). If performance collapses to
roughly the base-feature level, that supports the theory features being
doing real, transaction-specific work in the non-permuted data.

Column selection logic mirrors `is_theory_feature()` in
scripts/graph/motif_graph_builder_static.py exactly, so the columns treated
as "theory features" here are the same ones consumed as edge_attr theory_cols
by the graph builder -- duplicated here (not imported) because that script
runs its full pipeline at import time and isn't safe to import as a module.

Usage:
    python scripts/analysis/permute_theory_features.py \\
        --input_csv ibm_transcations_datasets/RAT/HI-Small_Trans_RAT_medium.csv \\
        --output_csv ibm_transcations_datasets/RAT/HI-Small_Trans_RAT_medium_permuted.csv \\
        --seed 123
"""

import argparse
import os

import numpy as np
import pandas as pd

# Kept in exact sync with scripts/graph/motif_graph_builder_static.py's
# METADATA_COLS / is_theory_feature(). If that file's definition changes,
# update this one too.
METADATA_COLS = {
    "RAT_injected", "RAT_intensity_level",
    "SLT_injected", "SLT_intensity_level",
    "STRAIN_injected", "STRAIN_intensity_level",
    "src_is_high_risk_peer", "dst_is_high_risk_peer",
}


def is_theory_feature(col):
    if col in METADATA_COLS:
        return False
    if col.startswith(("RAT_", "motif_", "SLT_", "STRAIN_")):
        return True
    if col.startswith("src_SLT_") or col.startswith("dst_SLT_"):
        return True
    if col in ("src_peer_risk_score", "dst_peer_risk_score"):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Placebo control: permute theory (RAT/SLT/motif) feature "
                     "columns across transaction rows."
    )
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to an already-built injected CSV "
                             "(RAT_low/medium/high, SLT_low/medium/high, or "
                             "a _pristine snapshot).")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output path (default: <input>_permuted.csv "
                             "next to the input file).")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.output_csv is None:
        base, ext = os.path.splitext(args.input_csv)
        args.output_csv = f"{base}_permuted{ext}"

    print(f"Loading: {args.input_csv}")
    df = pd.read_csv(args.input_csv, low_memory=False)

    theory_cols = [c for c in df.columns if is_theory_feature(c)]
    if not theory_cols:
        raise RuntimeError(
            f"No theory feature columns found in {args.input_csv} -- "
            f"is this really a RAT/SLT injected CSV?"
        )
    print(f"Permuting {len(theory_cols)} theory feature columns across "
          f"{len(df)} rows (seed={args.seed}):")
    for c in theory_cols:
        print(f"  - {c}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(df))

    # Reassign only the theory-feature sub-block using the permuted row
    # order; every other column (labels, IDs, amounts, timestamps, metadata
    # flags) is untouched and stays aligned to its original transaction.
    permuted_block = df.loc[perm, theory_cols].reset_index(drop=True)
    df[theory_cols] = permuted_block

    print(f"Saving: {args.output_csv}")
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {args.output_csv} [{len(df)} rows, "
          f"{len(theory_cols)} columns permuted]")


if __name__ == "__main__":
    main()
