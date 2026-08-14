"""
verify_pipeline_state.py
---------------------------------------------------------------------------
Diagnostic-only script (no side effects) to confirm the pristine CSVs,
static graphs, DyRep graphs, and splits built by the "Part 1" command
sequence are all present and internally consistent, before moving on to
the placebo test / training runs.

Checks, for the 4-condition primary comparison
(baseline, rat_natural=RAT_pristine, slt_natural=SLT_pristine, structural_only)
in both static (graphs/, splits/) and DyRep (graphs_dyrep/, splits_dyrep/)
form:

  1. Pristine CSVs exist, row count matches the raw HI-Small_Trans.csv row
     count, and the Timestamp column is uniformly formatted (the bug fixed
     earlier -- would show up here as >1 distinct string length).
  2. Each graph directory has all the files its format needs.
  3. edge_attr_cols.json is present and, for structural_only specifically,
     contains ONLY baseline + motif_* columns (zero RAT_/SLT_ columns --
     confirms no leakage snuck into that condition).
  4. Splits exist and train+val+test edge counts sum to the graph's total
     edge count (no rows silently dropped or duplicated).
  5. node_degree_fix.json is present in each graph dir (confirms
     fix_node_degree_leakage.py actually ran) and its summary is printed.

Usage (run from the project root):
    python scripts/analysis/verify_pipeline_state.py
"""

import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2] if "__file__" in dir() else Path.cwd()

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

_results = []


def report(status, msg):
    _results.append(status)
    print(f"[{status}] {msg}")


STATIC_FILES = [
    "edge_index.pt", "x.pt", "edge_attr.pt", "edge_attr_cols.json",
    "y_edge.pt", "y_node.pt", "node_mapping.json", "graph_stats.json",
]
DYREP_FILES = [
    "src.pt", "dst.pt", "ts.pt", "event_type.pt", "edge_attr.pt",
    "node_features.pt", "labels.pt", "y_node.pt", "edge_attr_cols.json",
    "node_mapping.json", "graph_stats.json",
]

DATASETS = [
    "HI-Small_Trans",
    "HI-Small_Trans_RAT_pristine",
    "HI-Small_Trans_SLT_pristine",
    "HI-Small_Trans_RAT_pristine_structural_only",
]

BASELINE_COLS = {
    "log_amt_rec", "log_amt_paid", "same_bank", "same_currency",
    "hour_of_day", "day_of_week", "is_weekend", "ts_normalized",
    "log_time_since_src", "log_time_since_dst", "pf_code", "rc_code",
}
MOTIF_COLS = {"motif_fanin", "motif_fanout", "motif_chain", "motif_cycle"}


def check_pristine_csv(theory, filename, raw_row_count):
    path = PROJECT_ROOT / "ibm_transcations_datasets" / theory / filename
    print(f"\n--- Pristine CSV: {path} ---")
    if not path.exists():
        report(FAIL, f"{path} does not exist.")
        return
    report(PASS, "File exists.")

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        n_rows = sum(1 for _ in f)

    if raw_row_count is not None:
        if n_rows == raw_row_count:
            report(PASS, f"Row count matches raw transactions: {n_rows:,}")
        else:
            report(FAIL, f"Row count {n_rows:,} != raw transaction count {raw_row_count:,} "
                          f"(accounts-duplicate join bug or a filtering step may still be off).")
    else:
        report(WARN, f"Row count = {n_rows:,} (no raw baseline count to compare against).")

    try:
        cols = [c.strip() for c in header.split(",")]
        ts_idx = cols.index("Timestamp")
    except ValueError:
        report(WARN, "Could not find a 'Timestamp' column to check format uniformity.")
        return

    lengths = set()
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i >= 5000:
                break
            parts = line.split(",")
            if len(parts) > ts_idx:
                lengths.add(len(parts[ts_idx]))
    if len(lengths) <= 1:
        report(PASS, f"Timestamp column string length is uniform ({lengths}) -- "
                      f"format-inference crash from before should not recur.")
    else:
        report(FAIL, f"Timestamp column has MIXED string lengths {lengths} in the first "
                      f"5000 rows -- the date_format fix may not have applied.")


def check_graph_dir(root_name, dataset_name, required_files, is_structural_only):
    d = PROJECT_ROOT / root_name / dataset_name
    print(f"\n--- Graph dir: {d} ---")
    if not d.is_dir():
        report(FAIL, f"{d} does not exist.")
        return None

    missing = [f for f in required_files if not (d / f).exists()]
    if missing:
        report(FAIL, f"Missing files: {missing}")
    else:
        report(PASS, f"All {len(required_files)} expected files present.")

    num_edges = None
    edge_attr_cols_path = d / "edge_attr_cols.json"
    if edge_attr_cols_path.exists():
        with open(edge_attr_cols_path) as f:
            cols = json.load(f)
        report(PASS if len(cols) > 0 else FAIL, f"edge_attr_cols.json has {len(cols)} columns.")

        if is_structural_only:
            colset = set(cols)
            expected = BASELINE_COLS | MOTIF_COLS
            leaked = [c for c in cols if c.startswith(("RAT_", "SLT_", "motif_")) and c not in MOTIF_COLS]
            if colset == expected and not leaked:
                report(PASS, "structural_only has EXACTLY baseline+motif columns, no RAT_/SLT_ leakage.")
            else:
                extra = colset - expected
                missing_expected = expected - colset
                report(FAIL, f"structural_only column mismatch. Unexpected extra: {extra or 'none'}. "
                              f"Missing expected: {missing_expected or 'none'}.")

    gstats_path = d / "graph_stats.json"
    if gstats_path.exists():
        with open(gstats_path) as f:
            gstats = json.load(f)
        num_edges = gstats.get("num_edges")
        print(f"  graph_stats.json: {gstats}")

    if (d / "edge_index.pt").exists():
        ei = torch.load(d / "edge_index.pt")
        actual_edges = ei.shape[1]
        report(PASS, f"edge_index.pt has {actual_edges:,} edges.")
        num_edges = actual_edges
    elif (d / "src.pt").exists():
        actual_edges = torch.load(d / "src.pt").shape[0]
        report(PASS, f"src.pt has {actual_edges:,} edges.")
        num_edges = actual_edges

    degfix_path = d / "node_degree_fix.json"
    if degfix_path.exists():
        with open(degfix_path) as f:
            degfix = json.load(f)
        report(PASS, f"node_degree_fix.json present -- degree fix ran. "
                      f"{degfix.get('nodes_with_changed_total_degree')} / {degfix.get('num_nodes')} "
                      f"nodes changed, using {degfix.get('num_train_edges'):,} train edges.")
    else:
        report(FAIL, "node_degree_fix.json MISSING -- fix_node_degree_leakage.py has not "
                      "been run on this graph yet.")

    return num_edges


def check_splits_dir(root_name, dataset_name, is_temporal, num_edges):
    splits_root = "splits_dyrep" if root_name == "graphs_dyrep" else "splits"
    d = PROJECT_ROOT / splits_root / dataset_name
    print(f"\n--- Splits dir: {d} ---")
    if not d.is_dir():
        report(FAIL, f"{d} does not exist.")
        return

    required = ["train_edge_idx.pt", "val_edge_idx.pt", "test_edge_idx.pt"]
    missing = [f for f in required if not (d / f).exists()]
    if missing:
        report(FAIL, f"Missing files: {missing}")
        return
    report(PASS, "train/val/test split files present.")

    train_n = len(torch.load(d / "train_edge_idx.pt"))
    val_n = len(torch.load(d / "val_edge_idx.pt"))
    test_n = len(torch.load(d / "test_edge_idx.pt"))
    total = train_n + val_n + test_n
    print(f"  train={train_n:,}  val={val_n:,}  test={test_n:,}  sum={total:,}")

    if num_edges is not None:
        if total == num_edges:
            report(PASS, f"train+val+test ({total:,}) == graph's total edge count ({num_edges:,}).")
        else:
            report(FAIL, f"train+val+test ({total:,}) != graph's total edge count ({num_edges:,}) "
                          f"-- splits may be stale relative to the current graph.")
    else:
        report(WARN, "No graph edge count available to cross-check against.")


def main():
    print("=" * 78)
    print("PIPELINE STATE VERIFICATION")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 78)

    raw_path = PROJECT_ROOT / "ibm_transcations_datasets" / "HI-Small_Trans.csv"
    raw_row_count = None
    if raw_path.exists():
        with open(raw_path, "r", encoding="utf-8") as f:
            next(f)
            raw_row_count = sum(1 for _ in f)
        print(f"\nRaw HI-Small_Trans.csv row count: {raw_row_count:,}")
    else:
        report(WARN, f"Raw file not found at {raw_path}, skipping row-count cross-checks.")

    check_pristine_csv("RAT", "HI-Small_Trans_RAT_pristine.csv", raw_row_count)
    check_pristine_csv("SLT", "HI-Small_Trans_SLT_pristine.csv", raw_row_count)

    for root_name, files, temporal in [("graphs", STATIC_FILES, False), ("graphs_dyrep", DYREP_FILES, True)]:
        for dataset_name in DATASETS:
            is_structural_only = dataset_name.endswith("_structural_only")
            num_edges = check_graph_dir(root_name, dataset_name, files, is_structural_only)
            check_splits_dir(root_name, dataset_name, temporal, num_edges)

    print("\n" + "=" * 78)
    n_fail = _results.count(FAIL)
    n_warn = _results.count(WARN)
    n_pass = _results.count(PASS)
    print(f"SUMMARY: {n_pass} passed, {n_warn} warnings, {n_fail} FAILED")
    print("=" * 78)
    if n_fail > 0:
        print("\nFix the FAILED items above before running the placebo test / training.")
        sys.exit(1)
    else:
        print("\nEverything checked out -- safe to proceed to the placebo test.")


if __name__ == "__main__":
    main()
