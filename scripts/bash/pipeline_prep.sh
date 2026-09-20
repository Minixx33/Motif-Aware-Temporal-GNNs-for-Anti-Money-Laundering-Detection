#!/bin/bash
# ===========================================================================
# pipeline_prep.sh
#
# Full data pipeline for the 5-condition primary comparison, from raw CSVs
# through trained-model-ready graphs: injection -> graph building ->
# structural_only / slt_plus_structural splicing -> splits -> node-degree
# leakage fix -> sanity check. CPU/pandas-bound (no GPU compute happens in
# this stage) -- runs once, sequentially, before any training job starts.
#
# Requires ibm_transcations_datasets/{HI-Small_Trans.csv,HI-Small_accounts.csv,
# HI-Small_Patterns.txt} to already exist (symlinked or copied in).
#
# SLURM:  sbatch scripts/bash/pipeline_prep.sh
# LOCAL:  bash scripts/bash/pipeline_prep.sh
#
# Runs on the CPU partition (confirmed via `sinfo`: cpu-dy-c5-0-* nodes have
# 8 CPUs, no GPU) -- this stage is pandas/CPU-bound, so it doesn't touch any
# of the 3 GPUs, and can start immediately without competing with training
# for GPU nodes.
#
# SLURM directives -- ignored when run with bash directly:
#SBATCH --job-name=aml_pipeline_prep
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=14G
#SBATCH --time=12:00:00
#SBATCH --output=scripts/bash/logs/pipeline_prep_%j.log
#SBATCH --error=scripts/bash/logs/pipeline_prep_%j.err
# ===========================================================================
set -e
set -o pipefail

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    cd "$SCRIPT_DIR/../.."
fi
PROJECT_ROOT="$(pwd)"
echo "Project root: $PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Portable conda activation
# ---------------------------------------------------------------------------
CONDA_BASE=""
if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
    CONDA_BASE="$("${CONDA_EXE}" info --base)"
elif command -v conda > /dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
else
    for candidate in \
        "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" \
        "/opt/anaconda3" "/opt/miniconda3" "/opt/conda" \
        "/opt/miniconda/etc/profile.d/conda.sh"; do
        if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
            CONDA_BASE="$candidate"
            break
        fi
    done
fi

if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "ERROR: Could not locate conda. Set CONDA_EXE or put conda on PATH."
    exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-aml_project}"
conda activate "$CONDA_ENV"

echo "Using Python: $(which python)"
python --version

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
mkdir -p "scripts/bash/logs"
if [ -z "${SLURM_JOB_ID:-}" ]; then
    ts=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="scripts/bash/logs/pipeline_prep_${ts}.log"
    log() { echo "$@" | tee -a "$LOG_FILE"; }
else
    log() { echo "$@"; }
fi

log "==============================================================="
log " PIPELINE PREP: injection -> graphs -> splits -> degree-fix"
log " Host: $(hostname)  PID: $$  SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
log "==============================================================="

RAW_DIR="ibm_transcations_datasets"
for f in "$RAW_DIR/HI-Small_Trans.csv" "$RAW_DIR/HI-Small_accounts.csv" "$RAW_DIR/HI-Small_Patterns.txt"; do
    [ -e "$f" ] || { log "ERROR: Missing raw input: $f"; exit 1; }
done

elapsed() { printf "%dh %dm %ds" $(($1/3600)) $((($1%3600)/60)) $(($1%60)); }
total_start=$(date +%s)

# ---------------------------------------------------------------------------
# Phase 1: theory injection (writes low/medium/high + pristine snapshots)
# ---------------------------------------------------------------------------
log ""
log ">>> [$(date +%H:%M:%S)] PHASE 1: injection"
t0=$(date +%s)
python scripts/rat/rat_injector.py --dump_pristine
python scripts/SLT/slt_injector.py --dump_pristine
log ">>> done in $(elapsed $(($(date +%s) - t0)))"

# ---------------------------------------------------------------------------
# Phase 2: build source graphs (static + DyRep) for baseline + both pristine sets
# ---------------------------------------------------------------------------
log ""
log ">>> [$(date +%H:%M:%S)] PHASE 2: graph building"
t0=$(date +%s)
python scripts/graph/baseline_graph_builder.py
python scripts/graph/baseline_dyrep_graph_builder.py
python scripts/graph/motif_graph_builder_static.py --dataset RAT/HI-Small_Trans_RAT_pristine.csv
python scripts/graph/motif_dyrep_graph_builder.py  --dataset RAT/HI-Small_Trans_RAT_pristine.csv
python scripts/graph/motif_graph_builder_static.py --dataset SLT/HI-Small_Trans_SLT_pristine.csv
python scripts/graph/motif_dyrep_graph_builder.py  --dataset SLT/HI-Small_Trans_SLT_pristine.csv
log ">>> done in $(elapsed $(($(date +%s) - t0)))"

# ---------------------------------------------------------------------------
# Phase 3: structural_only + slt_plus_structural (splices motif_* from RAT-pristine)
# ---------------------------------------------------------------------------
log ""
log ">>> [$(date +%H:%M:%S)] PHASE 3: structural conditions"
t0=$(date +%s)
python scripts/analysis/build_structural_only_graph.py
python scripts/analysis/build_structural_only_graph_dyrep.py
python scripts/analysis/build_slt_plus_structural_graph.py
python scripts/analysis/build_slt_plus_structural_graph_dyrep.py
log ">>> done in $(elapsed $(($(date +%s) - t0)))"

# ---------------------------------------------------------------------------
# Phase 4 + 5: splits, then node-degree leakage fix, per condition
# ---------------------------------------------------------------------------
DATASET_NAMES=(
    "HI-Small_Trans"
    "HI-Small_Trans_RAT_pristine"
    "HI-Small_Trans_RAT_pristine_structural_only"
    "HI-Small_Trans_SLT_pristine"
    "HI-Small_Trans_SLT_pristine_plus_structural"
)

log ""
log ">>> [$(date +%H:%M:%S)] PHASE 4: splits"
t0=$(date +%s)
for d in "${DATASET_NAMES[@]}"; do
    python scripts/create_splits.py --graph_folder "graphs/$d"
    python scripts/create_splits.py --graph_folder "graphs_dyrep/$d"
done
log ">>> done in $(elapsed $(($(date +%s) - t0)))"

log ""
log ">>> [$(date +%H:%M:%S)] PHASE 5: node-degree leakage fix"
t0=$(date +%s)
for d in "${DATASET_NAMES[@]}"; do
    python scripts/analysis/fix_node_degree_leakage.py --graph_dir "graphs/$d"
    python scripts/analysis/fix_node_degree_leakage.py --graph_dir "graphs_dyrep/$d"
done
log ">>> done in $(elapsed $(($(date +%s) - t0)))"

# ---------------------------------------------------------------------------
# Phase 6: sanity check -- abort the whole prep job rather than let a bad
# graph silently reach training. Expected: num_node_features=12 everywhere;
# num_edge_features = 12 (baseline) / 16 (structural_only) / 36 (rat_pristine)
# / 52 (slt_pristine) / 56 (slt_plus_structural).
# ---------------------------------------------------------------------------
log ""
log ">>> [$(date +%H:%M:%S)] PHASE 6: sanity check"
python - <<'PYEOF'
import json, sys

datasets = [
    "HI-Small_Trans",
    "HI-Small_Trans_RAT_pristine",
    "HI-Small_Trans_RAT_pristine_structural_only",
    "HI-Small_Trans_SLT_pristine",
    "HI-Small_Trans_SLT_pristine_plus_structural",
]
expected_edge_features = {
    "HI-Small_Trans": 12,
    "HI-Small_Trans_RAT_pristine": 36,
    "HI-Small_Trans_RAT_pristine_structural_only": 16,
    "HI-Small_Trans_SLT_pristine": 52,
    "HI-Small_Trans_SLT_pristine_plus_structural": 56,
}

ok = True
for d in datasets:
    with open(f"graphs/{d}/graph_stats.json") as f:
        stats = json.load(f)
    nf = stats.get("num_node_features")
    ef = stats.get("num_edge_features")
    exp_ef = expected_edge_features[d]
    status = "OK" if (nf == 12 and ef == exp_ef) else "MISMATCH"
    if status != "OK":
        ok = False
    print(f"  {d:<55} node_feats={nf} edge_feats={ef} (expected {exp_ef})  [{status}]")

if not ok:
    print("\nABORTING: one or more graphs failed the sanity check. "
          "Do not proceed to training until this is resolved.")
    sys.exit(1)
print("\nAll 5 static conditions passed the sanity check.")
PYEOF

total_time=$(($(date +%s) - total_start))
log ""
log "==============================================================="
log " PIPELINE PREP COMPLETE -- total time: $(elapsed $total_time)"
log "==============================================================="
