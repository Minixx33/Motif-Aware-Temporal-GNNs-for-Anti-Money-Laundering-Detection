#!/bin/bash
# ===========================================================================
# create_slt_ablation_variants.sh
#
# Builds SLT ablation graph variants for GraphSAGE-T only (run_slt_ablations.sh
# never trains DyRep on these, so no DyRep graph is built here):
#   1. slt_injector.py                → ibm_transcations_datasets/SLT/<variant>/
#   2. motif_graph_builder_static.py  → graphs/<dataset_name>/
#   3. create_splits.py (chronological) on the static graph dir
#   4. fix_node_degree_leakage.py on the static graph dir (overwrites x.pt's
#      degree columns, which motif_graph_builder_static.py computes from the
#      FULL graph regardless of split -- same fix applied to the primary
#      5-condition experiment in pipeline_prep.sh). This script never had
#      either fix until now: create_splits.py was called with no --split_mode
#      (defaulting to stratified_random) and the degree fix was never called
#      at all, so every existing SLT ablation result predates both fixes.
#
# The "current" variant's INJECTION/GRAPH-BUILD steps are still skipped -- it
# uses the exact same weights as your main production SLT pipeline, so
# rebuilding those would just reproduce graphs/HI-Small_Trans_SLT_medium
# byte-for-byte. But its SPLIT and DEGREE FIX are NOT skipped: that existing
# split predates the chronological-split fix too (it was built by the
# original, pre-fix pipeline), so it gets regenerated in place just like
# every other variant, pointed at the existing graph dir.
#
# LOCAL:  bash create_slt_ablation_variants.sh   (runs the 4 non-current variants)
# SLURM:  sbatch create_slt_ablation_variants.sh (parallel jobs, one per variant)
#
# This does pandas/CPU-bound preprocessing only (injection, graph building,
# splitting, degree-fix) -- no GPU op anywhere in slt_injector.py or
# motif_graph_builder_static.py -- so it runs on the cpu partition, like
# pipeline_prep.sh. It previously requested --partition=gpu with
# --qos=gpu-long-mialhajri-001, which wasted a GPU allocation for no reason
# AND competed with actual training jobs for that QoS's 30-job submit cap
# (the same GrpSubmitJobsLimit error hit during the primary experiment).
#
# SLURM directives — ignored when run with bash directly:
#SBATCH --job-name=slt_create_variants
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --array=0-4
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=scripts/bash/logs/slt_create_%A_%a.log
#SBATCH --error=scripts/bash/logs/slt_create_%A_%a.err
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
        "/c/ProgramData/Anaconda3" "/c/ProgramData/Miniconda3"; do
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
# Script paths
# ---------------------------------------------------------------------------
INJECTOR="scripts/SLT/slt_injector.py"
STATIC_BUILDER="scripts/graph/motif_graph_builder_static.py"
SPLITS_SCRIPT="scripts/create_splits.py"
DEGREE_FIX_SCRIPT="scripts/analysis/fix_node_degree_leakage.py"

for s in "$INJECTOR" "$STATIC_BUILDER" "$SPLITS_SCRIPT" "$DEGREE_FIX_SCRIPT"; do
    [ -f "$s" ] || { echo "ERROR: Missing script: $s"; exit 1; }
done

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
mkdir -p "scripts/bash/logs"

# SLURM captures stdout/stderr via --output/--error above.
# Local runs get a combined log file.
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    ts=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="scripts/bash/logs/slt_create_variants_${ts}.log"
    log() { echo "$@" | tee -a "$LOG_FILE"; }
else
    log() { echo "$@"; }
fi

log "==============================================================="
log " CREATE SLT ABLATION VARIANTS"
log " Host: $(hostname)  PID: $$"
log " SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-none (local run)}"
log "==============================================================="

# ---------------------------------------------------------------------------
# Variant table: NAME  W_NEIGHBOR  W_AMOUNT  W_STRONG_TIE  W_DELTA  W_CUM
# Index matches SLURM array task ID (0-4)
# ---------------------------------------------------------------------------
ALL_VARIANTS=(
    "current        0.30 0.25 0.20 0.15 0.10"   # 0
    "equal          0.20 0.20 0.20 0.20 0.20"   # 1
    "neighbor_heavy 0.40 0.20 0.15 0.15 0.10"   # 2
    "amount_heavy   0.20 0.40 0.15 0.15 0.10"   # 3
    "temporal_heavy 0.20 0.15 0.15 0.25 0.25"   # 4
)

INTENSITIES=("medium")

# SLURM array → run only the variant at this task ID
# Local run   → run all variants sequentially
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    VARIANTS_TO_RUN=("${ALL_VARIANTS[$SLURM_ARRAY_TASK_ID]}")
else
    VARIANTS_TO_RUN=("${ALL_VARIANTS[@]}")
fi

# ---------------------------------------------------------------------------
# Helper: elapsed time
# ---------------------------------------------------------------------------
elapsed() { printf "%dh %dm %ds" $(($1/3600)) $((($1%3600)/60)) $(($1%60)); }

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
total_start=$(date +%s)

for VARIANT_LINE in "${VARIANTS_TO_RUN[@]}"; do
    read -r VARIANT W_NBR W_AMT W_STR W_DEL W_CUM <<< "$VARIANT_LINE"

    log ""
    log "==============================================================="
    log " VARIANT: $VARIANT"
    log "   neighbor=$W_NBR  amount=$W_AMT  strong_tie=$W_STR"
    log "   delta=$W_DEL  cumulative=$W_CUM"
    log "==============================================================="

    if [ "$VARIANT" = "current" ]; then
        log ""
        log ">>> VARIANT=current uses the same weights as your main production"
        log ">>> SLT pipeline, but is NOT assumed to already exist on disk --"
        log ">>> it is built fresh like every other variant below. slt_injector.py"
        log ">>> has built-in handling for --variant current: it writes to the bare"
        log ">>> ibm_transcations_datasets/SLT/HI-Small_Trans_SLT_<intensity>.csv"
        log ">>> (no subfolder, no variant suffix), matching what run_slt_ablations.sh"
        log ">>> expects for this variant (graphs/HI-Small_Trans_SLT_medium, no suffix)."
    fi

    # STEP 1: Inject (medium intensity only -- that is all this pipeline
    # ever trains on; slt_injector.py would otherwise also produce unused
    # low/high CSVs for every variant)
    log ""
    log ">>> [$(date +%H:%M:%S)] STEP 1: Injection"
    t0=$(date +%s)
    python "$INJECTOR" \
        --variant      "$VARIANT" \
        --w_neighbor   "$W_NBR" \
        --w_amount     "$W_AMT" \
        --w_strong_tie "$W_STR" \
        --w_delta      "$W_DEL" \
        --w_cum        "$W_CUM" \
        --intensities  "medium"
    log ">>> Injection done in $(elapsed $(($(date +%s) - t0)))"

    for INTENSITY in "${INTENSITIES[@]}"; do

        # "current" writes to the bare SLT/ folder with no variant suffix
        # (matches slt_injector.py's own --variant current handling, and
        # run_slt_ablations.sh's PREFIX="HI-Small_Trans_SLT" for "current").
        # Every other variant gets its own subfolder + suffixed filename.
        if [ "$VARIANT" = "current" ]; then
            DATASET_REL="SLT/HI-Small_Trans_SLT_${INTENSITY}.csv"
            DATASET_NAME="HI-Small_Trans_SLT_${INTENSITY}"
        else
            DATASET_REL="SLT/${VARIANT}/HI-Small_Trans_SLT_${VARIANT}_${INTENSITY}.csv"
            DATASET_NAME="HI-Small_Trans_SLT_${VARIANT}_${INTENSITY}"
        fi
        STATIC_OUT="${PROJECT_ROOT}/graphs/${DATASET_NAME}"
        STATIC_SPLIT_OUT="${PROJECT_ROOT}/splits/${DATASET_NAME}"

        log ""
        log "--- intensity=$INTENSITY ---"

        # STEP 2: Static graph
        log ">>> [$(date +%H:%M:%S)] STEP 2: Static graph"
        t0=$(date +%s)
        python "$STATIC_BUILDER" --dataset "$DATASET_REL"
        log ">>> done in $(elapsed $(($(date +%s) - t0)))"

        # STEP 3: Splits — static (chronological; --out_dir avoids the
        # sibling "_chrono" folder create_splits.py would otherwise use,
        # which build_paths() doesn't know to look for)
        log ">>> [$(date +%H:%M:%S)] STEP 3: Splits (chronological)"
        t0=$(date +%s)
        python "$SPLITS_SCRIPT" --graph_folder "$STATIC_OUT" --split_mode chronological --out_dir "$STATIC_SPLIT_OUT"
        log ">>> splits done in $(elapsed $(($(date +%s) - t0)))"

        # STEP 4: Node-degree leakage fix
        log ">>> [$(date +%H:%M:%S)] STEP 4: Node-degree leakage fix"
        t0=$(date +%s)
        python "$DEGREE_FIX_SCRIPT" --graph_dir "$STATIC_OUT" --splits_dir "$STATIC_SPLIT_OUT"
        log ">>> degree-fix done in $(elapsed $(($(date +%s) - t0)))"

    done  # intensities

done  # variants

log ""
log "==============================================================="
log " DONE — total time: $(elapsed $(($(date +%s) - total_start)))"
log "==============================================================="
