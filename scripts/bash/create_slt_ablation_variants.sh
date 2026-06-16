#!/bin/bash
# ===========================================================================
# create_slt_ablation_variants.sh
#
# Builds all SLT ablation graph variants:
#   1. slt_injector.py                → ibm_transcations_datasets/SLT/<variant>/
#   2. motif_graph_builder_static.py  → graphs/<dataset_name>/
#   3. motif_dyrep_graph_builder.py   → graphs_dyrep/<dataset_name>/
#   4. create_splits.py on both graph dirs
#
# LOCAL:  bash create_slt_ablation_variants.sh   (runs all 5 variants sequentially)
# SLURM:  sbatch create_slt_ablation_variants.sh (5 parallel jobs, one per variant)
#
# SLURM directives — ignored when run with bash directly:
#SBATCH --job-name=slt_create_variants
#SBATCH --array=0-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=scripts/bash/logs/slt_create_%A_%a.log
#SBATCH --error=scripts/bash/logs/slt_create_%A_%a.err
# ===========================================================================
set -e
set -o pipefail

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
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
conda activate "${CONDA_ENV:-aml_project}"

echo "Using Python: $(which python)"
python --version

# ---------------------------------------------------------------------------
# Script paths
# ---------------------------------------------------------------------------
INJECTOR="scripts/SLT/slt_injector.py"
STATIC_BUILDER="scripts/graph/motif_graph_builder_static.py"
DYREP_BUILDER="scripts/graph/motif_dyrep_graph_builder.py"
SPLITS_SCRIPT="scripts/create_splits.py"

for s in "$INJECTOR" "$STATIC_BUILDER" "$DYREP_BUILDER" "$SPLITS_SCRIPT"; do
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

INTENSITIES=("low" "medium" "high")

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

    # STEP 1: Inject (produces all 3 intensities at once)
    log ""
    log ">>> [$(date +%H:%M:%S)] STEP 1: Injection"
    t0=$(date +%s)
    python "$INJECTOR" \
        --variant      "$VARIANT" \
        --w_neighbor   "$W_NBR" \
        --w_amount     "$W_AMT" \
        --w_strong_tie "$W_STR" \
        --w_delta      "$W_DEL" \
        --w_cum        "$W_CUM"
    log ">>> Injection done in $(elapsed $(($(date +%s) - t0)))"

    for INTENSITY in "${INTENSITIES[@]}"; do

        # "current" uses the default flat path; all others use variant subfolder
        if [ "$VARIANT" = "current" ]; then
            DATASET_REL="SLT/HI-Small_Trans_SLT_${INTENSITY}.csv"
            DATASET_NAME="HI-Small_Trans_SLT_${INTENSITY}"
        else
            DATASET_REL="SLT/${VARIANT}/HI-Small_Trans_SLT_${VARIANT}_${INTENSITY}.csv"
            DATASET_NAME="HI-Small_Trans_SLT_${VARIANT}_${INTENSITY}"
        fi
        STATIC_OUT="${PROJECT_ROOT}/graphs/${DATASET_NAME}"
        DYREP_OUT="${PROJECT_ROOT}/graphs_dyrep/${DATASET_NAME}"

        log ""
        log "--- intensity=$INTENSITY ---"

        # STEP 2: Static graph
        log ">>> [$(date +%H:%M:%S)] STEP 2: Static graph"
        t0=$(date +%s)
        python "$STATIC_BUILDER" --dataset "$DATASET_REL"
        log ">>> done in $(elapsed $(($(date +%s) - t0)))"

        # STEP 3: DyRep graph
        log ">>> [$(date +%H:%M:%S)] STEP 3: DyRep graph"
        t0=$(date +%s)
        python "$DYREP_BUILDER" --dataset "$DATASET_REL"
        log ">>> done in $(elapsed $(($(date +%s) - t0)))"

        # STEP 4a: Splits — static
        log ">>> [$(date +%H:%M:%S)] STEP 4a: Splits (static)"
        t0=$(date +%s)
        python "$SPLITS_SCRIPT" --graph_folder "$STATIC_OUT" --out_dir "$STATIC_OUT"
        log ">>> done in $(elapsed $(($(date +%s) - t0)))"

        # STEP 4b: Splits — DyRep
        log ">>> [$(date +%H:%M:%S)] STEP 4b: Splits (DyRep)"
        t0=$(date +%s)
        python "$SPLITS_SCRIPT" --graph_folder "$DYREP_OUT" --out_dir "$DYREP_OUT"
        log ">>> done in $(elapsed $(($(date +%s) - t0)))"

    done  # intensities

done  # variants

log ""
log "==============================================================="
log " DONE — total time: $(elapsed $(($(date +%s) - total_start)))"
log "==============================================================="
