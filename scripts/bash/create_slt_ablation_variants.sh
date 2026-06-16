#!/bin/bash
set -e
set -o pipefail

# ---------------------------------------------------------------------------
# run_slt_ablations.sh
#
# For each SLT weight variant:
#   1. Run slt_injector.py          → ibm_transcations_datasets/SLT/<variant>/
#   2. Run motif_graph_builder_static.py  → graphs/<dataset_name>/
#   3. Run motif_dyrep_graph_builder.py   → graphs_dyrep/<dataset_name>/
#   4. Run create_splits.py on both graph dirs
#
# Variants from the ablation table:
#   current        (0.30, 0.25, 0.20, 0.15, 0.10)
#   equal          (0.20, 0.20, 0.20, 0.20, 0.20)
#   neighbor_heavy (0.40, 0.20, 0.15, 0.15, 0.10)
#   amount_heavy   (0.20, 0.40, 0.15, 0.15, 0.10)
#   temporal_heavy (0.20, 0.15, 0.15, 0.25, 0.25)
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
PROJECT_ROOT="$(pwd)"

echo "Running from project root: $PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Python executable
# ---------------------------------------------------------------------------
PYTHON_EXE="/c/Users/g00084287/AppData/Local/miniconda3/envs/aml_project/python.exe"

if [ ! -f "$PYTHON_EXE" ]; then
    echo "ERROR: Python executable not found: $PYTHON_EXE"
    exit 1
fi

echo "Using Python: $PYTHON_EXE"
"$PYTHON_EXE" --version

# ---------------------------------------------------------------------------
# Script paths
# ---------------------------------------------------------------------------
INJECTOR="scripts/SLT/slt_injector.py"
STATIC_BUILDER="scripts/graph/motif_graph_builder_static.py"
DYREP_BUILDER="scripts/graph/motif_dyrep_graph_builder.py"
SPLITS_SCRIPT="scripts/create_splits.py"

for s in "$INJECTOR" "$STATIC_BUILDER" "$DYREP_BUILDER" "$SPLITS_SCRIPT"; do
    if [ ! -f "$s" ]; then
        echo "ERROR: Missing script: $s"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
ts=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="scripts/bash/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/slt_ablations_${ts}.log"

log() { echo "$@" | tee -a "$LOG_FILE"; }

log "==============================================================="
log " SLT ABLATION PIPELINE"
log " Timestamp: $ts"
log " Log: $LOG_FILE"
log "==============================================================="

# ---------------------------------------------------------------------------
# Ablation variants: NAME W_NEIGHBOR W_AMOUNT W_STRONG_TIE W_DELTA W_CUM
# ---------------------------------------------------------------------------
declare -a VARIANTS=(
    "current        0.30 0.25 0.20 0.15 0.10"
    "equal          0.20 0.20 0.20 0.20 0.20"
    "neighbor_heavy 0.40 0.20 0.15 0.15 0.10"
    "amount_heavy   0.20 0.40 0.15 0.15 0.10"
    "temporal_heavy 0.20 0.15 0.15 0.25 0.25"
)

INTENSITIES=("low" "medium" "high")

# ---------------------------------------------------------------------------
# Helper: elapsed time
# ---------------------------------------------------------------------------
elapsed() {
    local s=$1
    printf "%dh %dm %ds" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
total_start=$(date +%s)

for VARIANT_LINE in "${VARIANTS[@]}"; do
    read -r VARIANT W_NBR W_AMT W_STR W_DEL W_CUM <<< "$VARIANT_LINE"

    log ""
    log "==============================================================="
    log " VARIANT: $VARIANT"
    log "   neighbor=$W_NBR  amount=$W_AMT  strong_tie=$W_STR"
    log "   delta=$W_DEL  cumulative=$W_CUM"
    log "==============================================================="

    # -----------------------------------------------------------------------
    # STEP 1: Inject SLT features for this variant (all 3 intensities at once)
    # -----------------------------------------------------------------------
    log ""
    log ">>> [$(date +%H:%M:%S)] STEP 1: Injection — variant=$VARIANT"

    inject_start=$(date +%s)

    "$PYTHON_EXE" "$INJECTOR" \
        --variant      "$VARIANT" \
        --w_neighbor   "$W_NBR" \
        --w_amount     "$W_AMT" \
        --w_strong_tie "$W_STR" \
        --w_delta      "$W_DEL" \
        --w_cum        "$W_CUM" \
        2>&1 | tee -a "$LOG_FILE"

    inject_end=$(date +%s)
    log ">>> Injection done in $(elapsed $((inject_end - inject_start)))"

    # -----------------------------------------------------------------------
    # STEPS 2-4: Per intensity — build graphs + splits
    # -----------------------------------------------------------------------
    for INTENSITY in "${INTENSITIES[@]}"; do

        # Relative path from ibm_transcations_datasets/ to the injected CSV
        DATASET_REL="SLT/${VARIANT}/HI-Small_Trans_SLT_${VARIANT}_${INTENSITY}.csv"
        DATASET_NAME="HI-Small_Trans_SLT_${VARIANT}_${INTENSITY}"

        STATIC_OUT="${PROJECT_ROOT}/graphs/${DATASET_NAME}"
        DYREP_OUT="${PROJECT_ROOT}/graphs_dyrep/${DATASET_NAME}"

        log ""
        log "---------------------------------------------------------------"
        log " variant=$VARIANT  intensity=$INTENSITY"
        log " CSV:    ibm_transcations_datasets/$DATASET_REL"
        log " Static: graphs/$DATASET_NAME"
        log " DyRep:  graphs_dyrep/$DATASET_NAME"
        log "---------------------------------------------------------------"

        # -------------------------------------------------------------------
        # STEP 2: Static graph builder
        # -------------------------------------------------------------------
        log ""
        log ">>> [$(date +%H:%M:%S)] STEP 2: Static graph — $DATASET_NAME"

        step_start=$(date +%s)

        "$PYTHON_EXE" "$STATIC_BUILDER" \
            --dataset "$DATASET_REL" \
            2>&1 | tee -a "$LOG_FILE"

        step_end=$(date +%s)
        log ">>> Static graph done in $(elapsed $((step_end - step_start)))"

        # -------------------------------------------------------------------
        # STEP 3: DyRep graph builder
        # -------------------------------------------------------------------
        log ""
        log ">>> [$(date +%H:%M:%S)] STEP 3: DyRep graph — $DATASET_NAME"

        step_start=$(date +%s)

        "$PYTHON_EXE" "$DYREP_BUILDER" \
            --dataset "$DATASET_REL" \
            2>&1 | tee -a "$LOG_FILE"

        step_end=$(date +%s)
        log ">>> DyRep graph done in $(elapsed $((step_end - step_start)))"

        # -------------------------------------------------------------------
        # STEP 4: create_splits — static graph
        # -------------------------------------------------------------------
        log ""
        log ">>> [$(date +%H:%M:%S)] STEP 4a: Splits (static) — $DATASET_NAME"

        step_start=$(date +%s)

        "$PYTHON_EXE" "$SPLITS_SCRIPT" \
            --graph_folder "$STATIC_OUT" \
            --out_dir      "$STATIC_OUT" \
            2>&1 | tee -a "$LOG_FILE"

        step_end=$(date +%s)
        log ">>> Static splits done in $(elapsed $((step_end - step_start)))"

        # -------------------------------------------------------------------
        # STEP 4b: create_splits — DyRep graph
        # -------------------------------------------------------------------
        log ""
        log ">>> [$(date +%H:%M:%S)] STEP 4b: Splits (DyRep) — $DATASET_NAME"

        step_start=$(date +%s)

        "$PYTHON_EXE" "$SPLITS_SCRIPT" \
            --graph_folder "$DYREP_OUT" \
            --out_dir      "$DYREP_OUT" \
            2>&1 | tee -a "$LOG_FILE"

        step_end=$(date +%s)
        log ">>> DyRep splits done in $(elapsed $((step_end - step_start)))"

    done  # intensities

done  # variants

total_end=$(date +%s)

log ""
log "==============================================================="
log " ALL SLT ABLATION VARIANTS COMPLETED"
log " Total time: $(elapsed $((total_end - total_start)))"
log " Log: $LOG_FILE"
log "==============================================================="

touch "$LOG_DIR/SLT_ABLATIONS_DONE_${ts}.done"
echo ""
echo "SUCCESS — all variants and intensities processed."
