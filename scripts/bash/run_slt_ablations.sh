#!/bin/bash
set -e
set -o pipefail

# ---------------------------------------------------------------------------
# run_slt_ablations.sh
#
# Runs GraphSAGE-T training with 5 seeds for every SLT ablation variant
# across all 3 intensities (low / medium / high).
#
# Variants:
#   current        (0.30, 0.25, 0.20, 0.15, 0.10)
#   equal          (0.20, 0.20, 0.20, 0.20, 0.20)
#   neighbor_heavy (0.40, 0.20, 0.15, 0.15, 0.10)
#   amount_heavy   (0.20, 0.40, 0.15, 0.15, 0.10)
#   temporal_heavy (0.20, 0.15, 0.15, 0.25, 0.25)
#
# Prerequisites: run create_slt_ablation_variants.sh first to build graphs.
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
# Paths
# ---------------------------------------------------------------------------
BASE_CONFIG="configs/base.yaml"
MODEL_CONFIG="configs/models/graphsage_t.yaml"
TRAIN_SCRIPT="scripts/training/train_graphsage_t.py"

for f in "$BASE_CONFIG" "$MODEL_CONFIG" "$TRAIN_SCRIPT"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing file: $f"
        exit 1
    fi
done

# Back up base.yaml and restore on exit
BASE_BACKUP="${BASE_CONFIG}.backup_slt_ablations"
cp "$BASE_CONFIG" "$BASE_BACKUP"
restore_base_config() {
    echo "Restoring original base.yaml..."
    cp "$BASE_BACKUP" "$BASE_CONFIG"
    rm -f "$BASE_BACKUP"
}
trap restore_base_config EXIT

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
ts=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="scripts/bash/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/slt_ablations_training_${ts}.log"

log() { echo "$@" | tee -a "$LOG_FILE"; }

log "==============================================================="
log " SLT ABLATION TRAINING — GraphSAGE-T, 5 seeds"
log " Timestamp: $ts"
log " Log: $LOG_FILE"
log "==============================================================="

log ""
log ">>> GPU INFO:"
nvidia-smi 2>&1 | tee -a "$LOG_FILE" || log "GPU info unavailable"

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
VARIANTS=(
    "current"
    "equal"
    "neighbor_heavy"
    "amount_heavy"
    "temporal_heavy"
)
INTENSITIES=("low" "medium" "high")
SEEDS=(1 2 3 4 5)

# ---------------------------------------------------------------------------
# Helper: patch base.yaml seed + experiment_name
# ---------------------------------------------------------------------------
update_base_config() {
    local SEED="$1"
    local EXP_NAME="$2"

    "$PYTHON_EXE" - <<EOF
from pathlib import Path
import re

path = Path("$BASE_CONFIG")
text = path.read_text()

text = re.sub(
    r'^(\s*seed:\s*).*$',
    r'\g<1>$SEED',
    text,
    flags=re.MULTILINE
)

text = re.sub(
    r'^(\s*experiment_name:\s*).*$',
    r'\g<1>"$EXP_NAME"',
    text,
    flags=re.MULTILINE
)

path.write_text(text)
EOF
}

# ---------------------------------------------------------------------------
# Helper: write a temp dataset config for this variant
# ---------------------------------------------------------------------------
make_dataset_config() {
    local VARIANT="$1"
    local TMP_CFG="configs/datasets/slt_${VARIANT}_tmp.yaml"

    cat > "$TMP_CFG" <<EOF
dataset:
  theory: "SLT"
  prefix: "HI-Small_Trans_SLT_${VARIANT}"
  available_intensities: ["low", "medium", "high"]
  requires_intensity: true
EOF

    echo "$TMP_CFG"
}

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

for VARIANT in "${VARIANTS[@]}"; do

    log ""
    log "==============================================================="
    log " VARIANT: $VARIANT"
    log "==============================================================="

    DATASET_CONFIG=$(make_dataset_config "$VARIANT")
    log "Dataset config: $DATASET_CONFIG"

    for INTENSITY in "${INTENSITIES[@]}"; do
        log ""
        log "---------------------------------------------------------------"
        log " variant=$VARIANT  intensity=$INTENSITY"
        log "---------------------------------------------------------------"

        for SEED in "${SEEDS[@]}"; do
            EXP_NAME="slt_${VARIANT}_${INTENSITY}_graphsage_t_seed${SEED}"

            log ""
            log ">>> [$(date +%H:%M:%S)] $EXP_NAME"

            update_base_config "$SEED" "$EXP_NAME"

            run_start=$(date +%s)

            "$PYTHON_EXE" "$TRAIN_SCRIPT" \
                --config      "$MODEL_CONFIG" \
                --dataset     "$DATASET_CONFIG" \
                --base_config "$BASE_CONFIG" \
                --intensity   "$INTENSITY" \
                2>&1 | tee -a "$LOG_FILE"

            run_end=$(date +%s)
            log ">>> Finished $EXP_NAME in $(elapsed $((run_end - run_start)))"

        done  # seeds

    done  # intensities

    # Clean up temp dataset config
    rm -f "$DATASET_CONFIG"

done  # variants

total_end=$(date +%s)

log ""
log "==============================================================="
log " ALL SLT ABLATION TRAINING COMPLETED"
log " Total time: $(elapsed $((total_end - total_start)))"
log " Log: $LOG_FILE"
log "==============================================================="

touch "$LOG_DIR/SLT_ABLATIONS_TRAINING_DONE_${ts}.done"
echo ""
echo "SUCCESS — all variants, intensities, and seeds completed."
