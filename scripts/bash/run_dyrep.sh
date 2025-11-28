#!/bin/bash
set -e
set -o pipefail

# ---------------------------------------------------------
# Move to project root
# ---------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
PROJECT_ROOT="$(pwd)"

echo "Running from project root: $PROJECT_ROOT"

# ---------------------------------------------------------
# Activate Conda (Windows–GitBash compatible)
# ---------------------------------------------------------
source "/c/ProgramData/Anaconda3/etc/profile.d/conda.sh"
conda activate aml_project

echo "Using Python: $(which python)"
python --version

# ---------------------------------------------------------
# Paths (relative)
# ---------------------------------------------------------
BASE_CONFIG="configs/base.yaml"

# Dataset configs
BASELINE_DS="configs/datasets/baseline.yaml"
RAT_DS="configs/datasets/rat.yaml"

# DyRep model config
DYREP_CONFIG="configs/models/dyrep.yaml"

TRAIN_DYREP="scripts/training/train_dyrep.py"

# Verify config + script existence
for f in "$BASE_CONFIG" "$DYREP_CONFIG" "$TRAIN_DYREP" \
         "$BASELINE_DS" "$RAT_DS" "$SLT_DS" "$STRAIN_DS"; do
    if [ ! -f "$PROJECT_ROOT/$f" ]; then
        echo "ERROR: Missing file: $PROJECT_ROOT/$f"
        exit 1
    fi
done

# ---------------------------------------------------------
# Logging setup
# ---------------------------------------------------------
ts=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/dyrep_runs_${ts}.log"

echo "=================================================================" | tee -a "$LOG_FILE"
echo "      Running DyRep on ALL datasets (baseline + 3 theories)       " | tee -a "$LOG_FILE"
echo " Timestamp: $ts " | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"

# GPU INFO
echo "" | tee -a "$LOG_FILE"
echo ">>> GPU INFO:" | tee -a "$LOG_FILE"
nvidia-smi 2>&1 | tee -a "$LOG_FILE" || echo "GPU not available" | tee -a "$LOG_FILE"

# ---------------------------------------------------------
# Helper: run DyRep once
# ---------------------------------------------------------
run_dyrep() {
    local NAME="$1"
    local DATASET_CFG="$2"
    local INTENSITY="$3"   # can be "" for baseline

    echo "" | tee -a "$LOG_FILE"
    echo ">>> Running DyRep on: $NAME (intensity=$INTENSITY)" | tee -a "$LOG_FILE"

    start=$(date +%s)

    if python "$TRAIN_DYREP" \
        --config "$DYREP_CONFIG" \
        --dataset "$DATASET_CFG" \
        --intensity "$INTENSITY" \
        --base_config "$BASE_CONFIG" \
        2>&1 | tee -a "$LOG_FILE"; then

        end=$(date +%s)
        echo ">>> COMPLETED: $NAME ($((end-start))s)" | tee -a "$LOG_FILE"
    else
        echo ">>> FAILURE: DyRep on $NAME" | tee -a "$LOG_FILE"
        exit 1
    fi
}

# ---------------------------------------------------------
# RUN: BASELINE
# ---------------------------------------------------------
run_dyrep "BASELINE" "$BASELINE_DS" ""

# ---------------------------------------------------------
# RUN: RAT
# ---------------------------------------------------------
run_dyrep "RAT_low"    "$RAT_DS"    "low"
run_dyrep "RAT_medium" "$RAT_DS"    "medium"
run_dyrep "RAT_high"   "$RAT_DS"    "high"

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
echo "" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"
echo "         ALL DYREP EXPERIMENTS COMPLETED SUCCESSFULLY            " | tee -a "$LOG_FILE"
echo " Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"

touch "$LOG_DIR/DYREP_ALL_DONE_${ts}.done"

echo ""
echo "SUCCESS! All DyRep experiments finished."
echo "Check results in: results/"
