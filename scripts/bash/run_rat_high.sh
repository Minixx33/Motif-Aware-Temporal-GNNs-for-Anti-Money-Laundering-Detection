#!/bin/bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
PROJECT_ROOT="$(pwd)"

source "/c/ProgramData/Anaconda3/etc/profile.d/conda.sh"
conda activate aml_project

BASE_CONFIG="configs/base.yaml"
DATASET_CONFIG="configs/datasets/rat.yaml"

ts=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/rat_high_${ts}.log"

echo "===============================================================" | tee -a "$LOG_FILE"
echo " Running RAT HIGH Experiments " | tee -a "$LOG_FILE"
echo " Timestamp: $ts " | tee -a "$LOG_FILE"
echo "===============================================================" | tee -a "$LOG_FILE"

TRAIN_SAGET="scripts/training/train_graphsage_t.py"
TRAIN_TGAT="scripts/training/train_tgat.py"

SAGET_CONFIG="configs/models/graphsage_t.yaml"
TGAT_CONFIG="configs/models/tgat.yaml"

run_model() {
    local NAME="$1"
    local SCRIPT="$2"
    local CFG="$3"

    echo "" | tee -a "$LOG_FILE"
    echo ">>> Running $NAME (RAT HIGH)..." | tee -a "$LOG_FILE"

    model_start=$(date +%s)
    if python "$SCRIPT" \
        --config "$CFG" \
        --dataset "$DATASET_CONFIG" \
        --base_config "$BASE_CONFIG" \
        --intensity high \
        2>&1 | tee -a "$LOG_FILE"; then

        model_end=$(date +%s)
        echo ">>> $NAME finished in $((model_end - model_start)) seconds" | tee -a "$LOG_FILE"
    else
        echo ">>> ERROR: $NAME FAILED" | tee -a "$LOG_FILE"
        exit 1
    fi
}

start=$(date +%s)

run_model "GraphSAGE-T" "$TRAIN_SAGET" "$SAGET_CONFIG"
run_model "TGAT"        "$TRAIN_TGAT"  "$TGAT_CONFIG"

end=$(date +%s)
echo "Total time: $((end - start)) seconds" | tee -a "$LOG_FILE"

touch "$LOG_DIR/RAT_HIGH_DONE_${ts}.done"
echo "SUCCESS (RAT HIGH)"
