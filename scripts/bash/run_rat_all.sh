#!/bin/bash
set -e
set -o pipefail

# =============================================================================
# RUN ALL RAT EXPERIMENTS (LOW, MEDIUM, HIGH)
# Models: GraphSAGE, GraphSAGE-T, DyRep
# Dataset config: configs/datasets/rat.yaml
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
PROJECT_ROOT="$(pwd)"

echo "Running from project root: $PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Conda Activation (adjust if teammate uses a different path)
# ---------------------------------------------------------------------------
source "/c/ProgramData/Anaconda3/etc/profile.d/conda.sh"
conda activate aml_project

echo "Using Python: $(which python)"
python --version

# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------
BASE_CONFIG="configs/base.yaml"
DATASET_CONFIG="configs/datasets/rat.yaml"

if [ ! -f "$BASE_CONFIG" ]; then
    echo "ERROR: Missing base config: $BASE_CONFIG"
    exit 1
fi

if [ ! -f "$DATASET_CONFIG" ]; then
    echo "ERROR: Missing dataset config: $DATASET_CONFIG"
    exit 1
fi

# RAT intensities
INTENSITIES=("low" "medium" "high")

# ---------------------------------------------------------------------------
# Training scripts
# ---------------------------------------------------------------------------
TRAIN_SAGE="scripts/training/train_graphsage.py"
TRAIN_SAGET="scripts/training/train_graphsage_t.py"
TRAIN_DYREP="scripts/training/train_dyrep.py"

for script in "$TRAIN_SAGE" "$TRAIN_SAGET" "$TRAIN_DYREP"; do
    [ ! -f "$script" ] && echo "ERROR: Missing training script: $script" && exit 1
done

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
SAGE_CONFIG="configs/models/graphsage.yaml"
SAGET_CONFIG="configs/models/graphsage_t.yaml"
DYREP_CONFIG="configs/models/dyrep.yaml"

for config in "$SAGE_CONFIG" "$SAGET_CONFIG" "$DYREP_CONFIG"; do
    [ ! -f "$config" ] && echo "ERROR: Missing model config: $config" && exit 1
done

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
ts=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/rat_all_${ts}.log"

echo "" | tee -a "$LOG_FILE"
echo "===============================================================" | tee -a "$LOG_FILE"
echo " RUNNING ALL RAT EXPERIMENTS: LOW → MEDIUM → HIGH " | tee -a "$LOG_FILE"
echo " Models: GraphSAGE, GraphSAGE-T, DyRep " | tee -a "$LOG_FILE"
echo " Timestamp: $ts " | tee -a "$LOG_FILE"
echo "===============================================================" | tee -a "$LOG_FILE"

# GPU Info
echo "" | tee -a "$LOG_FILE"
echo ">>> GPU INFO:" | tee -a "$LOG_FILE"
nvidia-smi 2>&1 | tee -a "$LOG_FILE" || echo "GPU info unavailable" | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Helper: run a single model
# ---------------------------------------------------------------------------
run_model() {
    local MODEL_NAME="$1"
    local SCRIPT_PATH="$2"
    local MODEL_CONFIG="$3"
    local INTENSITY="$4"

    echo "" | tee -a "$LOG_FILE"
    echo "[ $(date +"%Y-%m-%d %H:%M:%S") ] >>> Running $MODEL_NAME (RAT-$INTENSITY)" | tee -a "$LOG_FILE"

    model_start=$(date +%s)

    python "$SCRIPT_PATH" \
        --config "$MODEL_CONFIG" \
        --dataset "$DATASET_CONFIG" \
        --base_config "$BASE_CONFIG" \
        --intensity "$INTENSITY" \
        2>&1 | tee -a "$LOG_FILE"

    model_end=$(date +%s)
    elapsed=$((model_end - model_start))

    h=$((elapsed / 3600))
    m=$(((elapsed % 3600) / 60))
    s=$((elapsed % 60))

    echo ">>> Finished: $MODEL_NAME (RAT-$INTENSITY) in ${h}h ${m}m ${s}s" | tee -a "$LOG_FILE"
}

# ---------------------------------------------------------------------------
# MAIN LOOP: run all intensities
# ---------------------------------------------------------------------------
start_time=$(date +%s)

for INTENSITY in "${INTENSITIES[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "---------------------------------------------------------------" | tee -a "$LOG_FILE"
    echo ">>> STARTING RAT INTENSITY: $INTENSITY" | tee -a "$LOG_FILE"
    echo "---------------------------------------------------------------" | tee -a "$LOG_FILE"

    run_model "GraphSAGE"   "$TRAIN_SAGE"   "$SAGE_CONFIG"   "$INTENSITY"
    run_model "GraphSAGE-T" "$TRAIN_SAGET"  "$SAGET_CONFIG"  "$INTENSITY"
    # run_model "DyRep"       "$TRAIN_DYREP"  "$DYREP_CONFIG"  "$INTENSITY"
done

end_time=$(date +%s)
total=$((end_time - start_time))

th=$((total / 3600))
tm=$(((total % 3600) / 60))
ts2=$((total % 60))

echo "" | tee -a "$LOG_FILE"
echo "===============================================================" | tee -a "$LOG_FILE"
echo " ALL RAT EXPERIMENTS COMPLETED SUCCESSFULLY " | tee -a "$LOG_FILE"
echo " Total time: ${th}h ${tm}m ${ts2}s ($total seconds) " | tee -a "$LOG_FILE"
echo " Log saved to: $LOG_FILE " | tee -a "$LOG_FILE"
echo "===============================================================" | tee -a "$LOG_FILE"

touch "$LOG_DIR/RAT_ALL_FINISHED_${ts}.done"

echo ""
echo "SUCCESS! All RAT experiments completed."
