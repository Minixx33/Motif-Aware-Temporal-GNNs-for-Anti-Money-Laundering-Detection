#!/bin/bash
set -e
set -o pipefail

# =============================================================================
# RAT LOW INTENSITY EXPERIMENTS
# Runs: GraphSAGE, GraphSAGE-T, TGAT with low RAT intensity
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
PROJECT_ROOT="$(pwd)"

echo "Running from project root: $PROJECT_ROOT"

# -----------------------------------------------------------------------------
# Activate Conda Environment
# -----------------------------------------------------------------------------
source "/c/ProgramData/Anaconda3/etc/profile.d/conda.sh"
conda activate aml_project

echo "Using Python: $(which python)"
python --version

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_CONFIG="configs/base.yaml"
DATASET_CONFIG="configs/datasets/rat.yaml"
INTENSITY="low"

echo "Base config: $BASE_CONFIG"
echo "Dataset config: $DATASET_CONFIG"
echo "RAT Intensity: $INTENSITY"

# Verify configs exist
if [ ! -f "$BASE_CONFIG" ]; then
    echo "ERROR: Base config not found: $BASE_CONFIG"
    exit 1
fi

if [ ! -f "$DATASET_CONFIG" ]; then
    echo "ERROR: Dataset config not found: $DATASET_CONFIG"
    exit 1
fi

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
ts=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/rat_low_${ts}.log"

echo "===============================================================" | tee -a "$LOG_FILE"
echo " Running RAT LOW Intensity Experiments " | tee -a "$LOG_FILE"
echo " Models: GraphSAGE, GraphSAGE-T, TGAT " | tee -a "$LOG_FILE"
echo " Timestamp: $ts " | tee -a "$LOG_FILE"
echo "===============================================================" | tee -a "$LOG_FILE"

# GPU Info
echo "" | tee -a "$LOG_FILE"
echo ">>> GPU INFO:" | tee -a "$LOG_FILE"
nvidia-smi 2>&1 | tee -a "$LOG_FILE" || echo "GPU info not available" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# -----------------------------------------------------------------------------
# Training Scripts
# -----------------------------------------------------------------------------
TRAIN_SAGE="scripts/training/train_graphsage.py"
TRAIN_SAGET="scripts/training/train_graphsage_t.py"
TRAIN_TGAT="scripts/training/train_tgat.py"

# Verify scripts exist
for script in "$TRAIN_SAGE" "$TRAIN_SAGET" "$TRAIN_TGAT"; do
    if [ ! -f "$script" ]; then
        echo "ERROR: Training script not found: $script" | tee -a "$LOG_FILE"
        exit 1
    fi
done

# -----------------------------------------------------------------------------
# Model Configs
# -----------------------------------------------------------------------------
SAGE_CONFIG="configs/models/graphsage.yaml"
SAGET_CONFIG="configs/models/graphsage_t.yaml"
TGAT_CONFIG="configs/models/tgat.yaml"

# Verify model configs exist
for config in "$SAGE_CONFIG" "$SAGET_CONFIG" "$TGAT_CONFIG"; do
    if [ ! -f "$config" ]; then
        echo "ERROR: Model config not found: $config" | tee -a "$LOG_FILE"
        exit 1
    fi
done

# -----------------------------------------------------------------------------
# Helper Function: Run Single Model
# -----------------------------------------------------------------------------
run_model() {
    local MODEL_NAME="$1"
    local SCRIPT_PATH="$2"
    local MODEL_CONFIG="$3"

    echo "" | tee -a "$LOG_FILE"
    echo "[ $(date +"%Y-%m-%d %H:%M:%S") ] >>> Running $MODEL_NAME (RAT LOW)..." | tee -a "$LOG_FILE"

    model_start=$(date +%s)

    if python "$SCRIPT_PATH" \
        --config "$MODEL_CONFIG" \
        --dataset "$DATASET_CONFIG" \
        --base_config "$BASE_CONFIG" \
        --intensity "$INTENSITY" \
        2>&1 | tee -a "$LOG_FILE"; then

        model_end=$(date +%s)
        model_time=$((model_end - model_start))
        model_h=$((model_time / 3600))
        model_m=$(((model_time % 3600) / 60))
        model_s=$((model_time % 60))

        echo ">>> $MODEL_NAME finished in ${model_h}h ${model_m}m ${model_s}s" | tee -a "$LOG_FILE"
    else
        echo ">>> ERROR: $MODEL_NAME FAILED!" | tee -a "$LOG_FILE"
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Run All Models
# -----------------------------------------------------------------------------
start_time=$(date +%s)

run_model "GraphSAGE"   "$TRAIN_SAGE"   "$SAGE_CONFIG"
run_model "GraphSAGE-T" "$TRAIN_SAGET"  "$SAGET_CONFIG"
run_model "TGAT"        "$TRAIN_TGAT"   "$TGAT_CONFIG"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
end_time=$(date +%s)
total_time=$((end_time - start_time))
total_h=$((total_time / 3600))
total_m=$(((total_time % 3600) / 60))
total_s=$((total_time % 60))

echo "" | tee -a "$LOG_FILE"
echo "===============================================================" | tee -a "$LOG_FILE"
echo " ALL RAT LOW EXPERIMENTS COMPLETED SUCCESSFULLY " | tee -a "$LOG_FILE"
echo " Total time: ${total_h}h ${total_m}m ${total_s}s ($total_time seconds) " | tee -a "$LOG_FILE"
echo " Log saved to: $LOG_FILE " | tee -a "$LOG_FILE"
echo "===============================================================" | tee -a "$LOG_FILE"

# Create completion marker
touch "$LOG_DIR/RAT_LOW_FINISHED_${ts}.done"

echo ""
echo "SUCCESS! All RAT LOW experiments completed."
echo "Check results in: results/"