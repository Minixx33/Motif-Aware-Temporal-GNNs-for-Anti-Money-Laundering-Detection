#!/bin/bash
set -e
set -o pipefail

# ---------------------------------------------------------
# Move to project root (go up 2 levels from scripts/batch/)
# ---------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
PROJECT_ROOT="$(pwd)"

echo "Running from project root: $PROJECT_ROOT"

# ---------------------------------------------------------
# Activate Conda
# ---------------------------------------------------------
source "/c/ProgramData/Anaconda3/etc/profile.d/conda.sh"
conda activate aml_project

echo "Using Python: $(which python)"
python --version

# ----------------------------------------------------------------------------
# Config paths - use RELATIVE paths for Python on Windows
# ----------------------------------------------------------------------------
BASE_CONFIG="configs/base.yaml"
DATASET_CONFIG="configs/datasets/baseline.yaml"

echo "Base config: $BASE_CONFIG"
echo "Dataset config: $DATASET_CONFIG"

# Verify configs exist (using absolute paths for bash)
if [ ! -f "$PROJECT_ROOT/$BASE_CONFIG" ]; then
    echo "ERROR: Base config not found at: $PROJECT_ROOT/$BASE_CONFIG"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/$DATASET_CONFIG" ]; then
    echo "ERROR: Dataset config not found at: $PROJECT_ROOT/$DATASET_CONFIG"
    exit 1
fi

# ----------------------------------------------------------------------------
# Timestamp + Logging
# ----------------------------------------------------------------------------
ts=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/baseline_runs_${ts}.log"

echo "=================================================================" | tee -a "$LOG_FILE"
echo " Running Baseline Experiments (SAGE, SAGE-T, TGAT) " | tee -a "$LOG_FILE"
echo " Timestamp: $ts " | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"

# GPU INFO
echo "" | tee -a "$LOG_FILE"
echo ">>> GPU INFO:" | tee -a "$LOG_FILE"
nvidia-smi 2>&1 | tee -a "$LOG_FILE" || echo "GPU info not available" | tee -a "$LOG_FILE"

# ----------------------------------------------------------------------------
# TRAINING SCRIPTS - use RELATIVE paths
# ----------------------------------------------------------------------------
TRAIN_SAGE="scripts/training/train_graphsage.py"
TRAIN_SAGET="scripts/training/train_graphsage_t.py"
TRAIN_TGAT="scripts/training/train_tgat.py"

# Verify scripts exist
for script in "$TRAIN_SAGE" "$TRAIN_SAGET" "$TRAIN_TGAT"; do
    if [ ! -f "$script" ]; then
        echo "ERROR: Training script not found: $script"
        exit 1
    fi
done

# ----------------------------------------------------------------------------
# Model configs - use RELATIVE paths
# ----------------------------------------------------------------------------
SAGE_CONFIG="configs/models/graphsage.yaml"
SAGET_CONFIG="configs/models/graphsage_t.yaml"
TGAT_CONFIG="configs/models/tgat.yaml"

# Verify model configs exist
for config in "$SAGE_CONFIG" "$SAGET_CONFIG" "$TGAT_CONFIG"; do
    if [ ! -f "$config" ]; then
        echo "ERROR: Model config not found: $config"
        exit 1
    fi
done

# ----------------------------------------------------------------------------
# Helper function: run a single model
# ----------------------------------------------------------------------------
run_model() {
    local MODEL_NAME="$1"
    local SCRIPT_PATH="$2"
    local MODEL_CONFIG="$3"

    echo "" | tee -a "$LOG_FILE"
    echo "[ $(date +"%Y-%m-%d %H:%M:%S") ] >>> Running $MODEL_NAME on baseline..." | tee -a "$LOG_FILE"

    model_start=$(date +%s)

    # Run from project root with relative paths
    if python "$SCRIPT_PATH" \
        --config "$MODEL_CONFIG" \
        --dataset "$DATASET_CONFIG" \
        --base_config "$BASE_CONFIG" \
        2>&1 | tee -a "$LOG_FILE"; then

        model_end=$(date +%s)
        model_time=$((model_end - model_start))

        echo ">>> $MODEL_NAME finished in ${model_time}s." | tee -a "$LOG_FILE"
    else
        echo ">>> ERROR: $MODEL_NAME FAILED!" | tee -a "$LOG_FILE"
        exit 1
    fi
}

# ----------------------------------------------------------------------------
# RUN MODELS
# ----------------------------------------------------------------------------
start_time=$(date +%s)

# run_model "GraphSAGE"   "$TRAIN_SAGE"   "$SAGE_CONFIG"
run_model "GraphSAGE-T" "$TRAIN_SAGET"  "$SAGET_CONFIG"
run_model "TGAT"        "$TRAIN_TGAT"   "$TGAT_CONFIG"

# ----------------------------------------------------------------------------
# TOTAL TIME SUMMARY
# ----------------------------------------------------------------------------
end_time=$(date +%s)
total_time=$((end_time - start_time))
total_m=$((total_time / 60))
total_s=$((total_time % 60))

echo "" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"
echo " ALL BASELINE EXPERIMENTS COMPLETED SUCCESSFULLY " | tee -a "$LOG_FILE"
echo " Total time: ${total_m}m ${total_s}s " | tee -a "$LOG_FILE"
echo " Log saved to: $LOG_FILE " | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"

# Marker file
touch "$LOG_DIR/BASELINE_FINISHED_${ts}.done"

echo ""
echo "SUCCESS! All experiments completed."
echo "Check results in: results/"