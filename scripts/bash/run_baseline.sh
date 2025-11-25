#!/bin/bash
# ============================================================================
# run_baseline_experiments.sh
# Run GraphSAGE, GraphSAGE-T, TGAT on BASELINE dataset only (WSL compatible)
# ============================================================================

# ========================================================================
# Conda environment activation for Git Bash on Windows
# ========================================================================

# Initialize conda for Git Bash
source "/c/ProgramData/Anaconda3/etc/profile.d/conda.sh"

# Activate your environment
echo "Activating Conda environment: aml_project"
conda activate aml_project

# Debugging info
echo "Using Python: $(which python)"
python --version

# ========================================================================

set -e          # Stop on error
set -u          # Stop on undefined variable
set -o pipefail # Catch errors in piped commands

# ----------------------------------------------------------------------------
# Resolve PROJECT ROOT directory (directory where this script lives)
# ----------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "Project root resolved to: $PROJECT_ROOT"

# ----------------------------------------------------------------------------
# Convert Windows paths → WSL Linux paths
# ----------------------------------------------------------------------------
win_to_wsl() {
    local win_path="$1"
    wsl_path=$(wslpath "$win_path" 2>/dev/null || echo "$win_path")
    echo "$wsl_path"
}


# ----------------------------------------------------------------------------
# Resolve absolute paths for configs
# ----------------------------------------------------------------------------
BASE_CONFIG_WIN="C:\\Users\\yasmi\\OneDrive\\Desktop\\Uni - Master's\\Fall 2025\\MLR 570\\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\\configs\\base.yaml"
DATASET_CONFIG_WIN="C:\\Users\\yasmi\\OneDrive\\Desktop\\Uni - Master's\\Fall 2025\\MLR 570\\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\\configs\\datasets\\baseline.yaml"

BASE_CONFIG=$(win_to_wsl "$BASE_CONFIG_WIN")
DATASET_CONFIG=$(win_to_wsl "$DATASET_CONFIG_WIN")

echo "Base config: $BASE_CONFIG"
echo "Dataset config: $DATASET_CONFIG"

# ----------------------------------------------------------------------------
# Timestamp + Logging
# ----------------------------------------------------------------------------
ts=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/baseline_runs_${ts}.log"

echo "=================================================================" | tee -a "$LOG_FILE"
echo " Running Baseline Experiments (SAGE, SAGE-T, TGAT) " | tee -a "$LOG_FILE"
echo " Timestamp: $ts " | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"

# GPU INFO
echo "" | tee -a "$LOG_FILE"
echo ">>> GPU INFO:" | tee -a "$LOG_FILE"
nvidia-smi | tee -a "$LOG_FILE"

# ----------------------------------------------------------------------------
# TRAINING SCRIPTS (absolute WSL paths)
# ----------------------------------------------------------------------------
TRAIN_SAGE="C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\scripts\training\train_graphsage.py"
TRAIN_SAGET="C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\scripts\training\train_graphsage_t.py"
TRAIN_TGAT="C:\Users\yasmi\OneDrive\Desktop\Uni - Master's\Fall 2025\MLR 570\Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection\scripts\training\train_tgat.py"

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

run_model "GraphSAGE"   "$TRAIN_SAGE"   "$PROJECT_ROOT/configs/models/graphsage.yaml"
run_model "GraphSAGE-T" "$TRAIN_SAGET"  "$PROJECT_ROOT/configs/models/graphsage_t.yaml"
run_model "TGAT"        "$TRAIN_TGAT"   "$PROJECT_ROOT/configs/models/tgat.yaml"

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
