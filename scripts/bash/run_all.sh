#!/bin/bash
set -e
set -o pipefail

# ============================================================
# Move to project root
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
PROJECT_ROOT="$(pwd)"

echo "Running from project root: $PROJECT_ROOT"

# ============================================================
# Activate Conda
# ============================================================
source "/c/ProgramData/Anaconda3/etc/profile.d/conda.sh"
conda activate aml_project

echo "Using Python: $(which python)"
python --version

# ============================================================
# CONFIG BASE
# ============================================================
BASE_CONFIG="configs/base.yaml"

if [ ! -f "$BASE_CONFIG" ]; then
    echo "ERROR: Cannot find $BASE_CONFIG"
    exit 1
fi

# ============================================================
# DATASETS TO LOOP OVER
# ============================================================
DATASETS=(
    "baseline"
    "rat"
)

# dataset → config path
declare -A DATASET_CONFIGS
DATASET_CONFIGS["baseline"]="configs/datasets/baseline.yaml"
DATASET_CONFIGS["rat"]="configs/datasets/rat.yaml"

# ============================================================
# MODELS TO LOOP OVER
# ============================================================
MODELS=(
    # "graphsage"
    "graphsage_t"
    # "dyrep"
)

# model → training script
declare -A MODEL_SCRIPTS
MODEL_SCRIPTS["graphsage"]="scripts/training/train_graphsage.py"
MODEL_SCRIPTS["graphsage_t"]="scripts/training/train_graphsage_t.py"
MODEL_SCRIPTS["dyrep"]="scripts/training/train_dyrep.py"

# model → config path
declare -A MODEL_CONFIGS
MODEL_CONFIGS["graphsage"]="configs/models/graphsage.yaml"
MODEL_CONFIGS["graphsage_t"]="configs/models/graphsage_t.yaml"
MODEL_CONFIGS["dyrep"]="configs/models/dyrep.yaml"

# ============================================================
# Logging
# ============================================================
ts=$(date +"%Y%m%d_%H%M%S")
mkdir -p logs
LOG_FILE="logs/ALL_RUNS_${ts}.log"

echo "=================================================================" | tee -a "$LOG_FILE"
echo " RUNNING ALL EXPERIMENTS: ALL MODELS × ALL DATASETS " | tee -a "$LOG_FILE"
echo " Timestamp: $ts" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo ">>> GPU INFO:" | tee -a "$LOG_FILE"
nvidia-smi 2>&1 | tee -a "$LOG_FILE" || echo "No GPU" | tee -a "$LOG_FILE"

# ============================================================
# Helper: run a model on a dataset
# ============================================================
run_model() {
    local MODEL_NAME="$1"
    local MODEL_SCRIPT="$2"
    local MODEL_CONFIG="$3"
    local DATASET_NAME="$4"
    local DATASET_CONFIG="$5"

    echo "" | tee -a "$LOG_FILE"
    echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "[ $(date +"%Y-%m-%d %H:%M:%S") ] >>> RUNNING: $MODEL_NAME on $DATASET_NAME" | tee -a "$LOG_FILE"
    echo "------------------------------------------------------------" | tee -a "$LOG_FILE"

    model_start=$(date +%s)

    python "$MODEL_SCRIPT" \
        --config "$MODEL_CONFIG" \
        --dataset "$DATASET_CONFIG" \
        --base_config "$BASE_CONFIG" \
        2>&1 | tee -a "$LOG_FILE"

    model_end=$(date +%s)
    elapsed=$((model_end - model_start))

    echo ">>> FINISHED: $MODEL_NAME on $DATASET_NAME in ${elapsed}s" | tee -a "$LOG_FILE"
}

# ============================================================
# MAIN LOOP: ALL MODELS × ALL DATASETS
# ============================================================
start_time=$(date +%s)

for DATASET in "${DATASETS[@]}"; do

    DATASET_CONFIG="${DATASET_CONFIGS[$DATASET]}"
    if [ ! -f "$DATASET_CONFIG" ]; then
        echo "ERROR: Missing dataset config $DATASET_CONFIG"
        exit 1
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "############################################################" | tee -a "$LOG_FILE"
    echo ">>> STARTING DATASET: $DATASET" | tee -a "$LOG_FILE"
    echo "############################################################" | tee -a "$LOG_FILE"

    for MODEL in "${MODELS[@]}"; do

        MODEL_SCRIPT="${MODEL_SCRIPTS[$MODEL]}"
        MODEL_CONFIG="${MODEL_CONFIGS[$MODEL]}"

        if [ ! -f "$MODEL_SCRIPT" ]; then
            echo "ERROR: Missing training script $MODEL_SCRIPT"
            exit 1
        fi

        if [ ! -f "$MODEL_CONFIG" ]; then
            echo "ERROR: Missing model config $MODEL_CONFIG"
            exit 1
        fi

        run_model "$MODEL" "$MODEL_SCRIPT" "$MODEL_CONFIG" "$DATASET" "$DATASET_CONFIG"

    done

done

# ============================================================
# Total summary
# ============================================================
end_time=$(date +%s)
TOTAL=$((end_time - start_time))
m=$((TOTAL/60))
s=$((TOTAL%60))

echo "" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"
echo " ALL EXPERIMENTS COMPLETED SUCCESSFULLY " | tee -a "$LOG_FILE"
echo " Total time: ${m}m ${s}s" | tee -a "$LOG_FILE"
echo " Log saved to: $LOG_FILE " | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"

touch "logs/ALL_EXPERIMENTS_DONE_${ts}.done"

echo ""
echo "SUCCESS! All experiments finished."
