#!/bin/bash
# SLURM directives — ignored when run with bash directly:
# Array layout: task_id // 3 = dataset (0=baseline,1=rat_low,2=rat_med,3=rat_high)
#               task_id  % 3 = model   (0=graphsage,1=graphsage_t,2=dyrep)
#SBATCH --job-name=all_experiments
#SBATCH --account=acc-mialhajri
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=500:00:00
#SBATCH --output=scripts/bash/logs/all_exp_%A_%a.log
#SBATCH --error=scripts/bash/logs/all_exp_%A_%a.err
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
# Activate Conda (portable: Linux / macOS / Windows-GitBash)
# ============================================================
# Override the env name with: CONDA_ENV=my_env bash run_all.sh
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
    echo "ERROR: could not locate a conda install."
    echo "       Set CONDA_EXE, put conda on your PATH, or activate the env manually."
    exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "/shared/conda_envs/aml_project"

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
    "graphsage"
    "graphsage_t"
    "dyrep"
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
# Flat task list for SLURM array indexing
# dataset index: task_id // 3    model index: task_id % 3
# ============================================================
DATASET_LIST=("baseline" "rat_low" "rat_medium" "rat_high")
DATASET_CFG_LIST=(
    "configs/datasets/baseline.yaml"
    "configs/datasets/rat.yaml"
    "configs/datasets/rat.yaml"
    "configs/datasets/rat.yaml"
)
DATASET_INT_LIST=("" "low" "medium" "high")
MODEL_LIST=("graphsage" "graphsage_t" "dyrep")

# ============================================================
# MAIN LOOP: ALL MODELS × ALL DATASETS
# ============================================================
start_time=$(date +%s)

run_pair() {
    local DS_IDX="$1"
    local MOD_IDX="$2"

    local DATASET="${DATASET_LIST[$DS_IDX]}"
    local DATASET_CONFIG="${DATASET_CFG_LIST[$DS_IDX]}"
    local INTENSITY="${DATASET_INT_LIST[$DS_IDX]}"
    local MODEL="${MODEL_LIST[$MOD_IDX]}"
    local MODEL_SCRIPT="${MODEL_SCRIPTS[$MODEL]}"
    local MODEL_CONFIG="${MODEL_CONFIGS[$MODEL]}"

    [ -f "$DATASET_CONFIG" ] || { echo "ERROR: Missing dataset config $DATASET_CONFIG"; exit 1; }
    [ -f "$MODEL_SCRIPT" ]   || { echo "ERROR: Missing training script $MODEL_SCRIPT";  exit 1; }
    [ -f "$MODEL_CONFIG" ]   || { echo "ERROR: Missing model config $MODEL_CONFIG";      exit 1; }

    if [ -n "$INTENSITY" ]; then
        python "$MODEL_SCRIPT" \
            --config "$MODEL_CONFIG" \
            --dataset "$DATASET_CONFIG" \
            --base_config "$BASE_CONFIG" \
            --intensity "$INTENSITY" \
            2>&1 | tee -a "$LOG_FILE"
    else
        python "$MODEL_SCRIPT" \
            --config "$MODEL_CONFIG" \
            --dataset "$DATASET_CONFIG" \
            --base_config "$BASE_CONFIG" \
            2>&1 | tee -a "$LOG_FILE"
    fi
}

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    # SLURM: run the single (dataset, model) pair for this task
    DS_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
    MOD_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
    echo "SLURM task $SLURM_ARRAY_TASK_ID → dataset=${DATASET_LIST[$DS_IDX]} model=${MODEL_LIST[$MOD_IDX]}" | tee -a "$LOG_FILE"
    run_pair "$DS_IDX" "$MOD_IDX"
else
    # Local: run all combinations sequentially
    for DS_IDX in 0 1 2 3; do
        for MOD_IDX in 0 1 2; do
            run_pair "$DS_IDX" "$MOD_IDX"
        done
    done
fi

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
