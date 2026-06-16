#!/bin/bash
# SLURM directives — ignored when run with bash directly:
# Array layout: 0=baseline  1=RAT-low  2=RAT-medium  3=RAT-high
#SBATCH --job-name=dyrep_all
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=scripts/bash/logs/dyrep_%A_%a.log
#SBATCH --error=scripts/bash/logs/dyrep_%A_%a.err
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
# Activate Conda (portable: Linux / macOS / Windows-GitBash)
# ---------------------------------------------------------
# Override the env name with: CONDA_ENV=my_env bash run_dyrep.sh
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
conda activate "${CONDA_ENV:-aml_project}"

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
         "$BASELINE_DS" "$RAT_DS"; do
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
# Dataset table (index matches SLURM array task ID)
# ---------------------------------------------------------
TASK_NAMES=("BASELINE" "RAT_low" "RAT_medium" "RAT_high")
TASK_DS=("$BASELINE_DS" "$RAT_DS" "$RAT_DS" "$RAT_DS")
TASK_INT=("" "low" "medium" "high")

# SLURM array → run only the task at this ID
# Local run   → run all sequentially
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    IDX=$SLURM_ARRAY_TASK_ID
    run_dyrep "${TASK_NAMES[$IDX]}" "${TASK_DS[$IDX]}" "${TASK_INT[$IDX]}"
else
    for IDX in 0 1 2 3; do
        run_dyrep "${TASK_NAMES[$IDX]}" "${TASK_DS[$IDX]}" "${TASK_INT[$IDX]}"
    done
fi

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
