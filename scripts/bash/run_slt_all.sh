#!/bin/bash
# SLURM directives — ignored when run with bash directly:
# Array layout: task_id // 3 = seed index (0-4 → seeds 1-5)
#               task_id  % 3 = intensity index (0=low, 1=medium, 2=high)
# Each task runs all 3 models (DyRep, GraphSAGE-T, GraphSAGE) sequentially.
#SBATCH --job-name=slt_all_5seeds
#SBATCH --account=acc-mialhajri
#SBATCH --array=0-14
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=500:00:00
#SBATCH --output=scripts/bash/logs/slt_all_%A_%a.log
#SBATCH --error=scripts/bash/logs/slt_all_%A_%a.err
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
PROJECT_ROOT="$(pwd)"

echo "Running from project root: $PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Portable conda activation
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------
BASE_CONFIG="configs/base.yaml"
DATASET_CONFIG="configs/datasets/slt.yaml"

[ -f "$BASE_CONFIG" ]    || { echo "ERROR: Missing base config: $BASE_CONFIG";    exit 1; }
[ -f "$DATASET_CONFIG" ] || { echo "ERROR: Missing dataset config: $DATASET_CONFIG"; exit 1; }

# ---------------------------------------------------------------------------
# Per-job base config (avoids race condition when array tasks run in parallel)
# ---------------------------------------------------------------------------
JOB_ID="${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
JOB_BASE_CONFIG="configs/base_slt_all_${JOB_ID}.yaml"
cp "$BASE_CONFIG" "$JOB_BASE_CONFIG"
cleanup() { rm -f "$JOB_BASE_CONFIG"; }
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
INTENSITIES=("low" "medium" "high")
SEEDS=(1 2 3 4 5)

# ---------------------------------------------------------------------------
# Training scripts + model configs
# ---------------------------------------------------------------------------
TRAIN_SAGE="scripts/training/train_graphsage.py"
TRAIN_SAGET="scripts/training/train_graphsage_t.py"
TRAIN_DYREP="scripts/training/train_dyrep.py"

SAGE_CONFIG="configs/models/graphsage.yaml"
SAGET_CONFIG="configs/models/graphsage_t.yaml"
DYREP_CONFIG="configs/models/dyrep.yaml"

for f in "$TRAIN_SAGE" "$TRAIN_SAGET" "$TRAIN_DYREP" \
         "$SAGE_CONFIG" "$SAGET_CONFIG" "$DYREP_CONFIG"; do
    [ -f "$f" ] || { echo "ERROR: Missing file: $f"; exit 1; }
done

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
ts=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="scripts/bash/logs"
mkdir -p "$LOG_DIR"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    LOG_FILE="$LOG_DIR/slt_all_5seeds_${ts}.log"
    log() { echo "$@" | tee -a "$LOG_FILE"; }
else
    log() { echo "$@"; }
fi

log "==============================================================="
log " RUNNING ALL SLT EXPERIMENTS WITH 5 SEEDS"
log " Host: $(hostname)  SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-none}"
log " Models: GraphSAGE, GraphSAGE-T, DyRep"
log " Intensities: ${INTENSITIES[*]}  Seeds: ${SEEDS[*]}"
log "==============================================================="

log ""
log ">>> GPU INFO:"
nvidia-smi 2>&1 || log "GPU info unavailable"

# ---------------------------------------------------------------------------
# Helper: patch the per-job base config
# ---------------------------------------------------------------------------
patch_base_config() {
    local SEED="$1"
    local EXP_NAME="$2"

    python - <<EOF
from pathlib import Path
import re

path = Path("$JOB_BASE_CONFIG")
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
# Helper: run a single model
# ---------------------------------------------------------------------------
run_model() {
    local MODEL_NAME="$1"
    local SCRIPT_PATH="$2"
    local MODEL_CONFIG="$3"
    local INTENSITY="$4"
    local SEED="$5"

    local SAFE_MODEL_NAME
    SAFE_MODEL_NAME="$(echo "$MODEL_NAME" | tr '-' '_' | tr '[:upper:]' '[:lower:]')"
    local EXP_NAME="slt_${INTENSITY}_${SAFE_MODEL_NAME}_seed${SEED}"

    log ""
    log ">>> [$(date +%H:%M:%S)] $EXP_NAME"

    patch_base_config "$SEED" "$EXP_NAME"

    t0=$(date +%s)

    python "$SCRIPT_PATH" \
        --config "$MODEL_CONFIG" \
        --dataset "$DATASET_CONFIG" \
        --base_config "$JOB_BASE_CONFIG" \
        --intensity "$INTENSITY" \
        2>&1

    log ">>> Finished: $EXP_NAME in $(( $(date +%s) - t0 ))s"
}

# ---------------------------------------------------------------------------
# Select which (seed, intensity) pairs to run
# SLURM: task_id // 3 = seed index, task_id % 3 = intensity index
# Local:  run all 15 combos sequentially
# ---------------------------------------------------------------------------
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SEED_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
    INT_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
    RUN_SEEDS=("${SEEDS[$SEED_IDX]}")
    RUN_INTENSITIES=("${INTENSITIES[$INT_IDX]}")
    log "Task $SLURM_ARRAY_TASK_ID → seed=${RUN_SEEDS[0]}  intensity=${RUN_INTENSITIES[0]}"
else
    RUN_SEEDS=("${SEEDS[@]}")
    RUN_INTENSITIES=("${INTENSITIES[@]}")
fi

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
start_time=$(date +%s)

for SEED in "${RUN_SEEDS[@]}"; do
    log ""
    log "==============================================================="
    log ">>> SEED: $SEED"
    log "==============================================================="

    for INTENSITY in "${RUN_INTENSITIES[@]}"; do
        log ""
        log "---------------------------------------------------------------"
        log ">>> intensity=$INTENSITY  seed=$SEED"
        log "---------------------------------------------------------------"

        run_model "DyRep"       "$TRAIN_DYREP"  "$DYREP_CONFIG"  "$INTENSITY" "$SEED"
        run_model "GraphSAGE-T" "$TRAIN_SAGET"  "$SAGET_CONFIG"  "$INTENSITY" "$SEED"
        run_model "GraphSAGE"   "$TRAIN_SAGE"   "$SAGE_CONFIG"   "$INTENSITY" "$SEED"
    done
done

total=$(( $(date +%s) - start_time ))
log ""
log "==============================================================="
log " ALL SLT EXPERIMENTS COMPLETED"
log " Total time: $(( total/3600 ))h $(( (total%3600)/60 ))m $(( total%60 ))s"
log "==============================================================="

touch "$LOG_DIR/SLT_ALL_5SEEDS_FINISHED_${ts}.done"
echo ""
echo "SUCCESS! All SLT experiments completed."
