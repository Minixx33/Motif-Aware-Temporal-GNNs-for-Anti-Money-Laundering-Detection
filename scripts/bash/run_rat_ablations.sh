#!/bin/bash
# ===========================================================================
# run_rat_ablations.sh
#
# Runs GraphSAGE-T training with 5 seeds for every RAT static ablation graph.
# Ablation graphs must already exist under graphs/HI-Small_Trans_RAT_medium__<name>/
# (build them first with: sbatch create_rat_ablation_graphs_static.sh)
#
# LOCAL:  bash run_rat_ablations.sh   (runs all 9 ablations × 5 seeds sequentially)
# SLURM:  sbatch run_rat_ablations.sh (9 parallel GPU jobs, one per ablation;
#                                      5 seeds per job)
#
# SLURM array layout (task ID → ablation):
#   0: no_struct       3: no_entity       6: no_motif
#   1: no_temp         4: no_rat_scores   7: no_crossbank
#   2: no_amount       5: no_burst_pattern 8: top20_features
#
# SLURM directives — ignored when run with bash directly:
#SBATCH --job-name=rat_ablations_train
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --qos=gpu-long-mialhajri-001
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=500:00:00
#SBATCH --output=scripts/bash/logs/rat_ablations_%A_%a.log
#SBATCH --error=scripts/bash/logs/rat_ablations_%A_%a.err
# ===========================================================================
set -e
set -o pipefail

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    cd "$SCRIPT_DIR/../.."
fi
PROJECT_ROOT="$(pwd)"
echo "Project root: $PROJECT_ROOT"

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
    echo "ERROR: Could not locate conda. Set CONDA_EXE or put conda on PATH."
    exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-aml_project}"
conda activate "$CONDA_ENV"

echo "Using Python: $(which python)"
python --version

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_CONFIG="configs/base.yaml"
MODEL_CONFIG="configs/models/graphsage_t.yaml"
TRAIN_SCRIPT="scripts/training/train_graphsage_t.py"

for f in "$BASE_CONFIG" "$MODEL_CONFIG" "$TRAIN_SCRIPT"; do
    [ -f "$f" ] || { echo "ERROR: Missing file: $f"; exit 1; }
done

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
mkdir -p "scripts/bash/logs"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    ts=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="scripts/bash/logs/rat_ablations_training_${ts}.log"
    log() { echo "$@" | tee -a "$LOG_FILE"; }
else
    log() { echo "$@"; }
fi

log "==============================================================="
log " RAT ABLATION TRAINING — GraphSAGE-T, 5 seeds"
log " Host: $(hostname)  PID: $$"
log " SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
log " SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-none (local run)}"
log "==============================================================="

log ""
log ">>> GPU INFO:"
nvidia-smi 2>&1 || log "GPU info unavailable"

# ---------------------------------------------------------------------------
# Ablation table
# Index must match SLURM array task ID.
# ---------------------------------------------------------------------------
ABLATIONS=(
    "no_struct"        # 0
    "no_temp"          # 1
    "no_amount"        # 2
    "no_burst_pattern" # 3
    "no_entity"        # 4
    "no_rat_scores"    # 5
    "no_motif"         # 6
    "no_crossbank"     # 7
    "top20_features"   # 8
)

SEEDS=(1 2 3 4 5)

# ---------------------------------------------------------------------------
# Select which ablations to run:
#   SLURM array job → single ablation from SLURM_ARRAY_TASK_ID
#   Local run       → all 9 ablations sequentially
# ---------------------------------------------------------------------------
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    RUN_ABLATIONS=("${ABLATIONS[$SLURM_ARRAY_TASK_ID]}")
else
    RUN_ABLATIONS=("${ABLATIONS[@]}")
fi

# ---------------------------------------------------------------------------
# Helper: patch a job-specific copy of base.yaml
# Per-job temp file avoids race conditions in parallel array tasks.
# ---------------------------------------------------------------------------
JOB_ID="${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
JOB_BASE_CONFIG="configs/base_rat_ablation_${JOB_ID}.yaml"
cp "$BASE_CONFIG" "$JOB_BASE_CONFIG"
cleanup() { rm -f "$JOB_BASE_CONFIG"; }
trap cleanup EXIT

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
    text, flags=re.MULTILINE
)
text = re.sub(
    r'^(\s*experiment_name:\s*).*$',
    r'\g<1>"$EXP_NAME"',
    text, flags=re.MULTILINE
)

path.write_text(text)
EOF
}

# ---------------------------------------------------------------------------
# Helper: write a temporary dataset config for this ablation.
# prefix = full graph folder name (intensity already embedded),
# requires_intensity: false so build_paths() uses the prefix as-is.
# ---------------------------------------------------------------------------
make_dataset_config() {
    local NAME="$1"
    local CFG="configs/datasets/rat_ablation_${NAME}_${JOB_ID}_tmp.yaml"
    cat > "$CFG" <<EOF
dataset:
  theory: "RAT"
  prefix: "HI-Small_Trans_RAT_medium__${NAME}"
  available_intensities: ["medium"]
  requires_intensity: false
EOF
    echo "$CFG"
}

# ---------------------------------------------------------------------------
# Helper: elapsed time
# ---------------------------------------------------------------------------
elapsed() { printf "%dh %dm %ds" $(($1/3600)) $((($1%3600)/60)) $(($1%60)); }

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
total_start=$(date +%s)

for NAME in "${RUN_ABLATIONS[@]}"; do
    GRAPH_DIR="graphs/HI-Small_Trans_RAT_medium__${NAME}"

    if [ ! -d "$GRAPH_DIR" ]; then
        log "ERROR: Ablation graph not found: $GRAPH_DIR"
        log "       Run create_rat_ablation_graphs_static.sh first."
        exit 1
    fi

    DATASET_CONFIG=$(make_dataset_config "$NAME")

    log ""
    log "==============================================================="
    log " ablation=$NAME"
    log " Graph dir: $GRAPH_DIR"
    log " Dataset config: $DATASET_CONFIG"
    log "==============================================================="

    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="rat_ablation_${NAME}_graphsage_t_seed${SEED}"

        log ""
        log ">>> [$(date +%H:%M:%S)] $EXP_NAME"

        patch_base_config "$SEED" "$EXP_NAME"

        t0=$(date +%s)

        python "$TRAIN_SCRIPT" \
            --config      "$MODEL_CONFIG" \
            --dataset     "$DATASET_CONFIG" \
            --base_config "$JOB_BASE_CONFIG"

        log ">>> Finished $EXP_NAME in $(elapsed $(($(date +%s) - t0)))"

    done  # seeds

    rm -f "$DATASET_CONFIG"

done  # ablations

log ""
log "==============================================================="
log " ALL DONE — total time: $(elapsed $(($(date +%s) - total_start)))"
log "==============================================================="
