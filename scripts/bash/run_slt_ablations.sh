#!/bin/bash
# ===========================================================================
# run_slt_ablations.sh
#
# Runs GraphSAGE-T training with 5 seeds for every SLT ablation variant
# across all 3 intensities (low / medium / high).
#
# LOCAL:  bash run_slt_ablations.sh   (runs all 75 combos sequentially)
# SLURM:  sbatch run_slt_ablations.sh (15 parallel GPU jobs, one per
#                                       variant×intensity; 5 seeds per job)
#
# SLURM array layout (task ID → variant, intensity):
#   task = variant_idx * 3 + intensity_idx
#   variants  (0-4): current, equal, neighbor_heavy, amount_heavy, temporal_heavy
#   intensities (0-2): low, medium, high
#
# SLURM directives — ignored when run with bash directly:
#SBATCH --job-name=slt_ablations_train
#SBATCH --array=0-14
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=scripts/bash/logs/slt_ablations_%A_%a.log
#SBATCH --error=scripts/bash/logs/slt_ablations_%A_%a.err
# ===========================================================================
set -e
set -o pipefail

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
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
conda activate "${CONDA_ENV:-aml_project}"

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
    LOG_FILE="scripts/bash/logs/slt_ablations_training_${ts}.log"
    log() { echo "$@" | tee -a "$LOG_FILE"; }
else
    log() { echo "$@"; }
fi

log "==============================================================="
log " SLT ABLATION TRAINING — GraphSAGE-T, 5 seeds"
log " Host: $(hostname)  PID: $$"
log " SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
log " SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-none (local run)}"
log "==============================================================="

log ""
log ">>> GPU INFO:"
nvidia-smi 2>&1 || log "GPU info unavailable"

# ---------------------------------------------------------------------------
# Variant / intensity tables
# Index must match SLURM array task ID mapping:
#   task_id = variant_idx * 3 + intensity_idx
# ---------------------------------------------------------------------------
VARIANTS=(
    "current"        # 0
    "equal"          # 1
    "neighbor_heavy" # 2
    "amount_heavy"   # 3
    "temporal_heavy" # 4
)
INTENSITIES=("low" "medium" "high")   # 0 1 2

SEEDS=(1 2 3 4 5)

# ---------------------------------------------------------------------------
# Select which (variant, intensity) pairs to run:
#   SLURM array job → single pair from SLURM_ARRAY_TASK_ID
#   Local run       → all 15 pairs sequentially
# ---------------------------------------------------------------------------
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    TASK_ID=$SLURM_ARRAY_TASK_ID
    VARIANT_IDX=$(( TASK_ID / 3 ))
    INTENSITY_IDX=$(( TASK_ID % 3 ))
    RUN_PAIRS=("${VARIANTS[$VARIANT_IDX]} ${INTENSITIES[$INTENSITY_IDX]}")
else
    RUN_PAIRS=()
    for v_idx in "${!VARIANTS[@]}"; do
        for i_idx in "${!INTENSITIES[@]}"; do
            RUN_PAIRS+=("${VARIANTS[$v_idx]} ${INTENSITIES[$i_idx]}")
        done
    done
fi

# ---------------------------------------------------------------------------
# Helper: patch a job-specific copy of base.yaml
# Using a per-job temp file avoids race conditions when SLURM runs
# multiple array tasks in parallel against the same repo.
# ---------------------------------------------------------------------------
JOB_ID="${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
JOB_BASE_CONFIG="configs/base_slt_ablation_${JOB_ID}.yaml"
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
# Helper: write a temporary dataset config for this variant
# ---------------------------------------------------------------------------
make_dataset_config() {
    local VARIANT="$1"
    local CFG="configs/datasets/slt_${VARIANT}_${JOB_ID}_tmp.yaml"
    cat > "$CFG" <<EOF
dataset:
  theory: "SLT"
  prefix: "HI-Small_Trans_SLT_${VARIANT}"
  available_intensities: ["low", "medium", "high"]
  requires_intensity: true
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

for PAIR in "${RUN_PAIRS[@]}"; do
    read -r VARIANT INTENSITY <<< "$PAIR"

    DATASET_CONFIG=$(make_dataset_config "$VARIANT")

    log ""
    log "==============================================================="
    log " variant=$VARIANT  intensity=$INTENSITY"
    log " Dataset config: $DATASET_CONFIG"
    log "==============================================================="

    for SEED in "${SEEDS[@]}"; do
        EXP_NAME="slt_${VARIANT}_${INTENSITY}_graphsage_t_seed${SEED}"

        log ""
        log ">>> [$(date +%H:%M:%S)] $EXP_NAME"

        patch_base_config "$SEED" "$EXP_NAME"

        t0=$(date +%s)

        python "$TRAIN_SCRIPT" \
            --config      "$MODEL_CONFIG" \
            --dataset     "$DATASET_CONFIG" \
            --base_config "$JOB_BASE_CONFIG" \
            --intensity   "$INTENSITY"

        log ">>> Finished $EXP_NAME in $(elapsed $(($(date +%s) - t0)))"

    done  # seeds

    rm -f "$DATASET_CONFIG"

done  # variant × intensity pairs

log ""
log "==============================================================="
log " ALL DONE — total time: $(elapsed $(($(date +%s) - total_start)))"
log "==============================================================="
