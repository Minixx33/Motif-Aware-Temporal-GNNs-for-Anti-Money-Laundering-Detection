#!/bin/bash
# ===========================================================================
# run_slt_ablations.sh
#
# Runs GraphSAGE-T training with 5 seeds for every SLT ablation variant
# across all 3 intensities (low / medium / high).
#
# LOCAL:  bash run_slt_ablations.sh   (runs all 75 combos sequentially)
# SLURM:  sbatch run_slt_ablations.sh (5 parallel GPU jobs, one per
#                                       SLT variant at medium intensity;
#                                       5 seeds per job)
#
# SLURM array layout (task ID → variant, intensity):
#   task = variant_idx * 3 + intensity_idx
#   variants  (0-4): current, equal, neighbor_heavy, amount_heavy, temporal_heavy
#   intensity: medium
#
# SLURM directives — ignored when run with bash directly:
#SBATCH --job-name=slt_ablations_train
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --qos=gpu-long-mialhajri-001
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=500:00:00
#SBATCH --output=scripts/bash/logs/slt_ablations_%A_%a.log
#SBATCH --error=scripts/bash/logs/slt_ablations_%A_%a.err
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
INTENSITIES=("medium")   # 0 1 2

SEEDS="${SEEDS:-1 2 3 4 5}"   # override: SEEDS="1 2 3" bash ...

# ---------------------------------------------------------------------------
# Select which (variant, intensity) pairs to run:
#   SLURM array job → single pair from SLURM_ARRAY_TASK_ID
#   Local run       → all 15 pairs sequentially
# ---------------------------------------------------------------------------
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    TASK_ID=$SLURM_ARRAY_TASK_ID
    RUN_PAIRS=("${VARIANTS[$TASK_ID]} medium")
else
    RUN_PAIRS=()
    for v_idx in "${!VARIANTS[@]}"; do
        RUN_PAIRS+=("${VARIANTS[$v_idx]} medium")
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
    # build_paths() resolves dataset_name as "${prefix}_${intensity}" whenever
    # requires_intensity is true. The "current" variant's graphs were built
    # WITHOUT a "_current" suffix (graphs/HI-Small_Trans_SLT_medium, same as
    # the main non-ablation dataset) -- see create_slt_ablation_variants.sh.
    # All other variants DO carry the suffix (graphs/HI-Small_Trans_SLT_<variant>_medium).
    # So "current" needs the bare prefix; everything else needs "_<variant>".
    local PREFIX
    if [ "$VARIANT" = "current" ]; then
        PREFIX="HI-Small_Trans_SLT"
    else
        PREFIX="HI-Small_Trans_SLT_${VARIANT}"
    fi
    cat > "$CFG" <<EOF
dataset:
  theory: "SLT"
  prefix: "${PREFIX}"
  available_intensities: ["medium"]
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
# Seed-outer / variant-inner: one complete seed (all variants) finishes
# before the next seed starts, so a full single-seed result set is available
# as early as possible.
# ---------------------------------------------------------------------------
total_start=$(date +%s)
FAILED_RUNS=()

for SEED in $SEEDS; do
    log ""
    log "################ SEED $SEED: full SLT ablation sweep ################"

    for PAIR in "${RUN_PAIRS[@]}"; do
        read -r VARIANT INTENSITY <<< "$PAIR"

        DATASET_CONFIG=$(make_dataset_config "$VARIANT")
        EXP_NAME="slt_${VARIANT}_${INTENSITY}_graphsage_t_seed${SEED}"

        log ""
        log "==============================================================="
        log " seed=$SEED  variant=$VARIANT  intensity=$INTENSITY"
        log " Dataset config: $DATASET_CONFIG"
        log "==============================================================="

        patch_base_config "$SEED" "$EXP_NAME"

        t0=$(date +%s)

        if python "$TRAIN_SCRIPT" \
            --config      "$MODEL_CONFIG" \
            --dataset     "$DATASET_CONFIG" \
            --base_config "$JOB_BASE_CONFIG" \
            --intensity   "$INTENSITY"; then
            log ">>> OK: $EXP_NAME finished in $(elapsed $(($(date +%s) - t0)))"
        else
            log ">>> FAILED: $EXP_NAME"
            FAILED_RUNS+=("seed=$SEED variant=$VARIANT")
        fi

        rm -f "$DATASET_CONFIG"

    done  # variant × intensity pairs
done  # seeds

log ""
log "==============================================================="
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    log " FAILED RUNS (${#FAILED_RUNS[@]}):"
    for r in "${FAILED_RUNS[@]}"; do
        log "   $r"
    done
fi
log " ALL DONE — total time: $(elapsed $(($(date +%s) - total_start)))"
log "==============================================================="
