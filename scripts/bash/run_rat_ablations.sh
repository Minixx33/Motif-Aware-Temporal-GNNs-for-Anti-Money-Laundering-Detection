#!/bin/bash
# ===========================================================================
# run_rat_ablations.sh
#
# Runs GraphSAGE-T training with 5 seeds for every RAT static ablation graph.
# Ablation graphs must already exist under graphs/HI-Small_Trans_RAT_medium__<name>/
# (build them first with: bash scripts/bash/create_rat_ablation_graphs_static.sh)
#
# WINDOWS (Git Bash): open Git Bash, cd to project root, then:
#   bash scripts/bash/run_rat_ablations.sh
#
# LINUX/MAC:  bash scripts/bash/run_rat_ablations.sh
# SLURM/AWS:  sbatch scripts/bash/run_rat_ablations.sh
#
# SLURM array layout (task ID -> ablation):
#   0: no_struct        3: no_entity        6: no_motif
#   1: no_temp          4: no_rat_scores    7: no_crossbank
#   2: no_amount        5: no_burst_pattern 8: top20_features
#
# SLURM directives -- ignored when run with bash directly:
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
# Works on: Windows (Git Bash), Linux, macOS, SLURM/AWS
# Override env name with: CONDA_ENV=my_env bash run_rat_ablations.sh
# ---------------------------------------------------------------------------
CONDA_BASE=""
if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
    CONDA_BASE="$("${CONDA_EXE}" info --base)"
elif command -v conda > /dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
else
    for candidate in \
        "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" \
        "$HOME/AppData/Local/anaconda3" \
        "$HOME/AppData/Local/miniconda3" \
        "$HOME/AppData/Local/miniforge3" \
        "/c/Users/$USERNAME/AppData/Local/anaconda3" \
        "/c/Users/$USERNAME/AppData/Local/miniconda3" \
        "/c/Users/$USERNAME/anaconda3" \
        "/c/Users/$USERNAME/miniconda3" \
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
log " RAT ABLATION TRAINING -- GraphSAGE-T, 5 seeds"
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
#   SLURM array job -> single ablation from SLURM_ARRAY_TASK_ID
#   Local run       -> all 9 ablations sequentially
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

    python - "$JOB_BASE_CONFIG" "$SEED" "$EXP_NAME" <<'PYEOF'
import sys, re
from pathlib import Path

cfg_path, seed, exp_name = sys.argv[1], sys.argv[2], sys.argv[3]
text = Path(cfg_path).read_text()
text = re.sub(r'^(\s*seed:\s*).*$',            rf'\g<1>{seed}',       text, flags=re.MULTILINE)
text = re.sub(r'^(\s*experiment_name:\s*).*$', rf'\g<1>"{exp_name}"', text, flags=re.MULTILINE)
Path(cfg_path).write_text(text)
PYEOF
}

# ---------------------------------------------------------------------------
# Helper: write a temporary dataset config for this ablation.
# prefix = full graph folder name (intensity already embedded),
# requires_intensity: false so build_paths() uses the prefix as-is
# -> resolves to graphs/HI-Small_Trans_RAT_medium__<name>
# ---------------------------------------------------------------------------
make_dataset_config() {
    local NAME="$1"
    local CFG="configs/datasets/rat_ablation_${NAME}_${JOB_ID}_tmp.yaml"
    cat > "$CFG" <<YAMLEOF
dataset:
  theory: "RAT"
  prefix: "HI-Small_Trans_RAT_medium__${NAME}"
  available_intensities: ["medium"]
  requires_intensity: false
YAMLEOF
    echo "$CFG"
}

# ---------------------------------------------------------------------------
# Helper: elapsed time
# ---------------------------------------------------------------------------
elapsed() {
    local t=$1
    printf "%dh %dm %ds" $((t/3600)) $(((t%3600)/60)) $((t%60))
}

# ---------------------------------------------------------------------------
# MAIN
# Seed-outer / ablation-inner: one complete seed (all 9 ablations) finishes
# before the next seed starts, so a full single-seed result set is available
# as early as possible.
# ---------------------------------------------------------------------------
total_start=$(date +%s)
FAILED_RUNS=()

# Pre-flight: make sure every ablation graph this run needs already exists,
# so we fail fast instead of partway through seed 1.
for NAME in "${RUN_ABLATIONS[@]}"; do
    GRAPH_DIR="graphs/HI-Small_Trans_RAT_medium__${NAME}"
    if [ ! -d "$GRAPH_DIR" ]; then
        log "ERROR: Ablation graph not found: $GRAPH_DIR"
        log "       Run create_rat_ablation_graphs_static.sh first."
        exit 1
    fi
done

for SEED in "${SEEDS[@]}"; do
    log ""
    log "################ SEED $SEED: full RAT ablation sweep ################"

    for NAME in "${RUN_ABLATIONS[@]}"; do
        GRAPH_DIR="graphs/HI-Small_Trans_RAT_medium__${NAME}"
        DATASET_CONFIG=$(make_dataset_config "$NAME")
        EXP_NAME="rat_ablation_${NAME}_graphsage_t_seed${SEED}"

        log ""
        log "==============================================================="
        log " seed=$SEED  ablation=$NAME"
        log " Graph dir: $GRAPH_DIR"
        log " Dataset config: $DATASET_CONFIG"
        log "==============================================================="

        patch_base_config "$SEED" "$EXP_NAME"

        t0=$(date +%s)

        if python "$TRAIN_SCRIPT" \
            --config      "$MODEL_CONFIG" \
            --dataset     "$DATASET_CONFIG" \
            --base_config "$JOB_BASE_CONFIG"; then
            log ">>> OK: $EXP_NAME finished in $(elapsed $(($(date +%s) - t0)))"
        else
            log ">>> FAILED: $EXP_NAME"
            FAILED_RUNS+=("seed=$SEED ablation=$NAME")
        fi

        rm -f "$DATASET_CONFIG"

    done  # ablations
done  # seeds

log ""
log "==============================================================="
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    log " FAILED RUNS (${#FAILED_RUNS[@]}):"
    for r in "${FAILED_RUNS[@]}"; do
        log "   $r"
    done
fi
log " ALL DONE -- total time: $(elapsed $(($(date +%s) - total_start)))"
log "==============================================================="
