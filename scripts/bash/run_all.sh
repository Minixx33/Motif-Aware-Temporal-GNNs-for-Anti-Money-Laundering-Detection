#!/bin/bash
# ============================================================
# run_all.sh — ALL MODELS x ALL DATASETS x ALL SEEDS (local)
#
# Runs: {baseline, rat_low/med/high, slt_low/med/high}
#     x {graphsage, graphsage_t, dyrep}
#     x seeds (default: 1 2 3 4 5)
#
# Windows (Git Bash) usage — recommended:
#   1. Open "Anaconda Prompt" or activate your env, OR just make sure
#      `python` on PATH has torch installed.
#   2. From the repo root:  bash scripts/bash/run_all.sh
#
# Options via environment variables:
#   SEEDS="1 2"        bash scripts/bash/run_all.sh   # custom seeds (default: 1 2 3)
#   CONDA_ENV=aml_project                             # env to activate if
#                                                     # python/torch not found
#   DATASETS_ONLY="rat slt"                           # skip baseline
# ============================================================
set -u

# ---------- project root ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
PROJECT_ROOT="$(pwd)"
echo "Running from project root: $PROJECT_ROOT"

# ---------- python / conda ----------
# If the current python already has torch, use it (typical on Windows when
# the user activated their env before running). Otherwise try conda.
if python -c "import torch" > /dev/null 2>&1; then
    echo "Using python on PATH: $(command -v python)"
else
    CONDA_ENV="${CONDA_ENV:-aml_project}"
    CONDA_BASE=""
    if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
        CONDA_BASE="$("${CONDA_EXE}" info --base)"
    elif command -v conda > /dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
    else
        for candidate in \
            "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" \
            "$HOME/AppData/Local/miniconda3" "$HOME/AppData/Local/anaconda3" \
            "/c/ProgramData/Anaconda3" "/c/ProgramData/Miniconda3" \
            "/opt/anaconda3" "/opt/miniconda3" "/opt/conda"; do
            if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
                CONDA_BASE="$candidate"; break
            fi
        done
    fi
    if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        echo "ERROR: python lacks torch and no conda found."
        echo "       Activate your environment first, then re-run this script."
        exit 1
    fi
    # conda's activate.d hook scripts often reference unset variables
    # (e.g. OCL_ICD_FILENAMES), which "set -u" would turn into fatal errors.
    # Relax strict mode just for activation.
    set +u
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || { echo "ERROR: conda activate $CONDA_ENV failed."; exit 1; }
    set -u
    echo "Activated conda env: $CONDA_ENV -> $(command -v python)"
fi
python --version

# ---------- config ----------
BASE_CONFIG="configs/base.yaml"
[ -f "$BASE_CONFIG" ] || { echo "ERROR: missing $BASE_CONFIG"; exit 1; }

SEEDS="${SEEDS:-1 2 3}"

# One line per dataset: "name|dataset_config|intensity".
# Comment a line out (put # in front) to skip that dataset.
DATASETS=(
    "baseline|configs/datasets/baseline.yaml|"
    "rat_low|configs/datasets/rat.yaml|low"
    "rat_medium|configs/datasets/rat.yaml|medium"
    "rat_high|configs/datasets/rat.yaml|high"
    "slt_low|configs/datasets/slt.yaml|low"
    "slt_medium|configs/datasets/slt.yaml|medium"
    "slt_high|configs/datasets/slt.yaml|high"
)

# One line per model: "name|training_script|model_config".
# Comment a line out (put # in front) to skip that model.
MODELS=(
    "graphsage|scripts/training/train_graphsage.py|configs/models/graphsage.yaml"
    "graphsage_t|scripts/training/train_graphsage_t.py|configs/models/graphsage_t.yaml"
    "dyrep|scripts/training/train_dyrep.py|configs/models/dyrep.yaml"
)

# ---------- logging ----------
ts=$(date +"%Y%m%d_%H%M%S")
mkdir -p logs logs/tmp_configs
LOG_FILE="logs/ALL_RUNS_${ts}.log"
FAILED_RUNS=()

echo "=================================================================" | tee -a "$LOG_FILE"
echo " ALL EXPERIMENTS: ${#DATASETS[@]} datasets x ${#MODELS[@]} models x seeds [$SEEDS]" | tee -a "$LOG_FILE"
echo " Timestamp: $ts"                                                   | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"
nvidia-smi 2>&1 | head -12 | tee -a "$LOG_FILE" || echo "No GPU detected" | tee -a "$LOG_FILE"

# ---------- per-seed base config ----------
# Seed comes from base.yaml (no --seed CLI arg), so generate a temp copy
# per seed with only the `seed:` line changed. Run dirs will be named
# seed<N>_seed1_experiment, matching the existing convention.
make_seed_config() {
    local SEED="$1"
    local OUT="logs/tmp_configs/base_seed${SEED}.yaml"
    sed "s/^  seed: .*/  seed: ${SEED}/" "$BASE_CONFIG" > "$OUT"
    echo "$OUT"
}

# ---------- runner ----------
run_one() {
    local SEED="$1" DS_ENTRY="$2" MOD_ENTRY="$3"
    local DATASET DATASET_CONFIG INTENSITY MODEL MODEL_SCRIPT MODEL_CONFIG
    IFS="|" read -r DATASET DATASET_CONFIG INTENSITY <<< "$DS_ENTRY"
    IFS="|" read -r MODEL MODEL_SCRIPT MODEL_CONFIG <<< "$MOD_ENTRY"
    local SEED_CONFIG
    SEED_CONFIG="$(make_seed_config "$SEED")"

    echo "" | tee -a "$LOG_FILE"
    echo "[ $(date +"%H:%M:%S") ] >>> seed=$SEED  $MODEL  on  $DATASET" | tee -a "$LOG_FILE"

    local t0; t0=$(date +%s)
    local ARGS=(--config "$MODEL_CONFIG" --dataset "$DATASET_CONFIG" --base_config "$SEED_CONFIG")
    [ -n "$INTENSITY" ] && ARGS+=(--intensity "$INTENSITY")

    if python "$MODEL_SCRIPT" "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        echo ">>> OK: seed=$SEED $MODEL/$DATASET ($(( $(date +%s) - t0 ))s)" | tee -a "$LOG_FILE"
    else
        echo ">>> FAILED: seed=$SEED $MODEL/$DATASET" | tee -a "$LOG_FILE"
        FAILED_RUNS+=("seed=$SEED $MODEL/$DATASET")
    fi
}

# ---------- main loop ----------
# Dataset-outer / seed-inner keeps all runs of one dataset together.
start_time=$(date +%s)
FILTER="${DATASETS_ONLY:-}"

for DS_ENTRY in "${DATASETS[@]}"; do
    DS_NAME="${DS_ENTRY%%|*}"
    if [ -n "$FILTER" ]; then
        keep=0
        for f in $FILTER; do
            case "$DS_NAME" in "$f"*) keep=1 ;; esac
        done
        [ "$keep" -eq 1 ] || continue
    fi
    for SEED in $SEEDS; do
        for MOD_ENTRY in "${MODELS[@]}"; do
            run_one "$SEED" "$DS_ENTRY" "$MOD_ENTRY"
        done
    done
done

# ---------- summary ----------
TOTAL=$(( $(date +%s) - start_time ))
echo "" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"
echo " DONE in $((TOTAL/3600))h $(((TOTAL%3600)/60))m $((TOTAL%60))s"    | tee -a "$LOG_FILE"
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo " FAILED RUNS (${#FAILED_RUNS[@]}):"                            | tee -a "$LOG_FILE"
    printf '   %s\n' "${FAILED_RUNS[@]}"                                 | tee -a "$LOG_FILE"
    exit 1
else
    echo " ALL RUNS SUCCEEDED"                                           | tee -a "$LOG_FILE"
    touch "logs/ALL_EXPERIMENTS_DONE_${ts}.done"
fi
echo " Log: $LOG_FILE"                                                   | tee -a "$LOG_FILE"
