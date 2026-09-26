#!/bin/bash
# ===========================================================================
# train_single_run.sh
#
# SLURM JOB ARRAY: one array task = exactly ONE (dataset, model, seed)
# training run. 5 datasets x 3 models x 5 seeds = 75 tasks (indices 0-74).
#
# WHY an array instead of the old 3-big-job split (train_condition_group.sh):
# a single run takes multiple days (per your own local timing), so bundling
# up to 18 sequential runs into one job with --time=48:00:00 was never going
# to finish -- that job needed weeks, not 2 days. Splitting to one run per
# job fixes the time-limit sizing problem (each job's --time only has to
# cover ONE run, not eighteen).
#
# WHY --array=...%3 specifically: your 3-GPU cap is self-enforced, not a
# QoS/account limit SLURM applies for you. `%3` is a SLURM-native throttle --
# it caps how many array tasks run concurrently regardless of QoS, so it's
# what actually keeps you at <=3 GPUs at once. The other 72 tasks just sit
# PENDING in the queue and start automatically as running ones finish. No
# manual job-launching or babysitting needed.
#
# --time=500:00:00 matches the convention already used in
# run_rat_ablations.sh / run_slt_ablations.sh. "Multiple days per run" is a
# rough estimate, so this is deliberately generous -- a job that finishes
# early just ends early, it doesn't get penalized for the unused allocation.
#
# Usage:
#   sbatch --dependency=afterok:<prep_jobid> scripts/bash/train_single_run.sh
#
# To test with just the first few combos before committing all 75:
#   sbatch --array=0-2 scripts/bash/train_single_run.sh
#
# To resubmit only failed/missing combos later, pass an explicit index list:
#   sbatch --array=7,12,29 scripts/bash/train_single_run.sh
#
# Or just run submit_full_pipeline.sh, which submits this for you after prep.
#
# SLURM directives -- ignored when run with bash directly:
#SBATCH --job-name=aml_train
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --qos=gpu-long-mialhajri-001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=500:00:00
#SBATCH --array=0-29%3
#SBATCH --output=scripts/bash/logs/train_%A_%a.log
#SBATCH --error=scripts/bash/logs/train_%A_%a.err
# ===========================================================================
set -e
set -o pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    cd "$SCRIPT_DIR/../.."
fi
echo "Project root: $(pwd)"

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
        "/opt/anaconda3" "/opt/miniconda3" "/opt/conda"; do
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
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true

# ---------------------------------------------------------------------------
# The 45 combos -- kept as a local, exact list (NOT run_all.sh's
# DATASETS_ONLY/MODELS_ONLY string-prefix filter, which is ambiguous here:
# MODELS_ONLY="graphsage" would match BOTH "graphsage" and "graphsage_t"
# since it's a prefix match. We need exactly one model per task, so we
# build the combo list ourselves and call each training script directly,
# the same way run_all.sh's run_one() does internally.
# ---------------------------------------------------------------------------
DATASETS_ARR=(
    "baseline|configs/datasets/baseline.yaml|"
    "structural_only|configs/datasets/structural_only.yaml|"
    "rat_natural|configs/datasets/rat_natural.yaml|"
    "slt_natural|configs/datasets/slt_natural.yaml|"
    "slt_plus_structural|configs/datasets/slt_plus_structural.yaml|"
)
MODELS_ARR=(
    "graphsage_t|scripts/training/train_graphsage_t.py|configs/models/graphsage_t.yaml"
    "graphsage|scripts/training/train_graphsage.py|configs/models/graphsage.yaml"
    "dyrep|scripts/training/train_dyrep.py|configs/models/dyrep.yaml"
)
# Override with e.g. --export=ALL,SEEDS_LIST="4 5" to generate a combo list
# for only specific seeds (e.g. adding seeds 4-5 on top of an earlier 1-3
# run, without re-running or duplicating 1-3). Remember to also override
# --array to match the new combo count (5 datasets x 3 models x N seeds),
# e.g. --array=0-29%1 for a 2-seed, 30-combo list.
if [ -n "${SEEDS_LIST:-}" ]; then
    read -ra SEEDS_ARR <<< "$SEEDS_LIST"
else
    SEEDS_ARR=(4 5)   # seeds 1-3 already done/running under the old job split
fi

COMBOS=()
for ds in "${DATASETS_ARR[@]}"; do
    for mod in "${MODELS_ARR[@]}"; do
        for seed in "${SEEDS_ARR[@]}"; do
            COMBOS+=("$ds##$mod##$seed")
        done
    done
done
echo "Total combos: ${#COMBOS[@]} (expect 75)"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -ge "${#COMBOS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$TASK_ID out of range (0-$((${#COMBOS[@]}-1)))."
    exit 1
fi

COMBO="${COMBOS[$TASK_ID]}"
DS_ENTRY="${COMBO%%##*}"
REST="${COMBO#*##}"
MOD_ENTRY="${REST%%##*}"
SEED="${REST#*##}"

IFS='|' read -r DATASET DATASET_CONFIG INTENSITY <<< "$DS_ENTRY"
IFS='|' read -r MODEL MODEL_SCRIPT MODEL_CONFIG <<< "$MOD_ENTRY"

echo "==============================================================="
echo " Array task $TASK_ID / $((${#COMBOS[@]}-1))"
echo " dataset=$DATASET  model=$MODEL  seed=$SEED"
echo "==============================================================="

BASE_CONFIG="configs/base.yaml"
[ -f "$BASE_CONFIG" ] || { echo "ERROR: missing $BASE_CONFIG"; exit 1; }

mkdir -p logs logs/tmp_configs
SEED_CONFIG="logs/tmp_configs/base_seed${SEED}_task${TASK_ID}.yaml"
sed "s/^  seed: .*/  seed: ${SEED}/" "$BASE_CONFIG" > "$SEED_CONFIG"

ARGS=(--config "$MODEL_CONFIG" --dataset "$DATASET_CONFIG" --base_config "$SEED_CONFIG")
[ -n "$INTENSITY" ] && ARGS+=(--intensity "$INTENSITY")

echo "Running: python $MODEL_SCRIPT ${ARGS[*]}"
t0=$(date +%s)
python "$MODEL_SCRIPT" "${ARGS[@]}"
echo "Done in $(( $(date +%s) - t0 ))s"
