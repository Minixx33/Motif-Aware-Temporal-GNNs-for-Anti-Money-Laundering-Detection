#!/bin/bash
# ===========================================================================
# train_single_run_v2.sh
#
# causal_leakfix_v2 runs of the FIXED GraphSAGE / GraphSAGE-T
# (scripts/training/train_graphsage_v2.py, train_graphsage_t_v2.py; shared
# code in scripts/training/causal_sage.py). One array task = ONE
# (dataset, model, seed) run: 5 datasets x 2 models x 5 seeds = 50 tasks
# (indices 0-49). Results: results/<dataset>/seed<N>_causal_leakfix_v2/<model>/
# v1 scripts/results are untouched.
#
# GrpSubmit=30 on the GPU QoS counts pending array tasks individually, so
# submit in batches of <=25, e.g.:
#   sbatch --array=0-24%3  scripts/bash/train_single_run_v2.sh
#   sbatch --array=25-49%3 scripts/bash/train_single_run_v2.sh   (after the first batch drains)
#
# Single test run first (baseline, graphsage_t, seed 1 = index 0):
#   sbatch --array=0 scripts/bash/train_single_run_v2.sh
#
# --time=12:00:00: v2 does 2-hop causal neighbour sampling per edge, so an
# epoch costs more than v1; checkpoint.pt resume covers a timeout (resubmit
# the same index).
#
#SBATCH --job-name=aml_train_v2
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --qos=gpu-long-mialhajri-001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-24%3
#SBATCH --output=scripts/bash/logs/train_v2_%A_%a.log
#SBATCH --error=scripts/bash/logs/train_v2_%A_%a.err
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
    "graphsage_t_v2|scripts/training/train_graphsage_t_v2.py|configs/models/graphsage_t_v2.yaml"
    "graphsage_v2|scripts/training/train_graphsage_v2.py|configs/models/graphsage_v2.yaml"
)
# Override with e.g. --export=ALL,SEEDS_LIST="4 5" to generate a combo list
# for only specific seeds (e.g. adding seeds 4-5 on top of an earlier 1-3
# run, without re-running or duplicating 1-3). Remember to also override
# --array to match the new combo count (5 datasets x 3 models x N seeds),
# e.g. --array=0-29%1 for a 2-seed, 30-combo list.
if [ -n "${SEEDS_LIST:-}" ]; then
    read -ra SEEDS_ARR <<< "$SEEDS_LIST"
else
    SEEDS_ARR=(1 2 3 4 5)
fi

COMBOS=()
for ds in "${DATASETS_ARR[@]}"; do
    for mod in "${MODELS_ARR[@]}"; do
        for seed in "${SEEDS_ARR[@]}"; do
            COMBOS+=("$ds##$mod##$seed")
        done
    done
done
echo "Total combos: ${#COMBOS[@]} (expect 50)"

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
SEED_CONFIG="logs/tmp_configs/base_v2_seed${SEED}_task${TASK_ID}.yaml"
sed "s/^  seed: .*/  seed: ${SEED}/" "$BASE_CONFIG" > "$SEED_CONFIG"

ARGS=(--config "$MODEL_CONFIG" --dataset "$DATASET_CONFIG" --base_config "$SEED_CONFIG")
[ -n "$INTENSITY" ] && ARGS+=(--intensity "$INTENSITY")

echo "Running: python $MODEL_SCRIPT ${ARGS[*]}"
t0=$(date +%s)
python "$MODEL_SCRIPT" "${ARGS[@]}"
echo "Done in $(( $(date +%s) - t0 ))s"
