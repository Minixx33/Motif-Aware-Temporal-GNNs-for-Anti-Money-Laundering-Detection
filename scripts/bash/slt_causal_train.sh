#!/bin/bash
# ===========================================================================
# slt_causal_train.sh
#
# GraphSAGE-T training for the causal-SLT study: 5 conditions x 5 seeds = 25
# SLURM array tasks, one (condition, seed) per task, max 3 GPUs (%3). 25 tasks
# is under the QoS GrpSubmit=30 cap, so everything can be submitted at once.
#
# Task layout (index = cond_idx*5 + (seed-1)):
#   0-4   slt_causal_1hop        5-9   slt_causal_multihop
#   10-14 selfhist_only          15-19 selfhist_plus_slt_causal
#   20-24 slt_causal_placebo
#
# Reference conditions (baseline, structural_only, slt_natural, ...) already
# have 5-seed GraphSAGE-T results from the primary experiment and are NOT
# re-run here.
#
# Resubmit only missing tasks:  sbatch --array=3,7,12%3 scripts/bash/slt_causal_train.sh
# (a run interrupted mid-training resumes from checkpoint.pt automatically)
#
#SBATCH --job-name=slt_causal_train
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --qos=gpu-long-mialhajri-001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=12:00:00
#SBATCH --array=0-24%3
#SBATCH --output=scripts/bash/logs/slt_causal_train_%A_%a.log
#SBATCH --error=scripts/bash/logs/slt_causal_train_%A_%a.err
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

CONDA_BASE=""
if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
    CONDA_BASE="$("${CONDA_EXE}" info --base)"
elif command -v conda > /dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
else
    for candidate in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" \
                     "/opt/anaconda3" "/opt/miniconda3" "/opt/conda"; do
        if [ -f "$candidate/etc/profile.d/conda.sh" ]; then CONDA_BASE="$candidate"; break; fi
    done
fi
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "ERROR: Could not locate conda."; exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-aml_project}"
echo "Using Python: $(which python)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true

CONDITIONS=(
    "slt_causal_1hop"
    "slt_causal_multihop"
    "selfhist_only"
    "selfhist_plus_slt_causal"
    "slt_causal_placebo"
)
SEEDS_ARR=(1 2 3 4 5)

MODEL_SCRIPT="scripts/training/train_graphsage_t_v2.py"
MODEL_CONFIG="configs/models/graphsage_t_v2.yaml"
BASE_CONFIG="configs/base.yaml"
for f in "$MODEL_SCRIPT" "$MODEL_CONFIG" "$BASE_CONFIG"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_COND=${#CONDITIONS[@]}
N_SEED=${#SEEDS_ARR[@]}
TOTAL=$((N_COND * N_SEED))
if [ "$TASK_ID" -ge "$TOTAL" ]; then
    echo "ERROR: task id $TASK_ID out of range (0-$((TOTAL-1)))"; exit 1
fi
COND="${CONDITIONS[$((TASK_ID / N_SEED))]}"
SEED="${SEEDS_ARR[$((TASK_ID % N_SEED))]}"
DATASET_CONFIG="configs/datasets/${COND}.yaml"
[ -f "$DATASET_CONFIG" ] || { echo "ERROR: missing $DATASET_CONFIG"; exit 1; }

echo "==============================================================="
echo " task $TASK_ID/$((TOTAL-1))  condition=$COND  seed=$SEED"
echo "==============================================================="

mkdir -p logs/tmp_configs
SEED_CONFIG="logs/tmp_configs/base_slt_causal_seed${SEED}_task${TASK_ID}.yaml"
sed "s/^  seed: .*/  seed: ${SEED}/" "$BASE_CONFIG" > "$SEED_CONFIG"

t0=$(date +%s)
python "$MODEL_SCRIPT" --config "$MODEL_CONFIG" --dataset "$DATASET_CONFIG" --base_config "$SEED_CONFIG"
echo "Done in $(( $(date +%s) - t0 ))s"
