#!/bin/bash
# ===========================================================================
# train_condition_group.sh
#
# Thin SLURM wrapper around run_all.sh (which has no SBATCH header of its
# own) -- trains whatever DATASETS_ONLY/MODELS_ONLY/SEEDS you pass as env
# vars, on ONE GPU. Submit this 3 times with disjoint DATASETS_ONLY values
# to use all 3 available GPUs in parallel without exceeding the cap.
#
# Usage:
#   sbatch --job-name=aml_j1 --dependency=afterok:<prep_jobid> \
#       --export=DATASETS_ONLY="baseline structural_only" \
#       scripts/bash/train_condition_group.sh
#
# Or just run submit_full_pipeline.sh, which does this for you.
#
# SLURM directives -- ignored when run with bash directly:
#SBATCH --job-name=aml_train_group
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --qos=gpu-long-mialhajri-001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=scripts/bash/logs/train_group_%j.log
#SBATCH --error=scripts/bash/logs/train_group_%j.err
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

: "${DATASETS_ONLY:?Set DATASETS_ONLY, e.g. --export=DATASETS_ONLY=\"baseline structural_only\"}"
: "${MODELS_ONLY:=graphsage graphsage_t dyrep}"
: "${SEEDS:=1 2 3}"

echo "DATASETS_ONLY=$DATASETS_ONLY"
echo "MODELS_ONLY=$MODELS_ONLY"
echo "SEEDS=$SEEDS"

DATASETS_ONLY="$DATASETS_ONLY" MODELS_ONLY="$MODELS_ONLY" SEEDS="$SEEDS" bash scripts/bash/run_all.sh
