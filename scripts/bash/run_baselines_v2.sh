#!/bin/bash
# ===========================================================================
# run_baselines_v2.sh -- non-GNN baselines (LR / RF / XGBoost / MLP) for the
# 5 primary conditions, v2 feature handling (scripts/analysis/train_baselines.py:
# ts_normalized dropped, pf/rc one-hot, val-chosen threshold, chronological
# splits enforced). One array task = one condition.
#   sbatch scripts/bash/run_baselines_v2.sh
# Output: results_baselines/v2/<dataset>_baselines.json
#
#SBATCH --job-name=aml_baselines_v2
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=14G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=scripts/bash/logs/baselines_v2_%A_%a.log
#SBATCH --error=scripts/bash/logs/baselines_v2_%A_%a.err
# ===========================================================================
set -e
set -o pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then cd "$SLURM_SUBMIT_DIR"; else cd "$(dirname "$0")/../.."; fi
CONDA_BASE=""
if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then CONDA_BASE="$("${CONDA_EXE}" info --base)"
elif command -v conda > /dev/null 2>&1; then CONDA_BASE="$(conda info --base)"
else for c in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" "/opt/anaconda3" "/opt/miniconda3" "/opt/conda"; do
    [ -f "$c/etc/profile.d/conda.sh" ] && { CONDA_BASE="$c"; break; }; done; fi
[ -n "$CONDA_BASE" ] || { echo "ERROR: conda not found"; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-aml_project}"

DATASETS=(
    "HI-Small_Trans"
    "HI-Small_Trans_RAT_pristine_structural_only"
    "HI-Small_Trans_RAT_pristine"
    "HI-Small_Trans_SLT_pristine"
    "HI-Small_Trans_SLT_pristine_plus_structural"
)
D="${DATASETS[${SLURM_ARRAY_TASK_ID:-0}]}"
case "$D" in *SLT*) [ -f logs/markers/slt_rebuild.done ] || { echo "ERROR: run slt_rebuild_prep.sh first"; exit 1; } ;; esac
mkdir -p results_baselines/v2
echo "Baselines for $D"
python scripts/analysis/train_baselines.py --graph_dir "graphs/$D" --split_dir "splits/$D" \
    --output_json "results_baselines/v2/${D}_baselines.json" --seed 42
