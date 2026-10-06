#!/bin/bash
# ===========================================================================
# slt_causal_baselines.sh
#
# Feature-only (non-graph) diagnostic for the causal-SLT study: logistic
# regression + gradient boosting on baseline + ONE feature group at a time
# (SELF control, PEER1, PEER1+MULTI, SELF+peer, placebo, natural SLT, and a
# reporting-lag sensitivity sweep). Needs only the baseline graph/split and
# the existing slt_natural graph -- it does NOT depend on slt_causal_prep.sh,
# so it can run in parallel with it. CPU only (no GPU used).
#
# SLURM:  sbatch scripts/bash/slt_causal_baselines.sh
#
#SBATCH --job-name=slt_causal_baselines
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=14G
#SBATCH --time=12:00:00
#SBATCH --output=scripts/bash/logs/slt_causal_baselines_%j.log
#SBATCH --error=scripts/bash/logs/slt_causal_baselines_%j.err
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

mkdir -p scripts/bash/logs results_baselines/v2
for p in graphs/HI-Small_Trans splits/HI-Small_Trans; do
    [ -d "$p" ] || { echo "ERROR: missing $p"; exit 1; }
done

python scripts/analysis/slt_causal_baselines.py \
    --output_json results_baselines/v2/slt_causal_baselines.json
echo "BASELINES COMPLETE"
