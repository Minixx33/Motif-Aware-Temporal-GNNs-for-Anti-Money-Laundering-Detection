#!/bin/bash
# ===========================================================================
# pilot_timefix.sh -- re-pilot GraphSAGE-T and DyRep-Full after the time-
# encoding fix (fixed sin/cos encoding instead of the learnable one, which
# re-randomised its features every optimiser step and made both models
# underfit). Baseline condition, seed 1. Two array tasks = 2 GPUs in parallel
# (task 0 = GraphSAGE-T, task 1 = DyRep-Full):
#
#     sbatch --array=0-1 scripts/bash/pilot_timefix.sh
#
# The first pilots' results are kept, renamed to <model>_pilot_learnable_timeenc,
# so the new runs write to the normal v2 folders. If these look good, uncomment
# GraphSAGE-T and DyRep-Full in train_v2_all.sh and resubmit it (these two
# seed-1 baseline runs are then skipped as already done).
# ===========================================================================
#SBATCH --job-name=aml_pilot_timefix
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --qos=gpu-long-mialhajri-001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=48:00:00
#SBATCH --output=scripts/bash/logs/pilot_timefix_%A_%a.log
#SBATCH --error=scripts/bash/logs/pilot_timefix_%A_%a.err

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
CONDA_BASE=""
if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then CONDA_BASE="$("${CONDA_EXE}" info --base)"
elif command -v conda > /dev/null 2>&1; then CONDA_BASE="$(conda info --base)"
else for c in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" "/opt/anaconda3" "/opt/miniconda3" "/opt/conda"; do
    [ -f "$c/etc/profile.d/conda.sh" ] && { CONDA_BASE="$c"; break; }; done; fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-aml_project}"

R="results/HI-Small_Trans/seed1_causal_leakfix_v2"
mkdir -p logs/tmp_configs
sed "s/^  seed: .*/  seed: 1/" configs/base.yaml > logs/tmp_configs/base_pilot_timefix_seed1.yaml

# the fix must be in the CODE, not only in the configs
grep -q 'mode="fixed"' scripts/training/causal_sage.py || { echo "ERROR: causal_sage.py has no fixed time encoding -- pull the fixed code"; exit 1; }
grep -q 'cfg.get("time_encoding", "fixed")' scripts/training/train_dyrep_full.py || { echo "ERROR: train_dyrep_full.py not fixed -- pull the fixed code"; exit 1; }

MODELS=("graphsage-t|scripts/training/train_graphsage_t_v2.py|configs/models/graphsage_t_v2.yaml"
        "dyrep-full|scripts/training/train_dyrep_full.py|configs/models/dyrep_full.yaml")
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then MODELS=("${MODELS[$SLURM_ARRAY_TASK_ID]}"); fi
for M in "${MODELS[@]}"; do
    IFS='|' read -r DIR SCRIPT CFG <<< "$M"
    grep -q 'time_encoding: "fixed"' "$CFG" || { echo "ERROR: $CFG is not the fixed-time-encoding version"; exit 1; }
    # keep the first pilot (learnable time encoding) under a different name
    if [ -d "$R/$DIR" ] && [ ! -d "$R/${DIR}_pilot_learnable_timeenc" ] && ! grep -q '"time_encoding": "fixed"' "$R/$DIR/experiment_config.json" 2>/dev/null; then
        mv "$R/$DIR" "$R/${DIR}_pilot_learnable_timeenc"
        echo "kept old pilot as $R/${DIR}_pilot_learnable_timeenc"
    fi
    echo "=== $DIR (fixed time encoding) $(date)"
    python "$SCRIPT" --config "$CFG" --dataset configs/datasets/baseline.yaml \
        --base_config logs/tmp_configs/base_pilot_timefix_seed1.yaml || echo "=== $DIR FAILED"
done

echo "=== train / val / test AUPR"
for DIR in graphsage-t_pilot_learnable_timeenc graphsage-t dyrep-full_pilot_learnable_timeenc dyrep-full graphsage; do
    f="$R/$DIR/metrics.json"; [ -f "$f" ] || continue
    python -c "import json;d=json.load(open('$f'));print('%-38s'%'$DIR',{s:round(d[s]['aupr'],3) for s in ('train','val','test')})"
done
