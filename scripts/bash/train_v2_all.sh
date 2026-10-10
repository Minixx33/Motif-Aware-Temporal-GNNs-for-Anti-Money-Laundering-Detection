#!/bin/bash
# ===========================================================================
# train_v2_all.sh -- all causal_leakfix_v2 base runs, ONE command:
#
#     sbatch --array=0-2 scripts/bash/train_v2_all.sh
#
# 3 array tasks = 3 GPUs = 3 workers. The combo list (DATASETS x MODELS x
# SEEDS below) is split between the workers; each worker trains its share
# one after another. Never more than 3 GPUs or 3 queued jobs, so the
# GrpSubmit=30 limit is never hit.
#
# Edit the lists below to choose what runs (comment a line out = skip it).
# Safe to resubmit the same command any time:
#   - runs with metrics.json are skipped (already done)
#   - interrupted runs resume from their checkpoint.pt
#   - a run currently being trained by another live job is skipped (lock file)
# Results: results/<dataset>/seed<N>_causal_leakfix_v2/<model>/
# ===========================================================================
#SBATCH --job-name=aml_v2
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --qos=gpu-long-mialhajri-001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=500:00:00
#SBATCH --output=scripts/bash/logs/aml_v2_%A_%a.log
#SBATCH --error=scripts/bash/logs/aml_v2_%A_%a.err

DATASETS_ARR=(
    "configs/datasets/baseline.yaml"
    "configs/datasets/structural_only.yaml"
    "configs/datasets/rat_natural.yaml"
    "configs/datasets/slt_natural.yaml"
    "configs/datasets/slt_plus_structural.yaml"
)
# name | script | config | results folder name
MODELS_ARR=(
    "graphsage|scripts/training/train_graphsage_v2.py|configs/models/graphsage_v2.yaml|graphsage"            # done (25/25)
    "dyrep_lite|scripts/training/train_dyrep_lite_v2.py|configs/models/dyrep_lite_v2.yaml|dyrep"             # done (25/25)
    "graphsage_t|scripts/training/train_graphsage_t_v2.py|configs/models/graphsage_t_v2.yaml|graphsage-t"   # fixed time encoding
    # "dyrep_full|scripts/training/train_dyrep_full.py|configs/models/dyrep_full.yaml|dyrep-full"            # ON HOLD
)
SEEDS_ARR=(1 2 3 4 5)
EXP_NAME="causal_leakfix_v2"

# ---------------------------------------------------------------------------
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
export PATH=$PATH:/opt/slurm/bin
CONDA_BASE=""
if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then CONDA_BASE="$("${CONDA_EXE}" info --base)"
elif command -v conda > /dev/null 2>&1; then CONDA_BASE="$(conda info --base)"
else for c in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" "/opt/anaconda3" "/opt/miniconda3" "/opt/conda"; do
    [ -f "$c/etc/profile.d/conda.sh" ] && { CONDA_BASE="$c"; break; }; done; fi
[ -n "$CONDA_BASE" ] || { echo "ERROR: conda not found"; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-aml_project}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

WORKER="${SLURM_ARRAY_TASK_ID:-0}"
N_WORKERS="${SLURM_ARRAY_TASK_COUNT:-1}"
MY_JOB="${SLURM_JOB_ID:-local$$}"
mkdir -p logs/tmp_configs

# GraphSAGE-T / DyRep-Full need the fixed time encoding in the CODE
grep -q 'mode="fixed"' scripts/training/causal_sage.py || { echo "ERROR: causal_sage.py lacks the fixed time encoding -- git pull"; exit 1; }
grep -q 'cfg.get("time_encoding", "fixed")' scripts/training/train_dyrep_full.py || { echo "ERROR: train_dyrep_full.py lacks the fix -- git pull"; exit 1; }
# Results trained with the old learnable time encoding are not valid: move them
# aside (once, by worker 0) so they are re-trained instead of skipped as "done".
if [ "$WORKER" -eq 0 ]; then
    for d in results/*/seed*_${EXP_NAME}/graphsage-t; do
        [ -d "$d" ] || continue
        grep -q '"time_encoding": "fixed"' "$d/experiment_config.json" 2>/dev/null && continue
        [ -f "$d/.lock" ] && continue
        mv "$d" "${d}_old_learnable_timeenc_$(date +%Y%m%d%H%M%S)" && echo "moved old run aside: $d"
    done
else
    sleep 60   # let worker 0 finish moving old results first
fi

COMBOS=()
for ds in "${DATASETS_ARR[@]}"; do
    for mod in "${MODELS_ARR[@]}"; do
        for seed in "${SEEDS_ARR[@]}"; do COMBOS+=("$ds##$mod##$seed"); done
    done
done
echo "Worker $WORKER of $N_WORKERS -- ${#COMBOS[@]} combos in total"

for i in "${!COMBOS[@]}"; do
    [ $((i % N_WORKERS)) -eq "$WORKER" ] || continue
    DS="${COMBOS[$i]%%##*}"; REST="${COMBOS[$i]#*##}"; MOD="${REST%%##*}"; SEED="${REST#*##}"
    IFS='|' read -r NAME SCRIPT CFG DISK <<< "$MOD"
    PREFIX="$(grep -m1 'prefix:' "$DS" | sed -E 's/.*prefix:[[:space:]]*"([^"]*)".*/\1/')"
    RDIR="results/$PREFIX/seed${SEED}_${EXP_NAME}/$DISK"
    TAG="[$i] $PREFIX $NAME seed$SEED"

    if [ -f "$RDIR/metrics.json" ]; then echo "$TAG: done already, skip"; continue; fi
    case "$PREFIX" in *SLT*) [ -f logs/markers/slt_rebuild.done ] || { echo "$TAG: SLT rebuild not done, skip"; continue; } ;; esac
    mkdir -p "$RDIR"
    if [ -f "$RDIR/.lock" ]; then
        OTHER="$(cat "$RDIR/.lock")"
        if squeue -h -j "$OTHER" 2>/dev/null | grep -q .; then echo "$TAG: being trained by job $OTHER, skip"; continue; fi
    fi
    echo "$MY_JOB" > "$RDIR/.lock"

    SEED_CFG="logs/tmp_configs/base_v2all_seed${SEED}_${MY_JOB}.yaml"
    sed "s/^  seed: .*/  seed: ${SEED}/" configs/base.yaml > "$SEED_CFG"
    echo "$TAG: START $(date)"
    if python "$SCRIPT" --config "$CFG" --dataset "$DS" --base_config "$SEED_CFG"; then
        echo "$TAG: OK $(date)"
    else
        echo "$TAG: FAILED (exit $?) -- resubmit later to resume from checkpoint"
    fi
    rm -f "$RDIR/.lock"
done
echo "Worker $WORKER finished $(date)"
