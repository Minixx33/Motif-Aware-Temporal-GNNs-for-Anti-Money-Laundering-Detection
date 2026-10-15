#!/bin/bash
# ===========================================================================
# train_v2_ablations.sh -- causal_leakfix_v2 ablation runs, ONE command:
#
#     sbatch --array=0-2 scripts/bash/train_v2_ablations.sh
#
# Same mechanics as train_v2_all.sh: 3 array tasks = 3 GPUs = 3 workers that
# split the DATASETS x MODELS x SEEDS list. Edit the lists (comment a line out
# = skip it). Safe to resubmit: finished runs (metrics.json) are skipped,
# interrupted runs resume from checkpoint.pt, runs locked by a live job are skipped.
# Reference conditions (baseline, structural_only, rat_natural, slt_natural, ...)
# come from train_v2_all.sh and are NOT re-run here.
# Results: results/<dataset>/seed<N>_causal_leakfix_v2/<model>/
# ===========================================================================
#SBATCH --job-name=aml_v2_abl
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --qos=gpu-long-mialhajri-001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=500:00:00
#SBATCH --output=scripts/bash/logs/aml_v2_abl_%A_%a.log
#SBATCH --error=scripts/bash/logs/aml_v2_abl_%A_%a.err

DATASETS_ARR=(
    # --- RAT feature-group ablations (vs rat_natural) ---
    "configs/datasets/rat_ablation_no_struct.yaml"
    "configs/datasets/rat_ablation_no_temp.yaml"
    "configs/datasets/rat_ablation_no_amount.yaml"
    "configs/datasets/rat_ablation_no_burst_pattern.yaml"
    "configs/datasets/rat_ablation_no_entity.yaml"
    "configs/datasets/rat_ablation_no_rat_scores.yaml"
    "configs/datasets/rat_ablation_no_motif.yaml"
    "configs/datasets/rat_ablation_no_crossbank.yaml"
    # --- causal-SLT study (vs baseline) ---
    "configs/datasets/slt_causal_1hop.yaml"
    "configs/datasets/slt_causal_multihop.yaml"
    "configs/datasets/selfhist_only.yaml"
    "configs/datasets/selfhist_plus_slt_causal.yaml"
    "configs/datasets/slt_causal_placebo.yaml"
    # --- leak-free top-20 (needs build_top20_chrono.sh first; last so the CPU job is done by then) ---
    "configs/datasets/rat_ablation_top20_chrono.yaml"
)
# name | script | config | results folder name
MODELS_ARR=(
    "graphsage_t|scripts/training/train_graphsage_t_v2.py|configs/models/graphsage_t_v2.yaml|graphsage-t"
    # "graphsage|scripts/training/train_graphsage_v2.py|configs/models/graphsage_v2.yaml|graphsage"
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

# Each ablation must use EXACTLY the same edges and chronological split as its
# reference condition, otherwise the comparison is meaningless. Checked once per dataset.
declare -A SPLIT_OK
check_split() {   # $1 = ablation prefix
    local P="$1" REF
    case "$P" in HI-Small_Trans_RAT_pristine__*) REF="HI-Small_Trans_RAT_pristine" ;; *) REF="HI-Small_Trans" ;; esac
    python - "$P" "$REF" <<'PYEOF'
import sys, torch
p, r = sys.argv[1], sys.argv[2]
ok = torch.equal(torch.load(f"graphs/{p}/timestamps.pt"), torch.load(f"graphs/{r}/timestamps.pt"))
for s in ("train", "val", "test"):
    ok &= torch.equal(torch.load(f"splits/{p}/{s}_edge_idx.pt"), torch.load(f"splits/{r}/{s}_edge_idx.pt"))
print(f"split check {p} vs {r}: {'OK' if ok else 'MISMATCH'}")
sys.exit(0 if ok else 1)
PYEOF
}

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
    [ -d "graphs/$PREFIX" ] && [ -d "splits/$PREFIX" ] || { echo "$TAG: graphs/ or splits/$PREFIX missing, skip"; continue; }
    if [ -z "${SPLIT_OK[$PREFIX]:-}" ]; then check_split "$PREFIX" && SPLIT_OK[$PREFIX]=1 || SPLIT_OK[$PREFIX]=0; fi
    [ "${SPLIT_OK[$PREFIX]}" = 1 ] || { echo "$TAG: edges/split differ from reference -- rebuild this graph, skip"; continue; }
    case "$PREFIX" in *SLT*) [ -f logs/markers/slt_rebuild.done ] || { echo "$TAG: SLT rebuild not done, skip"; continue; } ;; esac
    mkdir -p "$RDIR"
    if [ -f "$RDIR/.lock" ]; then
        OTHER="$(cat "$RDIR/.lock")"
        if squeue -h -j "$OTHER" 2>/dev/null | grep -q .; then echo "$TAG: being trained by job $OTHER, skip"; continue; fi
    fi
    echo "$MY_JOB" > "$RDIR/.lock"

    SEED_CFG="logs/tmp_configs/base_v2abl_seed${SEED}_${MY_JOB}.yaml"
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
