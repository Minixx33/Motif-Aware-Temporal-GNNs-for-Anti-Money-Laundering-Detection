#!/bin/bash
# ===========================================================================
# audit_runs.sh
#
# Filesystem-based completion audit for the primary 5-condition experiment.
# Mirrors train_single_run.sh's exact combo list and ordering (same
# DATASETS_ARR/MODELS_ARR/SEEDS_ARR, same SEEDS_LIST override), so the index
# printed for each combo here is the SAME SLURM_ARRAY_TASK_ID you'd pass to
# `sbatch --array=...` on train_single_run.sh to resubmit it.
#
# For each combo, checks results/<dataset>/seed<N>_causal_leakfix_v1/<model>/:
#   COMPLETE     - metrics.json exists (run finished, including final eval)
#   INTERRUPTED  - checkpoint.pt exists but no metrics.json (started, cut off
#                  -- likely by the old --time=48:00:00 limit or a crash;
#                  resubmitting the same array index auto-resumes from this
#                  checkpoint, see train_graphsage.py's load_checkpoint call)
#   NOT_STARTED  - neither file exists (never ran, or results dir missing)
#
# Usage:
#   bash scripts/bash/audit_runs.sh
#   SEEDS_LIST="4 5" bash scripts/bash/audit_runs.sh   # audit only seeds 4-5
# ===========================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
PROJECT_ROOT="$(pwd)"

# ---------------------------------------------------------------------------
# EXACT same combo source as train_single_run.sh -- keep these two files in
# sync if the condition/model list ever changes.
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
if [ -n "${SEEDS_LIST:-}" ]; then
    read -ra SEEDS_ARR <<< "$SEEDS_LIST"
else
    SEEDS_ARR=(1 2 3 4 5)
fi

EXP_NAME="causal_leakfix_v1"   # from configs/base.yaml experiment.name

COMBOS=()
for ds in "${DATASETS_ARR[@]}"; do
    for mod in "${MODELS_ARR[@]}"; do
        for seed in "${SEEDS_ARR[@]}"; do
            COMBOS+=("$ds##$mod##$seed")
        done
    done
done

# ---------------------------------------------------------------------------
# Model-name-on-disk mapping: model_cfg["model"]["name"].lower() from each
# yaml -- GraphSAGE->graphsage, GraphSAGE-T->graphsage-t (hyphen kept,
# only spaces get replaced), DyRep->dyrep.
# ---------------------------------------------------------------------------
disk_model_name() {
    case "$1" in
        graphsage_t) echo "graphsage-t" ;;
        graphsage)   echo "graphsage" ;;
        dyrep)       echo "dyrep" ;;
        *)           echo "$1" ;;
    esac
}

COMPLETE=0
INTERRUPTED=0
NOT_STARTED=0
INCOMPLETE_INDICES=()

printf "%-4s %-10s %-38s %-12s %-6s %s\n" "IDX" "MODEL" "DATASET" "SEED" "" "STATUS"
echo "--------------------------------------------------------------------------------------------------"

for i in "${!COMBOS[@]}"; do
    COMBO="${COMBOS[$i]}"
    DS_ENTRY="${COMBO%%##*}"
    REST="${COMBO#*##}"
    MOD_ENTRY="${REST%%##*}"
    SEED="${REST#*##}"

    IFS='|' read -r DATASET DATASET_CONFIG INTENSITY <<< "$DS_ENTRY"
    IFS='|' read -r MODEL MODEL_SCRIPT MODEL_CONFIG <<< "$MOD_ENTRY"
    DISK_MODEL="$(disk_model_name "$MODEL")"

    RESULTS_DIR="results/${DATASET}/seed${SEED}_${EXP_NAME}/${DISK_MODEL}"

    if [ -f "$RESULTS_DIR/metrics.json" ]; then
        STATUS="COMPLETE"
        COMPLETE=$((COMPLETE+1))
    elif [ -f "$RESULTS_DIR/checkpoint.pt" ]; then
        STATUS="INTERRUPTED"
        INTERRUPTED=$((INTERRUPTED+1))
        INCOMPLETE_INDICES+=("$i")
    else
        STATUS="NOT_STARTED"
        NOT_STARTED=$((NOT_STARTED+1))
        INCOMPLETE_INDICES+=("$i")
    fi

    printf "%-4s %-10s %-38s %-12s %-6s %s\n" "$i" "$MODEL" "$DATASET" "seed$SEED" "" "$STATUS"
done

TOTAL=${#COMBOS[@]}
echo "--------------------------------------------------------------------------------------------------"
echo "TOTAL: $TOTAL   COMPLETE: $COMPLETE   INTERRUPTED: $INTERRUPTED   NOT_STARTED: $NOT_STARTED"

if [ ${#INCOMPLETE_INDICES[@]} -gt 0 ]; then
    IDX_LIST=$(IFS=,; echo "${INCOMPLETE_INDICES[*]}")
    echo ""
    echo "To resubmit exactly the incomplete/interrupted runs (INTERRUPTED ones"
    echo "auto-resume from their checkpoint -- see load_checkpoint() in each"
    echo "training script -- NOT_STARTED ones just start fresh):"
    echo ""
    echo "  sbatch --array=${IDX_LIST}%3 scripts/bash/train_single_run.sh"
    if [ -n "${SEEDS_LIST:-}" ]; then
        echo "  (add --export=ALL,SEEDS_LIST=\"$SEEDS_LIST\" since you audited a custom seed list)"
    fi
else
    echo ""
    echo "All combos complete."
fi
