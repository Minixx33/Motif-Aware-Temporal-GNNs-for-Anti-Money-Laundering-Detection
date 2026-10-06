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
#   bash scripts/bash/audit_runs.sh                         # v1 (train_single_run.sh)
#   AUDIT_SET=v2 bash scripts/bash/audit_runs.sh            # train_single_run_v2.sh (fixed GraphSAGE/-T)
#   AUDIT_SET=dyrep_full bash scripts/bash/audit_runs.sh    # train_single_run_dyrep_full.sh
#   AUDIT_SET=dyrep_lite_v2 bash scripts/bash/audit_runs.sh # train_single_run_dyrep_lite_v2.sh
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
AUDIT_SET="${AUDIT_SET:-v1}"
case "$AUDIT_SET" in
    v1)
        MODELS_ARR=(
            "graphsage_t|scripts/training/train_graphsage_t.py|configs/models/graphsage_t.yaml"
            "graphsage|scripts/training/train_graphsage.py|configs/models/graphsage.yaml"
            "dyrep|scripts/training/train_dyrep.py|configs/models/dyrep.yaml"
        )
        EXP_NAME="causal_leakfix_v1"; SUBMIT_SCRIPT="scripts/bash/train_single_run.sh" ;;
    v2)   # must match train_single_run_v2.sh
        MODELS_ARR=(
            "graphsage_t_v2|scripts/training/train_graphsage_t_v2.py|configs/models/graphsage_t_v2.yaml"
            "graphsage_v2|scripts/training/train_graphsage_v2.py|configs/models/graphsage_v2.yaml"
        )
        EXP_NAME="causal_leakfix_v2"; SUBMIT_SCRIPT="scripts/bash/train_single_run_v2.sh" ;;
    dyrep_full)   # must match train_single_run_dyrep_full.sh
        MODELS_ARR=(
            "dyrep_full|scripts/training/train_dyrep_full.py|configs/models/dyrep_full.yaml"
        )
        EXP_NAME="causal_leakfix_v2"; SUBMIT_SCRIPT="scripts/bash/train_single_run_dyrep_full.sh" ;;
    dyrep_lite_v2)   # must match train_single_run_dyrep_lite_v2.sh
        MODELS_ARR=(
            "dyrep_lite_v2|scripts/training/train_dyrep_lite_v2.py|configs/models/dyrep_lite_v2.yaml"
        )
        EXP_NAME="causal_leakfix_v2"; SUBMIT_SCRIPT="scripts/bash/train_single_run_dyrep_lite_v2.sh" ;;
    *) echo "Unknown AUDIT_SET=$AUDIT_SET (v1 | v2 | dyrep_full | dyrep_lite_v2)"; exit 1 ;;
esac
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

# ---------------------------------------------------------------------------
# Model-name-on-disk mapping: model_cfg["model"]["name"].lower() from each
# yaml -- GraphSAGE->graphsage, GraphSAGE-T->graphsage-t (hyphen kept,
# only spaces get replaced), DyRep->dyrep.
# ---------------------------------------------------------------------------
disk_model_name() {
    case "$1" in
        graphsage_t|graphsage_t_v2) echo "graphsage-t" ;;
        graphsage|graphsage_v2)     echo "graphsage" ;;
        dyrep|dyrep_lite_v2)        echo "dyrep" ;;
        dyrep_full)                 echo "dyrep-full" ;;
        *)           echo "$1" ;;
    esac
}

COMPLETE=0
INTERRUPTED=0
NOT_STARTED=0
INCOMPLETE_INDICES=()

printf "%-4s %-10s %-15s %-45s %-8s %s\n" "IDX" "MODEL" "LABEL" "DATASET_ON_DISK" "SEED" "STATUS"
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

    # The results folder is named after the dataset config's "prefix" field
    # (e.g. HI-Small_Trans_RAT_pristine), NOT the short label used above
    # (e.g. rat_natural) -- read it straight from the yaml so this can't
    # drift out of sync with configs/datasets/*.yaml.
    DISK_DATASET="$(grep -m1 'prefix:' "$DATASET_CONFIG" | sed -E 's/.*prefix:[[:space:]]*"([^"]*)".*/\1/')"
    if [ -z "$DISK_DATASET" ]; then
        echo "WARNING: could not read prefix from $DATASET_CONFIG, falling back to '$DATASET'" >&2
        DISK_DATASET="$DATASET"
    fi

    RESULTS_DIR="results/${DISK_DATASET}/seed${SEED}_${EXP_NAME}/${DISK_MODEL}"

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

    printf "%-4s %-10s %-15s %-45s %-8s %s\n" "$i" "$MODEL" "$DATASET" "$DISK_DATASET" "seed$SEED" "$STATUS"
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
    echo "  sbatch --array=${IDX_LIST}%3 ${SUBMIT_SCRIPT}"
    if [ -n "${SEEDS_LIST:-}" ]; then
        echo "  (add --export=ALL,SEEDS_LIST=\"$SEEDS_LIST\" since you audited a custom seed list)"
    fi
else
    echo ""
    echo "All combos complete."
fi
