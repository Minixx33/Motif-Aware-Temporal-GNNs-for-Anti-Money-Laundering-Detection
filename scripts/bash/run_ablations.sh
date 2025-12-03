#!/bin/bash
set -e
set -o pipefail

# ---------------------------------------------------------
# Move to project root
# ---------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
PROJECT_ROOT="$(pwd)"

echo "Running from project root: $PROJECT_ROOT"

# ---------------------------------------------------------
# Activate Conda (Windows Git-Bash compatible)
# ---------------------------------------------------------
source "/c/ProgramData/Anaconda3/etc/profile.d/conda.sh"
conda activate aml_project

echo "Using Python: $(which python)"
python --version

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_CONFIG="configs/base.yaml"
MODEL_CONFIG="configs/models/dyrep.yaml"
DATASET_CONFIG="configs/datasets/rat.yaml"
TRAIN_SCRIPT="scripts/training/train_dyrep.py"

# ---------------------------------------------------------
# Verify files
# ---------------------------------------------------------
for f in "$BASE_CONFIG" "$MODEL_CONFIG" "$DATASET_CONFIG" "$TRAIN_SCRIPT"; do
    if [ ! -f "$PROJECT_ROOT/$f" ]; then
        echo "ERROR: Missing file: $PROJECT_ROOT/$f"
        exit 1
    fi
done

# ---------------------------------------------------------
# Logging setup
# ---------------------------------------------------------
ts=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/dyrep_rat_medium_ablation_${ts}.log"

echo "=====================================================================" | tee -a "$LOG_FILE"
echo "       Running DYREP RAT-MEDIUM Ablation Experiments (9 sets)         " | tee -a "$LOG_FILE"
echo " Timestamp: $ts " | tee -a "$LOG_FILE"
echo "=====================================================================" | tee -a "$LOG_FILE"

# GPU INFO
echo "" | tee -a "$LOG_FILE"
echo ">>> GPU INFO:" | tee -a "$LOG_FILE"
nvidia-smi 2>&1 | tee -a "$LOG_FILE" || echo "GPU not available" | tee -a "$LOG_FILE"

# =========================================================
# Helper: run DyRep on one ablation graph
# =========================================================
# NOTE: This assumes train_dyrep.py supports an optional
#       --graph_suffix argument that appends to the base
#       DyRep graph folder name, e.g.:
#       HI-Small_Trans_RAT_medium__no_struct
# =========================================================
run_dyrep_ablation() {
    local NAME="$1"
    local GRAPH_SUFFIX="$2"

    echo "" | tee -a "$LOG_FILE"
    echo ">>> Running DyRep on ablation: $NAME" | tee -a "$LOG_FILE"

    # Create dynamic base config
    TMP_BASE="configs/base_${NAME}.yaml"
    cp "$BASE_CONFIG" "$TMP_BASE"
    sed -i "s/name:.*/name: \"ablation_${NAME}\"/" "$TMP_BASE"

    start=$(date +%s)

    if python "$TRAIN_SCRIPT" \
        --config "$MODEL_CONFIG" \
        --dataset "$DATASET_CONFIG" \
        --intensity "medium" \
        --base_config "$TMP_BASE" \
        --graph_suffix "$GRAPH_SUFFIX" \
        2>&1 | tee -a "$LOG_FILE"; then

        end=$(date +%s)
        echo ">>> COMPLETED: $NAME in $((end-start))s" | tee -a "$LOG_FILE"

    else
        echo ">>> FAILURE: DyRep on ablation $NAME" | tee -a "$LOG_FILE"
        exit 1
    fi

    # Optional: cleanup
    rm "$TMP_BASE"
}


# =========================================================
# STEP 1 — Run DyRep on ALL 9 ablation folders
# (graphs_dyrep/HI-Small_Trans_RAT_medium__<name>)
# =========================================================

declare -a ABLATIONS=(
    "no_struct"
    # "no_temp"
    # "no_amount"
    # "no_burst_pattern"
    # "no_entity"
    # "no_rat_scores"
    # "no_motif"
    # "no_crossbank"
    # "top20_features"
)

for ab in "${ABLATIONS[@]}"; do
    run_dyrep_ablation "$ab" "__$ab"
done

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
echo "" | tee -a "$LOG_FILE"
echo "=====================================================================" | tee -a "$LOG_FILE"
echo "      ALL DYREP RAT-MEDIUM ABLATION EXPERIMENTS COMPLETED            " | tee -a "$LOG_FILE"
echo " Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "=====================================================================" | tee -a "$LOG_FILE"

touch "$LOG_DIR/DYREP_RAT_MEDIUM_ABLATIONS_DONE_${ts}.done"

echo ""
echo "SUCCESS! All ablation experiments finished."
echo "Check results in: results/"
