#!/bin/bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
PROJECT_ROOT="$(pwd)"

echo "Running from project root: $PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
PYTHON_EXE="/c/Users/g00084287/AppData/Local/miniconda3/envs/aml_project/python.exe"

if [ ! -f "$PYTHON_EXE" ]; then
    echo "ERROR: Python executable not found:"
    echo "$PYTHON_EXE"
    exit 1
fi

echo "Using Python: $PYTHON_EXE"
"$PYTHON_EXE" --version

# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------
BASE_CONFIG="configs/base.yaml"
DATASET_CONFIG="configs/datasets/slt.yaml"

if [ ! -f "$BASE_CONFIG" ]; then
    echo "ERROR: Missing base config: $BASE_CONFIG"
    exit 1
fi

if [ ! -f "$DATASET_CONFIG" ]; then
    echo "ERROR: Missing dataset config: $DATASET_CONFIG"
    exit 1
fi

BASE_BACKUP="${BASE_CONFIG}.backup_before_slt_5seeds"
cp "$BASE_CONFIG" "$BASE_BACKUP"

restore_base_config() {
    echo ""
    echo "Restoring original base.yaml..."
    cp "$BASE_BACKUP" "$BASE_CONFIG"
}
trap restore_base_config EXIT

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
INTENSITIES=("low" "medium" "high")
SEEDS=(1 2 3 4 5)

# ---------------------------------------------------------------------------
# Training scripts
# ---------------------------------------------------------------------------
TRAIN_SAGE="scripts/training/train_graphsage.py"
TRAIN_SAGET="scripts/training/train_graphsage_t.py"
TRAIN_DYREP="scripts/training/train_dyrep.py"

for script in "$TRAIN_SAGE" "$TRAIN_SAGET" "$TRAIN_DYREP"; do
    if [ ! -f "$script" ]; then
        echo "ERROR: Missing training script: $script"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
SAGE_CONFIG="configs/models/graphsage.yaml"
SAGET_CONFIG="configs/models/graphsage_t.yaml"
DYREP_CONFIG="configs/models/dyrep.yaml"

for config in "$SAGE_CONFIG" "$SAGET_CONFIG" "$DYREP_CONFIG"; do
    if [ ! -f "$config" ]; then
        echo "ERROR: Missing model config: $config"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
ts=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/slt_all_5seeds_${ts}.log"

echo "===============================================================" | tee -a "$LOG_FILE"
echo " RUNNING ALL SLT EXPERIMENTS WITH 5 SEEDS" | tee -a "$LOG_FILE"
echo " Models: GraphSAGE, GraphSAGE-T, DyRep" | tee -a "$LOG_FILE"
echo " Intensities: ${INTENSITIES[*]}" | tee -a "$LOG_FILE"
echo " Seeds: ${SEEDS[*]}" | tee -a "$LOG_FILE"
echo " Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "===============================================================" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo ">>> GPU INFO:" | tee -a "$LOG_FILE"
nvidia-smi 2>&1 | tee -a "$LOG_FILE" || echo "GPU info unavailable" | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Helper: update base.yaml
# ---------------------------------------------------------------------------
update_base_config() {
    local SEED="$1"
    local EXP_NAME="$2"

    "$PYTHON_EXE" - <<EOF
from pathlib import Path
import re

path = Path("$BASE_CONFIG")
text = path.read_text()

text = re.sub(
    r'^(\\s*seed:\\s*).*$',
    r'\\g<1>$SEED',
    text,
    flags=re.MULTILINE
)

text = re.sub(
    r'^(\\s*experiment_name:\\s*).*$',
    r'\\g<1>"$EXP_NAME"',
    text,
    flags=re.MULTILINE
)

path.write_text(text)
EOF
}

# ---------------------------------------------------------------------------
# Helper: run a single model
# ---------------------------------------------------------------------------
run_model() {
    local MODEL_NAME="$1"
    local SCRIPT_PATH="$2"
    local MODEL_CONFIG="$3"
    local INTENSITY="$4"
    local SEED="$5"

    local SAFE_MODEL_NAME
    SAFE_MODEL_NAME="$(echo "$MODEL_NAME" | tr '-' '_' | tr '[:upper:]' '[:lower:]')"

    local EXP_NAME
    EXP_NAME="slt_${INTENSITY}_${SAFE_MODEL_NAME}_seed${SEED}"

    echo "" | tee -a "$LOG_FILE"
    echo ">>> Updating base.yaml: seed=$SEED, experiment_name=$EXP_NAME" | tee -a "$LOG_FILE"

    update_base_config "$SEED" "$EXP_NAME"

    echo "[ $(date +"%Y-%m-%d %H:%M:%S") ] >>> Running $EXP_NAME" | tee -a "$LOG_FILE"

    model_start=$(date +%s)

    "$PYTHON_EXE" "$SCRIPT_PATH" \
        --config "$MODEL_CONFIG" \
        --dataset "$DATASET_CONFIG" \
        --base_config "$BASE_CONFIG" \
        --intensity "$INTENSITY" \
        2>&1 | tee -a "$LOG_FILE"

    model_end=$(date +%s)
    elapsed=$((model_end - model_start))

    h=$((elapsed / 3600))
    m=$(((elapsed % 3600) / 60))
    s=$((elapsed % 60))

    echo ">>> Finished: $EXP_NAME in ${h}h ${m}m ${s}s" | tee -a "$LOG_FILE"
}

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
start_time=$(date +%s)

for SEED in "${SEEDS[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "===============================================================" | tee -a "$LOG_FILE"
    echo ">>> STARTING SEED: $SEED" | tee -a "$LOG_FILE"
    echo "===============================================================" | tee -a "$LOG_FILE"

    for INTENSITY in "${INTENSITIES[@]}"; do
        echo "" | tee -a "$LOG_FILE"
        echo "---------------------------------------------------------------" | tee -a "$LOG_FILE"
        echo ">>> STARTING SLT INTENSITY: $INTENSITY | SEED: $SEED" | tee -a "$LOG_FILE"
        echo "---------------------------------------------------------------" | tee -a "$LOG_FILE"

        run_model "DyRep"       "$TRAIN_DYREP"  "$DYREP_CONFIG"  "$INTENSITY" "$SEED"
        run_model "GraphSAGE-T" "$TRAIN_SAGET"  "$SAGET_CONFIG"  "$INTENSITY" "$SEED"
        run_model "GraphSAGE"   "$TRAIN_SAGE"   "$SAGE_CONFIG"   "$INTENSITY" "$SEED"
    done
done

end_time=$(date +%s)
total=$((end_time - start_time))

th=$((total / 3600))
tm=$(((total % 3600) / 60))
ts2=$((total % 60))

echo "" | tee -a "$LOG_FILE"
echo "===============================================================" | tee -a "$LOG_FILE"
echo " ALL SLT EXPERIMENTS COMPLETED SUCCESSFULLY" | tee -a "$LOG_FILE"
echo " Total time: ${th}h ${tm}m ${ts2}s ($total seconds)" | tee -a "$LOG_FILE"
echo " Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "===============================================================" | tee -a "$LOG_FILE"

touch "$LOG_DIR/SLT_ALL_5SEEDS_FINISHED_${ts}.done"

echo ""
echo "SUCCESS! All SLT experiments completed for seeds: ${SEEDS[*]}"