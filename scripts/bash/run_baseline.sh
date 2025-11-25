#!/bin/bash
# ============================================
# run_baseline_experiments.sh
# Run GraphSAGE, GraphSAGE-T, TGAT on BASELINE dataset only
# ============================================

set -e  # Stop on error
set -u  # Stop on undefined variable
set -o pipefail

# Timestamp
ts=$(date +"%Y%m%d_%H%M%S")

# Create logs folder
mkdir -p logs
log_file="logs/baseline_runs_${ts}.log"

echo "===========================================" | tee -a "$log_file"
echo " Running Baseline Experiments (SAGE, SAGE-T, TGAT) " | tee -a "$log_file"
echo " Timestamp: $ts " | tee -a "$log_file"
echo "===========================================" | tee -a "$log_file"

# GPU info
echo "" | tee -a "$log_file"
echo ">>> GPU Info:" | tee -a "$log_file"
nvidia-smi | tee -a "$log_file"

# Config paths
BASE_CONFIG="configs/base.yaml"
DATASET_CONFIG="configs/datasets/baseline.yaml"

# Track start time
start_time=$(date +%s)

# --------------------------------------------
# Run GraphSAGE
# --------------------------------------------
echo "" | tee -a "$log_file"
echo "[`date +"%Y-%m-%d %H:%M:%S"`] >>> Running GraphSAGE on baseline..." | tee -a "$log_file"
model_start=$(date +%s)

if python train_graphsage.py \
    --config configs/models/graphsage.yaml \
    --dataset "$DATASET_CONFIG" \
    --base_config "$BASE_CONFIG" \
    2>&1 | tee -a "$log_file"; then
    
    model_end=$(date +%s)
    model_time=$((model_end - model_start))
    echo ">>> GraphSAGE finished successfully in ${model_time}s." | tee -a "$log_file"
else
    echo ">>> ERROR: GraphSAGE failed!" | tee -a "$log_file"
    exit 1
fi

# --------------------------------------------
# Run GraphSAGE-T
# --------------------------------------------
echo "" | tee -a "$log_file"
echo "[`date +"%Y-%m-%d %H:%M:%S"`] >>> Running GraphSAGE-T on baseline..." | tee -a "$log_file"
model_start=$(date +%s)

if python train_graphsage_t.py \
    --config configs/models/graphsage_t.yaml \
    --dataset "$DATASET_CONFIG" \
    --base_config "$BASE_CONFIG" \
    2>&1 | tee -a "$log_file"; then
    
    model_end=$(date +%s)
    model_time=$((model_end - model_start))
    echo ">>> GraphSAGE-T finished successfully in ${model_time}s." | tee -a "$log_file"
else
    echo ">>> ERROR: GraphSAGE-T failed!" | tee -a "$log_file"
    exit 1
fi

# --------------------------------------------
# Run TGAT
# --------------------------------------------
echo "" | tee -a "$log_file"
echo "[`date +"%Y-%m-%d %H:%M:%S"`] >>> Running TGAT on baseline..." | tee -a "$log_file"
model_start=$(date +%s)

if python train_tgat.py \
    --config configs/models/tgat.yaml \
    --dataset "$DATASET_CONFIG" \
    --base_config "$BASE_CONFIG" \
    2>&1 | tee -a "$log_file"; then
    
    model_end=$(date +%s)
    model_time=$((model_end - model_start))
    echo ">>> TGAT finished successfully in ${model_time}s." | tee -a "$log_file"
else
    echo ">>> ERROR: TGAT failed!" | tee -a "$log_file"
    exit 1
fi

# Calculate total time
end_time=$(date +%s)
total_time=$((end_time - start_time))
total_minutes=$((total_time / 60))
total_seconds=$((total_time % 60))

echo "" | tee -a "$log_file"
echo "===========================================" | tee -a "$log_file"
echo " ALL BASELINE EXPERIMENTS COMPLETED SUCCESSFULLY " | tee -a "$log_file"
echo " Total time: ${total_minutes}m ${total_seconds}s " | tee -a "$log_file"
echo " Log saved to: $log_file " | tee -a "$log_file"
echo "===========================================" | tee -a "$log_file"

touch "logs/BASELINE_FINISHED_${ts}.done"
