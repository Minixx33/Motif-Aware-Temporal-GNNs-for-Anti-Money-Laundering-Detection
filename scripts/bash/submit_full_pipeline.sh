#!/bin/bash
# ===========================================================================
# submit_full_pipeline.sh
#
# Queues the ENTIRE process in one shot: injection -> graphs -> structural
# conditions -> splits -> node-degree fix (pipeline_prep.sh, 1 job) -> then,
# only once that succeeds, 3 parallel training jobs (train_condition_group.sh),
# one per dataset group, each on its own GPU. Never uses more than 3 GPUs
# total, and the training jobs literally cannot start until prep finishes
# (SLURM --dependency=afterok), so there's no race with partially-built
# graphs.
#
# This script itself is NOT a SLURM job -- run it directly on the login
# node with plain bash. It only issues `sbatch` calls.
#
# Usage:
#   bash scripts/bash/submit_full_pipeline.sh
#
# Override the model/seed set for all 3 training jobs:
#   MODELS_ONLY="graphsage graphsage_t dyrep" SEEDS="1 2 3" \
#       bash scripts/bash/submit_full_pipeline.sh
# ===========================================================================
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
echo "Project root: $(pwd)"

MODELS_ONLY="${MODELS_ONLY:-graphsage graphsage_t dyrep}"
SEEDS="${SEEDS:-1 2 3}"

mkdir -p scripts/bash/logs

echo ""
echo "=== Submitting pipeline_prep.sh (injection -> graphs -> splits -> degree-fix) ==="
PREP_JOBID=$(sbatch --parsable scripts/bash/pipeline_prep.sh)
echo "  prep job id: $PREP_JOBID"

echo ""
echo "=== Submitting 3 training jobs, each depending on prep succeeding ==="

JOB1=$(sbatch --parsable \
    --job-name=aml_train_j1 \
    --dependency=afterok:"$PREP_JOBID" \
    --export=ALL,DATASETS_ONLY="baseline structural_only",MODELS_ONLY="$MODELS_ONLY",SEEDS="$SEEDS" \
    scripts/bash/train_condition_group.sh)
echo "  job 1 (baseline, structural_only): $JOB1"

JOB2=$(sbatch --parsable \
    --job-name=aml_train_j2 \
    --dependency=afterok:"$PREP_JOBID" \
    --export=ALL,DATASETS_ONLY="rat_natural",MODELS_ONLY="$MODELS_ONLY",SEEDS="$SEEDS" \
    scripts/bash/train_condition_group.sh)
echo "  job 2 (rat_natural): $JOB2"

JOB3=$(sbatch --parsable \
    --job-name=aml_train_j3 \
    --dependency=afterok:"$PREP_JOBID" \
    --export=ALL,DATASETS_ONLY="slt_natural slt_plus_structural",MODELS_ONLY="$MODELS_ONLY",SEEDS="$SEEDS" \
    scripts/bash/train_condition_group.sh)
echo "  job 3 (slt_natural, slt_plus_structural): $JOB3"

echo ""
echo "=== Queued. Track with: ==="
echo "  squeue -u \$(whoami)"
echo "  tail -f scripts/bash/logs/pipeline_prep_${PREP_JOBID}.log"
echo "  tail -f scripts/bash/logs/train_group_${JOB1}.log"
