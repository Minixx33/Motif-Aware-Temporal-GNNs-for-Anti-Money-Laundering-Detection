#!/bin/bash
# ===========================================================================
# submit_full_pipeline.sh
#
# Queues the ENTIRE process in one shot: injection -> graphs -> structural
# conditions -> splits (chronological for ALL models, including GraphSAGE/
# GraphSAGE-T -- see pipeline_prep.sh) -> node-degree fix (pipeline_prep.sh,
# 1 job) -> then, only once that succeeds, the 75-task training array
# (5 datasets x 3 models x 5 seeds) split into 3 chained batches of 25 tasks
# each (0-24, 25-49, 50-74).
#
# Two separate, real, cluster-enforced limits are at play here:
#   1. Your account's 3-GPU cap -- not QoS-enforced, so SLURM won't stop you
#      from running more than 3 at once unless something like %3 tells it to.
#   2. QoS gpu-long-mialhajri-001's GrpSubmit=30 -- max jobs SUBMITTED
#      (running + pending) at once under that QoS, and array tasks count
#      individually (confirmed via real `sacctmgr show qos` output, not
#      guessed). Submitting all 75 tasks as one array hit this immediately.
#      Batches of 25 stay safely under 30; each batch is submitted with
#      --dependency=afterany on the PREVIOUS batch's entire array finishing,
#      so batch N+1 isn't even submitted -- and doesn't count toward
#      GrpSubmit -- until batch N has fully cleared the queue.
#
# The first training batch literally cannot start until prep finishes
# (SLURM --dependency=afterok), so there's no race with partially-built
# graphs. Each array task = exactly ONE (dataset, model, seed) run;
# --time=06:00:00 in train_single_run.sh is sized from real measured
# per-run timing (max 3.55h observed across the 45 already-completed runs),
# not a guess.
#
# This script itself is NOT a SLURM job -- run it directly on the login
# node with plain bash. It only issues `sbatch` calls.
#
# Usage:
#   bash scripts/bash/submit_full_pipeline.sh
# ===========================================================================
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
echo "Project root: $(pwd)"

mkdir -p scripts/bash/logs

echo ""
echo "=== Submitting pipeline_prep.sh (injection -> graphs -> splits -> degree-fix) ==="
PREP_JOBID=$(sbatch --parsable scripts/bash/pipeline_prep.sh)
echo "  prep job id: $PREP_JOBID"

echo ""
echo "=== Submitting training array in 3 batches of 25 (QoS GrpSubmit=30 cap) ==="

BATCH1=$(sbatch --parsable \
    --dependency=afterok:"$PREP_JOBID" \
    --array=0-24%3 \
    scripts/bash/train_single_run.sh)
echo "  batch 1 (tasks 0-24):  $BATCH1  (waits for prep job $PREP_JOBID)"

BATCH2=$(sbatch --parsable \
    --dependency=afterany:"$BATCH1" \
    --array=25-49%3 \
    scripts/bash/train_single_run.sh)
echo "  batch 2 (tasks 25-49): $BATCH2  (waits for batch 1 to fully clear)"

BATCH3=$(sbatch --parsable \
    --dependency=afterany:"$BATCH2" \
    --array=50-74%3 \
    scripts/bash/train_single_run.sh)
echo "  batch 3 (tasks 50-74): $BATCH3  (waits for batch 2 to fully clear)"

echo ""
echo "=== Queued. Track with: ==="
echo "  squeue -u \$(whoami)"
echo "  tail -f scripts/bash/logs/pipeline_prep_${PREP_JOBID}.log"
echo "  tail -f scripts/bash/logs/train_${BATCH1}_0.log   # batch 1 task 0's log, etc."
echo "  bash scripts/bash/audit_runs.sh   # completion status across all 75 combos, any time"
