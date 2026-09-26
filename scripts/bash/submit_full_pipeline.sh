#!/bin/bash
# ===========================================================================
# submit_full_pipeline.sh
#
# Queues the ENTIRE process in one shot: injection -> graphs -> structural
# conditions -> splits (chronological for ALL models, including GraphSAGE/
# GraphSAGE-T -- see pipeline_prep.sh) -> node-degree fix (pipeline_prep.sh,
# 1 job) -> then, only once that succeeds, one SLURM job ARRAY
# (train_single_run.sh) with 75 tasks (5 datasets x 3 models x 5 seeds),
# throttled to 3 running at once via --array=0-74%3. That throttle is what
# actually caps GPU usage at 3 -- your
# account's 3-GPU limit isn't QoS-enforced, so SLURM won't stop you from
# submitting more than 3 concurrently unless something like %3 tells it to.
# The training array literally cannot start until prep finishes (SLURM
# --dependency=afterok), so there's no race with partially-built graphs.
#
# Each array task = exactly ONE (dataset, model, seed) run, so its 500-hour
# time limit only has to cover one run, not several bundled sequentially --
# see train_single_run.sh's header for why the earlier 3-job split
# (train_condition_group.sh, 48h limit, up to 18 runs per job) was wrong.
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
echo "=== Submitting training job array (75 tasks, 3 running at a time) ==="

TRAIN_JOBID=$(sbatch --parsable \
    --dependency=afterok:"$PREP_JOBID" \
    scripts/bash/train_single_run.sh)
echo "  train array job id: $TRAIN_JOBID"

echo ""
echo "=== Queued. Track with: ==="
echo "  squeue -u \$(whoami)"
echo "  tail -f scripts/bash/logs/pipeline_prep_${PREP_JOBID}.log"
echo "  tail -f scripts/bash/logs/train_${TRAIN_JOBID}_0.log   # array task 0's log, etc."
