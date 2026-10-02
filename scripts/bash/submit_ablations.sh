#!/bin/bash
# ===========================================================================
# submit_ablations.sh
#
# Queues the ENTIRE ablation study in one shot:
#   1. Build RAT ablation graphs/splits   (CPU, create_rat_ablation_graphs_static.sh)
#   2. Build SLT ablation graphs/splits   (CPU, create_slt_ablation_variants.sh, 5 tasks)
#      -- runs in parallel with (1), fully independent of it.
#   3. Train on the RAT ablations         (GPU, run_rat_ablations.sh, 9 tasks, %3)
#      -- starts only once (1) succeeds.
#   4. Train on the SLT ablations         (GPU, run_slt_ablations.sh, 5 tasks, %3)
#      -- starts only once (2) succeeds AND (3) has fully cleared the queue.
#
# WHY (4) waits on (3) specifically (not just (2)):
# Each training job's own --array=...%3 throttle only limits concurrency
# WITHIN that one job. If RAT training (9 tasks, %3) and SLT training
# (5 tasks, %3) were both submitted free-running, SLURM could run 3 from
# each at the same time -- 6 GPUs, over your 3-GPU cap. Chaining SLT
# training's start to RAT training's full completion (afterany, not
# afterok -- a few individual ablation failures shouldn't block the other
# theory's training) guarantees the two training jobs never overlap, so
# total GPU usage never exceeds 3 at any point.
#
# WHY this can submit everything up front, unlike submit_full_pipeline.sh:
# that script needed to poll-and-wait between batches because 75 training
# tasks at once exceeded QoS gpu-long-mialhajri-001's GrpSubmit=30 cap
# (which counts pending jobs immediately, not just running ones). Here,
# total GPU-QoS submissions are only 9 + 5 = 14, comfortably under 30, so
# a straightforward dependency chain submitted all at once is safe -- no
# polling needed.
#
# This script itself is NOT a SLURM job -- run it directly on the login
# node with plain bash. It only issues `sbatch` calls.
#
# Usage:
#   bash scripts/bash/submit_ablations.sh
# ===========================================================================
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
echo "Project root: $(pwd)"
mkdir -p scripts/bash/logs

echo ""
echo "=== Submitting RAT ablation graph/split build (cpu partition) ==="
RAT_PREP_JOBID=$(sbatch --parsable scripts/bash/create_rat_ablation_graphs_static.sh)
echo "  RAT build job id: $RAT_PREP_JOBID"

echo ""
echo "=== Submitting SLT ablation graph/split build (cpu partition, 5 tasks) ==="
SLT_PREP_JOBID=$(sbatch --parsable scripts/bash/create_slt_ablation_variants.sh)
echo "  SLT build job id: $SLT_PREP_JOBID"

echo ""
echo "=== Submitting RAT ablation training (gpu, 9 tasks, %3 throttle) ==="
RAT_TRAIN_JOBID=$(sbatch --parsable \
    --dependency=afterok:"$RAT_PREP_JOBID" \
    scripts/bash/run_rat_ablations.sh)
echo "  RAT training job id: $RAT_TRAIN_JOBID  (waits for RAT build $RAT_PREP_JOBID to succeed)"

echo ""
echo "=== Submitting SLT ablation training (gpu, 5 tasks, %3 throttle) ==="
SLT_TRAIN_JOBID=$(sbatch --parsable \
    --dependency=afterok:"$SLT_PREP_JOBID",afterany:"$RAT_TRAIN_JOBID" \
    scripts/bash/run_slt_ablations.sh)
echo "  SLT training job id: $SLT_TRAIN_JOBID  (waits for SLT build $SLT_PREP_JOBID to succeed"
echo "                        AND RAT training $RAT_TRAIN_JOBID to fully clear -- keeps"
echo "                        combined GPU usage at <=3 by never overlapping the two)"

echo ""
echo "=== Queued. Track with: ==="
echo "  squeue -u \$(whoami)"
echo "  sacct -j $RAT_PREP_JOBID,$SLT_PREP_JOBID,$RAT_TRAIN_JOBID,$SLT_TRAIN_JOBID --format=JobID,JobName,State,Elapsed,ExitCode"
echo "  tail -f scripts/bash/logs/rat_ablation_static_${RAT_PREP_JOBID}.log"
echo "  tail -f scripts/bash/logs/slt_create_${SLT_PREP_JOBID}_0.log      # per-task: _0 to _4"
echo "  tail -f scripts/bash/logs/rat_ablations_${RAT_TRAIN_JOBID}_0.log   # per-task: _0 to _8"
echo "  tail -f scripts/bash/logs/slt_ablations_${SLT_TRAIN_JOBID}_0.log   # per-task: _0 to _4"
