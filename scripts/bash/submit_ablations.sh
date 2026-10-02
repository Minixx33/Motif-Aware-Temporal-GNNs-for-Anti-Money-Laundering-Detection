#!/bin/bash
# ===========================================================================
# submit_ablations.sh
#
# Queues the ENTIRE ablation study in one shot:
#   1. Build RAT ablation graphs/splits   (CPU, create_rat_ablation_graphs_static.sh)
#   2. Build SLT ablation graphs/splits   (CPU, create_slt_ablation_variants.sh, 5 tasks)
#      -- runs in parallel with (1), fully independent of it.
#   3. Train on the RAT ablations         (GPU, run_rat_ablations.sh, 9 tasks, %2)
#      -- starts only once (1) succeeds. Independent of SLT training.
#   4. Train on the SLT ablations         (GPU, run_slt_ablations.sh, 5 tasks, %1)
#      -- starts only once (2) succeeds. Independent of RAT training -- no
#         cross-dependency, runs concurrently with (3) if both are ready.
#
# GPU cap without cross-dependency: RAT training is throttled to %2 and SLT
# training to %1 (down from %3/%3). Worst case both run at once: 2+1=3 GPUs,
# same ceiling as before, just enforced per-job instead of by serializing
# one training job behind the other.
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
echo "=== Submitting RAT ablation training (gpu, 9 tasks, %2 throttle) ==="
RAT_TRAIN_JOBID=$(sbatch --parsable \
    --dependency=afterok:"$RAT_PREP_JOBID" \
    scripts/bash/run_rat_ablations.sh)
echo "  RAT training job id: $RAT_TRAIN_JOBID  (waits only for RAT build $RAT_PREP_JOBID)"

echo ""
echo "=== Submitting SLT ablation training (gpu, 5 tasks, %1 throttle) ==="
SLT_TRAIN_JOBID=$(sbatch --parsable \
    --dependency=afterok:"$SLT_PREP_JOBID" \
    scripts/bash/run_slt_ablations.sh)
echo "  SLT training job id: $SLT_TRAIN_JOBID  (waits only for SLT build $SLT_PREP_JOBID --"
echo "                        no dependency on RAT training, fully independent)"

echo ""
echo "=== Queued. Track with: ==="
echo "  squeue -u \$(whoami)"
echo "  sacct -j $RAT_PREP_JOBID,$SLT_PREP_JOBID,$RAT_TRAIN_JOBID,$SLT_TRAIN_JOBID --format=JobID,JobName,State,Elapsed,ExitCode"
echo "  tail -f scripts/bash/logs/rat_ablation_static_${RAT_PREP_JOBID}.log"
echo "  tail -f scripts/bash/logs/slt_create_${SLT_PREP_JOBID}_0.log      # per-task: _0 to _4"
echo "  tail -f scripts/bash/logs/rat_ablations_${RAT_TRAIN_JOBID}_0.log   # per-task: _0 to _8"
echo "  tail -f scripts/bash/logs/slt_ablations_${SLT_TRAIN_JOBID}_0.log   # per-task: _0 to _4"
