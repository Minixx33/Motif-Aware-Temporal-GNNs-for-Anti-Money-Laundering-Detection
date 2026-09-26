#!/bin/bash
# ===========================================================================
# submit_full_pipeline.sh
#
# Queues the ENTIRE process: injection -> graphs -> structural conditions ->
# splits (chronological for ALL models, including GraphSAGE/GraphSAGE-T --
# see pipeline_prep.sh) -> node-degree fix (pipeline_prep.sh) -> then, only
# once that COMPLETES, the 75-task training array (5 datasets x 3 models x
# 5 seeds) submitted as 3 sequential batches of 25 tasks each.
#
# WHY THIS SCRIPT BLOCKS AND POLLS (important -- read before changing this):
# QoS gpu-long-mialhajri-001 has GrpSubmit=30 (confirmed via real
# `sacctmgr show qos` output): max jobs SUBMITTED (running + PENDING) at once
# under that QoS, and array tasks count individually. The first attempt at
# this script pre-submitted 3 batches of 25 up front, chained with
# --dependency=afterany -- that FAILED, because SLURM counts a
# dependency-held (pending) job toward GrpSubmit the instant it's submitted,
# not once it starts running. Submitting batch 2 while batch 1's 25 tasks
# were still in the queue meant 50 jobs registered at once, over the cap.
#
# The only correct fix: submit one batch, WAIT for it to fully leave the
# queue (not just "dependency satisfied"), THEN submit the next. That's what
# wait_for_queue_clear() below does, polling `squeue` every 60s.
#
# THIS MEANS THE SCRIPT BLOCKS FOR A LONG TIME (each 25-task batch at 3
# concurrent slots and the measured ~2.3h/run average takes roughly
# 25/3 * 2.3h =~ 19h to clear; ~57h total for all 3 batches). Run it under
# nohup or in a tmux/screen session so it survives you logging out:
#
#   nohup bash scripts/bash/submit_full_pipeline.sh \
#       > scripts/bash/logs/submit_full_pipeline.log 2>&1 &
#   disown
#
# Then track progress with:
#   tail -f scripts/bash/logs/submit_full_pipeline.log
#   squeue -u $(whoami)
#   bash scripts/bash/audit_runs.sh
#
# --time=06:00:00 in train_single_run.sh is sized from real measured
# per-run timing (max 3.55h observed across the 45 already-completed runs),
# not a guess. Your account's separate 3-GPU cap (not QoS-enforced) is what
# --array=...%3 handles within each batch.
# ===========================================================================
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
echo "Project root: $(pwd)"
mkdir -p scripts/bash/logs

# ---------------------------------------------------------------------------
# Poll sacct until a (non-array) job reaches a terminal state; prints it.
# ---------------------------------------------------------------------------
wait_for_state() {
    local jobid="$1"
    local state
    while true; do
        state=$(sacct -j "$jobid" --format=State --noheader -X 2>/dev/null | head -1 | tr -d ' ')
        case "$state" in
            COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL)
                echo "$state"
                return 0
                ;;
        esac
        sleep 60
    done
}

# ---------------------------------------------------------------------------
# Poll squeue until a job (and, for an array, ALL its tasks) has left the
# queue entirely -- this is what actually frees up GrpSubmit headroom.
# ---------------------------------------------------------------------------
wait_for_queue_clear() {
    local jobid="$1"
    while squeue -j "$jobid" -h 2>/dev/null | grep -q .; do
        sleep 60
    done
}

echo ""
echo "=== Submitting pipeline_prep.sh (injection -> graphs -> splits -> degree-fix) ==="
PREP_JOBID=$(sbatch --parsable scripts/bash/pipeline_prep.sh)
echo "  prep job id: $PREP_JOBID -- waiting for it to finish (this can take a while)..."
PREP_STATE=$(wait_for_state "$PREP_JOBID")
echo "  prep job $PREP_JOBID finished with state: $PREP_STATE"

if [ "$PREP_STATE" != "COMPLETED" ]; then
    echo "ERROR: prep job did not complete successfully (state=$PREP_STATE)." >&2
    echo "        Check scripts/bash/logs/pipeline_prep_${PREP_JOBID}.err and .log," >&2
    echo "        fix the problem, then re-run this script. Aborting before" >&2
    echo "        submitting any training (graphs/splits may be incomplete)." >&2
    exit 1
fi

echo ""
echo "=== Submitting training array in 3 batches of 25 (QoS GrpSubmit=30 cap) ==="
echo "This blocks between batches -- see this script's header for why. Make"
echo "sure you launched this under nohup/tmux if you want to log out."

BATCH1=$(sbatch --parsable --array=0-24%3 scripts/bash/train_single_run.sh)
echo "  batch 1 (tasks 0-24):  $BATCH1 -- waiting for it to clear the queue..."
wait_for_queue_clear "$BATCH1"
echo "  batch 1 cleared."

BATCH2=$(sbatch --parsable --array=25-49%3 scripts/bash/train_single_run.sh)
echo "  batch 2 (tasks 25-49): $BATCH2 -- waiting for it to clear the queue..."
wait_for_queue_clear "$BATCH2"
echo "  batch 2 cleared."

BATCH3=$(sbatch --parsable --array=50-74%3 scripts/bash/train_single_run.sh)
echo "  batch 3 (tasks 50-74): $BATCH3 -- waiting for it to clear the queue..."
wait_for_queue_clear "$BATCH3"
echo "  batch 3 cleared."

echo ""
echo "=== ALL 75 COMBOS SUBMITTED AND CLEARED THE QUEUE ==="
echo "This does NOT mean all 75 succeeded -- some may have failed/errored."
echo "Check final status with: bash scripts/bash/audit_runs.sh"
