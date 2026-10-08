#!/bin/bash
# ===========================================================================
# queue_job.sh -- ONE sbatch that runs the whole base experiment queue.
#
#   sbatch scripts/bash/queue_job.sh
#
# Runs on a CPU node (no GPU). Submits the GPU training runs itself, at most
# 3 at a time (max 3 GPUs, stays under GrpSubmit=30), skipping anything
# already submitted. When it hits its 12h limit it resubmits itself and
# continues, until every run is submitted.
#
# Default list = all base runs EXCEPT GraphSAGE-T and DyRep-Full (on hold):
# GraphSAGE + DyRep-Lite v2, 5 conditions x 5 seeds.
# Later, to run a held list:
#   sbatch --export=ALL,QUEUE=scripts/bash/queues/base_gst_hold.txt scripts/bash/queue_job.sh
# Progress: tail scripts/bash/logs/gpu_queue_*.log
#
#SBATCH --job-name=gpu_queue
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=12:00:00
#SBATCH --output=scripts/bash/logs/gpu_queue_%j.log
# ===========================================================================
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
export PATH=$PATH:/opt/slurm/bin
Q=scripts/bash/queues

# build the hold lists + the default run list from base_main.txt (idempotent)
grep    "graphsage-t"                     $Q/base_main.txt > $Q/base_gst_hold.txt
grep    "train_single_run_dyrep_full.sh"  $Q/base_main.txt > $Q/base_dyrep_full_hold.txt
grep -v "graphsage-t" $Q/base_main.txt | grep -v "train_single_run_dyrep_full.sh" > $Q/base_main_no_gst.txt

QUEUE="${QUEUE:-$Q/base_main_no_gst.txt}"
echo "Queue: $QUEUE ($(grep -c '^scripts' "$QUEUE") runs)"
timeout 11.5h bash scripts/bash/gpu_queue.sh "$QUEUE"
if [ $? -eq 124 ]; then
    echo "12h limit reached -- continuing in a new job"
    sbatch --export=ALL,QUEUE="$QUEUE" scripts/bash/queue_job.sh
fi
