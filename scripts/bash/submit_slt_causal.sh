#!/bin/bash
# ===========================================================================
# submit_slt_causal.sh
#
# Queues the whole causal-SLT study in one shot (run on the login node):
#
#   slt_causal_prep.sh       (CPU)  builds 5 graph conditions + degree fix
#       |-- afterok --> slt_causal_train.sh   (GPU, 25 tasks, max 3 GPUs)
#   slt_causal_baselines.sh  (CPU)  feature-only diagnostic; independent of
#                                   prep and of training (no cross-dependency)
#
# Total GPU-QoS submissions = 25 (< GrpSubmit=30), so no polling needed.
# Training waits ONLY on its own prep job. Nothing else depends on anything.
#
# Usage:  bash scripts/bash/submit_slt_causal.sh
# ===========================================================================
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
echo "Project root: $(pwd)"
mkdir -p scripts/bash/logs

PREP=$(sbatch --parsable scripts/bash/slt_causal_prep.sh)
echo "prep job:      $PREP"

TRAIN=$(sbatch --parsable --dependency=afterok:"$PREP" scripts/bash/slt_causal_train.sh)
echo "training job:  $TRAIN  (waits only for prep $PREP)"

BASE=$(sbatch --parsable scripts/bash/slt_causal_baselines.sh)
echo "baselines job: $BASE  (independent)"

echo ""
echo "Track:  squeue -u \$(whoami)"
echo "        sacct -j $PREP,$TRAIN,$BASE --format=JobID,JobName,State,ExitCode,Elapsed -X"
echo "Logs:   scripts/bash/logs/slt_causal_prep_${PREP}.log"
echo "        scripts/bash/logs/slt_causal_train_${TRAIN}_0.log   (per task 0-24)"
echo "        scripts/bash/logs/slt_causal_baselines_${BASE}.log"
