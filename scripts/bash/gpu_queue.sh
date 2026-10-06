#!/bin/bash
# ===========================================================================
# gpu_queue.sh -- submit GPU runs one array index at a time, never having more
# than MAX (default 3) of YOUR jobs queued+running in the GPU QoS. That keeps
# you at <=3 GPUs AND far below GrpSubmit=30 (pending tasks count too).
#
# Queue file: one run per line   <sbatch_script> <array_index> [<marker_file>]
#   marker_file (optional): the line waits until this file exists
#   (e.g. logs/markers/slt_rebuild.done for SLT conditions). '#' = comment.
#
# Restart-safe: submitted lines are recorded in <queue_file>.submitted and
# skipped next time. Run it on the login node inside tmux/screen:
#   tmux new -s q
#   bash scripts/bash/gpu_queue.sh scripts/bash/queues/base_main.txt
#   (detach: Ctrl-b d   reattach: tmux attach -t q)
# ===========================================================================
set -u
cd "$(dirname "$0")/../.."
export PATH=$PATH:/opt/slurm/bin
QUEUE="${1:?usage: gpu_queue.sh <queue_file> [max_in_flight]}"
MAX="${2:-3}"
QOS="${QOS:-gpu-long-mialhajri-001}"
POLL="${POLL:-120}"
DONE_FILE="${QUEUE}.submitted"
touch "$DONE_FILE"
mkdir -p scripts/bash/logs

inflight() { squeue -h -r -u "$USER" -q "$QOS" -t PENDING,RUNNING,CONFIGURING,COMPLETING 2>/dev/null | wc -l; }
ts() { date +"%Y-%m-%d %H:%M:%S"; }

mapfile -t LINES < <(tr -d '\r' < "$QUEUE" | sed -e 's/#.*//' -e '/^[[:space:]]*$/d')
echo "[$(ts)] $QUEUE: ${#LINES[@]} runs, max in flight $MAX (QoS $QOS)"

for line in "${LINES[@]}"; do
    read -r SCRIPT IDX MARKER <<< "$line"
    KEY="$SCRIPT $IDX"
    if grep -qxF "$KEY" "$DONE_FILE"; then continue; fi
    [ -f "$SCRIPT" ] || { echo "[$(ts)] ERROR: missing $SCRIPT"; exit 1; }
    if [ -n "${MARKER:-}" ]; then
        while [ ! -f "$MARKER" ]; do echo "[$(ts)] waiting for $MARKER before $KEY"; sleep "$POLL"; done
    fi
    while [ "$(inflight)" -ge "$MAX" ]; do sleep "$POLL"; done
    until JID=$(sbatch --parsable --array="$IDX" "$SCRIPT"); do
        echo "[$(ts)] sbatch failed for $KEY -- retrying in ${POLL}s"; sleep "$POLL"
    done
    echo "$KEY" >> "$DONE_FILE"
    echo "[$(ts)] submitted $KEY -> job $JID   (in flight now: $(inflight))"
    sleep 10
done
echo "[$(ts)] all runs in $QUEUE submitted. Check completion with scripts/bash/audit_runs.sh"
