#!/bin/bash
# ===========================================================================
# create_rat_ablation_graphs_static.sh
#
# Builds all RAT feature-ablation static graphs for GraphSAGE-T:
#   1. run_all_ablation_graphs_static.py  → graphs/HI-Small_Trans_RAT_medium__<name>/
#   2. create_splits.py on each of the 9 ablation graph folders
#
# LOCAL:  bash create_rat_ablation_graphs_static.sh
# SLURM:  sbatch create_rat_ablation_graphs_static.sh
#
#SBATCH --job-name=rat_ablation_static
#SBATCH --account=acc-mialhajri
#SBATCH --cpus-per-task=8
#SBATCH --time=500:00:00
#SBATCH --output=scripts/bash/logs/rat_ablation_static_%j.log
#SBATCH --error=scripts/bash/logs/rat_ablation_static_%j.err
# ===========================================================================
set -e
set -o pipefail

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."
PROJECT_ROOT="$(pwd)"
echo "Project root: $PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Portable conda activation
# ---------------------------------------------------------------------------
CONDA_BASE=""
if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
    CONDA_BASE="$("${CONDA_EXE}" info --base)"
elif command -v conda > /dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
else
    for candidate in \
        "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" \
        "/opt/anaconda3" "/opt/miniconda3" "/opt/conda" \
        "/c/ProgramData/Anaconda3" "/c/ProgramData/Miniconda3"; do
        if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
            CONDA_BASE="$candidate"
            break
        fi
    done
fi

if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "ERROR: Could not locate conda. Set CONDA_EXE or put conda on PATH."
    exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-aml_project}"
conda activate "$CONDA_ENV"

echo "Using Python: $(which python)"
python --version

# ---------------------------------------------------------------------------
# Script paths
# ---------------------------------------------------------------------------
ABLATION_SCRIPT="scripts/ablations/run_all_ablation_graphs_static.py"
SPLITS_SCRIPT="scripts/create_splits.py"
SOURCE_GRAPH="graphs/HI-Small_Trans_RAT_medium"

for s in "$ABLATION_SCRIPT" "$SPLITS_SCRIPT"; do
    [ -f "$s" ] || { echo "ERROR: Missing script: $s"; exit 1; }
done
[ -d "$SOURCE_GRAPH" ] || { echo "ERROR: Source graph not found: $SOURCE_GRAPH"; exit 1; }

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
mkdir -p "scripts/bash/logs"

ts=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="scripts/bash/logs/rat_ablation_static_${ts}.log"
log() { echo "$@" | tee -a "$LOG_FILE"; }

log "==============================================================="
log " CREATE RAT ABLATION STATIC GRAPHS"
log " Host: $(hostname)  PID: $$"
log " Source graph: $SOURCE_GRAPH"
log "==============================================================="

# ---------------------------------------------------------------------------
# Helper: elapsed time
# ---------------------------------------------------------------------------
elapsed() { printf "%dh %dm %ds" $(($1/3600)) $((($1%3600)/60)) $(($1%60)); }

total_start=$(date +%s)

# ---------------------------------------------------------------------------
# STEP 1: Build all 9 ablation graph folders
# ---------------------------------------------------------------------------
log ""
log ">>> [$(date +%H:%M:%S)] STEP 1: Building ablation graphs"
t0=$(date +%s)

cd scripts/ablations
python run_all_ablation_graphs_static.py --input_graph "../../${SOURCE_GRAPH}"
cd "$PROJECT_ROOT"

log ">>> Ablation graphs done in $(elapsed $(($(date +%s) - t0)))"

# ---------------------------------------------------------------------------
# STEP 2: Create splits for each ablation graph
# ---------------------------------------------------------------------------
ABLATION_NAMES=(
    "no_struct"
    "no_temp"
    "no_amount"
    "no_burst_pattern"
    "no_entity"
    "no_rat_scores"
    "no_motif"
    "no_crossbank"
    "top20_features"
)

log ""
log ">>> [$(date +%H:%M:%S)] STEP 2: Creating splits"

for NAME in "${ABLATION_NAMES[@]}"; do
    GRAPH_DIR="${SOURCE_GRAPH}__${NAME}"

    if [ ! -d "$GRAPH_DIR" ]; then
        log "  [WARN] Graph folder not found, skipping: $GRAPH_DIR"
        continue
    fi

    log ""
    log "  --- splits for: $NAME ---"
    t0=$(date +%s)
    python "$SPLITS_SCRIPT" --graph_folder "$GRAPH_DIR"
    log "  done in $(elapsed $(($(date +%s) - t0)))"
done

log ""
log "==============================================================="
log " DONE — splits land in splits/HI-Small_Trans_RAT_medium__<name>/"
log " Total time: $(elapsed $(($(date +%s) - total_start)))"
log "==============================================================="
