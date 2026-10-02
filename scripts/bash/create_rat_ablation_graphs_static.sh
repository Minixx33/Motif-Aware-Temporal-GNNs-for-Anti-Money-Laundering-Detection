#!/bin/bash
# ===========================================================================
# create_rat_ablation_graphs_static.sh
#
# Builds all RAT feature-ablation static graphs for GraphSAGE-T:
#   1. run_all_ablation_graphs_static.py  → graphs/HI-Small_Trans_RAT_pristine__<name>/
#   2. create_splits.py (chronological) on each of the 9 ablation graph folders
#   3. fix_node_degree_leakage.py on each (overwrites x.pt's degree columns,
#      which motif_graph_builder_static.py computes from the FULL graph
#      regardless of split -- see pipeline_prep.sh for the same fix applied
#      to the primary 5-condition experiment. This script never had either
#      fix until now: it used to call create_splits.py with no --split_mode
#      (defaulting to stratified_random) and never called the degree fix at
#      all, so every existing RAT ablation result predates both corrections.
#
# LOCAL:  bash create_rat_ablation_graphs_static.sh
# SLURM:  sbatch create_rat_ablation_graphs_static.sh
#
#SBATCH --job-name=rat_ablation_static
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=scripts/bash/logs/rat_ablation_static_%j.log
#SBATCH --error=scripts/bash/logs/rat_ablation_static_%j.err
# ===========================================================================
set -e
set -o pipefail

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    cd "$SCRIPT_DIR/../.."
fi
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
DEGREE_FIX_SCRIPT="scripts/analysis/fix_node_degree_leakage.py"
# HI-Small_Trans_RAT_pristine -- NOT HI-Small_Trans_RAT_medium. The old
# "_medium" graph was the pre-fix, label-conditioned-intensity-boosted RAT
# dataset (rat.yaml) and was never built on this cluster. The primary
# experiment's actual RAT condition is rat_natural.yaml (prefix
# HI-Small_Trans_RAT_pristine, no boosting) -- ablations must be built from
# that SAME graph so the ablation results are about the RAT condition you
# actually report, not a deprecated one.
SOURCE_GRAPH="graphs/HI-Small_Trans_RAT_pristine"

for s in "$ABLATION_SCRIPT" "$SPLITS_SCRIPT" "$DEGREE_FIX_SCRIPT"; do
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
log ">>> [$(date +%H:%M:%S)] STEP 2: Creating splits (chronological) + degree-leakage fix"

for NAME in "${ABLATION_NAMES[@]}"; do
    GRAPH_DIR="${SOURCE_GRAPH}__${NAME}"
    SPLIT_DIR="splits/$(basename "$SOURCE_GRAPH")__${NAME}"

    if [ ! -d "$GRAPH_DIR" ]; then
        log "  [WARN] Graph folder not found, skipping: $GRAPH_DIR"
        continue
    fi

    log ""
    log "  --- splits for: $NAME ---"
    t0=$(date +%s)
    # --split_mode chronological + explicit --out_dir: without --out_dir,
    # create_splits.py writes to a sibling "_chrono" folder instead of
    # "splits/<name>", which build_paths() doesn't know to look for (same
    # gotcha fixed in pipeline_prep.sh for the primary experiment).
    python "$SPLITS_SCRIPT" --graph_folder "$GRAPH_DIR" --split_mode chronological --out_dir "$SPLIT_DIR"
    log "  splits done in $(elapsed $(($(date +%s) - t0)))"

    t0=$(date +%s)
    python "$DEGREE_FIX_SCRIPT" --graph_dir "$GRAPH_DIR" --splits_dir "$SPLIT_DIR"
    log "  degree-fix done in $(elapsed $(($(date +%s) - t0)))"
done

log ""
log "==============================================================="
log " DONE — splits land in splits/HI-Small_Trans_RAT_pristine__<name>/"
log " Total time: $(elapsed $(($(date +%s) - total_start)))"
log "==============================================================="
