#!/bin/bash
# ===========================================================================
# slt_rebuild_prep.sh
#
# Rebuilds ONLY the two SLT conditions after the slt_injector.py fix
# (current-day SLT exposure features were full-day aggregates -> now
# strictly-prior same-day values; strong-tie median fit on the train period).
# Same commands as pipeline_prep.sh, restricted to SLT:
#   slt_injector (pristine) -> static + DyRep graphs -> slt_plus_structural
#   (motif cols spliced from the UNCHANGED RAT-pristine graph) -> chronological
#   splits -> node-degree fix -> sanity check.
# Baseline / RAT-pristine / structural_only graphs are not touched.
#
# Requires graphs/HI-Small_Trans_RAT_pristine(+ graphs_dyrep/...) to exist
# already (motif columns for slt_plus_structural are copied from them).
#
# SLURM:  sbatch scripts/bash/slt_rebuild_prep.sh
#
#SBATCH --job-name=aml_slt_rebuild
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=scripts/bash/logs/slt_rebuild_prep_%j.log
#SBATCH --error=scripts/bash/logs/slt_rebuild_prep_%j.err
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
        "/opt/miniconda/etc/profile.d/conda.sh"; do
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
# Logging
# ---------------------------------------------------------------------------
mkdir -p "scripts/bash/logs"
if [ -z "${SLURM_JOB_ID:-}" ]; then
    ts=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="scripts/bash/logs/slt_rebuild_prep_${ts}.log"
    log() { echo "$@" | tee -a "$LOG_FILE"; }
else
    log() { echo "$@"; }
fi

log "==============================================================="
log " SLT REBUILD PREP (causal same-day SLT exposure fix)"
log " Host: $(hostname)  PID: $$  SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
log "==============================================================="

RAW_DIR="ibm_transcations_datasets"
for f in "$RAW_DIR/HI-Small_Trans.csv" "$RAW_DIR/HI-Small_accounts.csv"; do
    [ -e "$f" ] || { log "ERROR: Missing raw input: $f"; exit 1; }
done
for d in "graphs/HI-Small_Trans_RAT_pristine" "graphs_dyrep/HI-Small_Trans_RAT_pristine"; do
    [ -d "$d" ] || { log "ERROR: $d missing (needed for the motif columns of slt_plus_structural)"; exit 1; }
done

log ">>> [$(date +%H:%M:%S)] 0: self-test of the causal same-day exposure code"
python scripts/SLT/slt_causal_day_exposure.py --selftest --csv "$RAW_DIR/HI-Small_Trans.csv" --nrows 300000

log ">>> [$(date +%H:%M:%S)] 1: SLT injection (pristine snapshot; medium also written)"
python scripts/SLT/slt_injector.py --dump_pristine --intensities medium

log ">>> [$(date +%H:%M:%S)] 2: graphs"
python scripts/graph/motif_graph_builder_static.py --dataset SLT/HI-Small_Trans_SLT_pristine.csv
python scripts/graph/motif_dyrep_graph_builder.py  --dataset SLT/HI-Small_Trans_SLT_pristine.csv
python scripts/analysis/build_slt_plus_structural_graph.py
python scripts/analysis/build_slt_plus_structural_graph_dyrep.py

SLT_DATASETS=("HI-Small_Trans_SLT_pristine" "HI-Small_Trans_SLT_pristine_plus_structural")
log ">>> [$(date +%H:%M:%S)] 3: splits + node-degree fix"
for d in "${SLT_DATASETS[@]}"; do
    python scripts/create_splits.py --graph_folder "graphs/$d" --split_mode chronological --out_dir "splits/$d"
    python scripts/create_splits.py --graph_folder "graphs_dyrep/$d"
    python scripts/analysis/fix_node_degree_leakage.py --graph_dir "graphs/$d"
    python scripts/analysis/fix_node_degree_leakage.py --graph_dir "graphs_dyrep/$d"
done

log ">>> [$(date +%H:%M:%S)] 4: sanity check"
python - <<'PYEOF'
import json, sys, torch
exp = {"HI-Small_Trans_SLT_pristine": 52, "HI-Small_Trans_SLT_pristine_plus_structural": 56}
base_n = torch.load("graphs/HI-Small_Trans/timestamps.pt").numel()
ok = True
for d, ef_exp in exp.items():
    st = json.load(open(f"graphs/{d}/graph_stats.json"))
    cols = json.load(open(f"graphs/{d}/edge_attr_cols.json"))
    n = torch.load(f"graphs/{d}/timestamps.pt").numel()
    ts = torch.load(f"graphs/{d}/timestamps.pt")
    tr = torch.load(f"splits/{d}/train_edge_idx.pt"); va = torch.load(f"splits/{d}/val_edge_idx.pt"); te = torch.load(f"splits/{d}/test_edge_idx.pt")
    chrono = bool(ts[tr].max() <= ts[va].min() and ts[va].max() <= ts[te].min())
    good = (st.get("num_node_features") == 12 and len(cols) == ef_exp and n == base_n and chrono
            and "SLT_src_susp_nbr_ratio" in cols and not any(c.endswith(("_injected", "_intensity_level")) for c in cols))
    ok &= good
    print(f"  {d:<50} node_feats={st.get('num_node_features')} edge_feats={len(cols)} (exp {ef_exp}) "
          f"edges={n} (baseline {base_n}) chronological={chrono}  [{'OK' if good else 'MISMATCH'}]")
if not ok:
    print("ABORTING: SLT rebuild failed the sanity check."); sys.exit(1)
print("SLT graphs rebuilt and verified.")
PYEOF
log "==============================================================="
log " SLT REBUILD COMPLETE"
log "==============================================================="
