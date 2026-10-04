#!/bin/bash
# ===========================================================================
# slt_causal_prep.sh
#
# Builds the 5 causal-SLT-study graph conditions from the plain BASELINE graph
# (graphs/HI-Small_Trans + its chronological split, both already produced by
# pipeline_prep.sh), then applies the node-degree leakage fix to each:
#
#   HI-Small_Trans_SLT_causal_1hop             baseline + 6  (direct peer exposure)
#   HI-Small_Trans_SLT_causal_multihop         baseline + 10 (+ decayed / 2-hop)
#   HI-Small_Trans_selfhist_only               baseline + 2  (reputation CONTROL, not SLT)
#   HI-Small_Trans_selfhist_plus_SLT_causal    baseline + 12 (peer exposure beyond reputation)
#   HI-Small_Trans_SLT_causal_placebo          baseline + 10 (train labels permuted)
#
# See scripts/analysis/build_slt_causal_graphs.py for the exact feature
# definitions and causality guarantees. CPU/numpy-bound; ~1-2 min of compute
# plus file IO. Splits are COPIED from the baseline split, so every condition
# uses exactly the same train/val/test rows as the primary experiment.
#
# SLURM:  sbatch scripts/bash/slt_causal_prep.sh
# LOCAL:  bash scripts/bash/slt_causal_prep.sh
#
#SBATCH --job-name=slt_causal_prep
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=14G
#SBATCH --time=12:00:00
#SBATCH --output=scripts/bash/logs/slt_causal_prep_%j.log
#SBATCH --error=scripts/bash/logs/slt_causal_prep_%j.err
# ===========================================================================
set -e
set -o pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    cd "$SCRIPT_DIR/../.."
fi
echo "Project root: $(pwd)"

CONDA_BASE=""
if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
    CONDA_BASE="$("${CONDA_EXE}" info --base)"
elif command -v conda > /dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
else
    for candidate in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" \
                     "/opt/anaconda3" "/opt/miniconda3" "/opt/conda"; do
        if [ -f "$candidate/etc/profile.d/conda.sh" ]; then CONDA_BASE="$candidate"; break; fi
    done
fi
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "ERROR: Could not locate conda."; exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-aml_project}"
echo "Using Python: $(which python)"

mkdir -p scripts/bash/logs

BASE_GRAPH="graphs/HI-Small_Trans"
BASE_SPLIT="splits/HI-Small_Trans"
for p in "$BASE_GRAPH" "$BASE_SPLIT"; do
    [ -d "$p" ] || { echo "ERROR: missing $p (run pipeline_prep.sh first)"; exit 1; }
done
# The builder itself also aborts unless the split is truly chronological
# (train max timestamp <= val/test min timestamp), checked on the real tensors.

echo ""
echo ">>> [$(date +%H:%M:%S)] building 5 conditions"
python scripts/analysis/build_slt_causal_graphs.py \
    --baseline_graph_dir "$BASE_GRAPH" --baseline_split_dir "$BASE_SPLIT"

CONDITIONS=(
    "HI-Small_Trans_SLT_causal_1hop"
    "HI-Small_Trans_SLT_causal_multihop"
    "HI-Small_Trans_selfhist_only"
    "HI-Small_Trans_selfhist_plus_SLT_causal"
    "HI-Small_Trans_SLT_causal_placebo"
)

echo ""
echo ">>> [$(date +%H:%M:%S)] node-degree leakage fix (idempotent)"
for d in "${CONDITIONS[@]}"; do
    python scripts/analysis/fix_node_degree_leakage.py \
        --graph_dir "graphs/$d" --splits_dir "splits/$d"
done

echo ""
echo ">>> [$(date +%H:%M:%S)] sanity check"
python - <<'PYEOF'
import json, sys
expected = {   # edge features = 12 baseline + added
    "HI-Small_Trans_SLT_causal_1hop": 18,
    "HI-Small_Trans_SLT_causal_multihop": 22,
    "HI-Small_Trans_selfhist_only": 14,
    "HI-Small_Trans_selfhist_plus_SLT_causal": 24,
    "HI-Small_Trans_SLT_causal_placebo": 22,
}
ok = True
for d, exp in expected.items():
    st = json.load(open(f"graphs/{d}/graph_stats.json"))
    sm = json.load(open(f"splits/{d}/split_metadata.json"))
    good = (st["num_edge_features"] == exp and st["num_node_features"] == 12
            and sm.get("split_method") == "chronological")
    ok &= good
    print(f"  {d:<44} edge_feats={st['num_edge_features']} (exp {exp}) "
          f"node_feats={st['num_node_features']} split={sm.get('split_method')}  "
          f"[{'OK' if good else 'MISMATCH'}]")
if not ok:
    print("ABORTING: sanity check failed"); sys.exit(1)
print("All 5 conditions passed.")
PYEOF

echo ""
echo "PREP COMPLETE"
