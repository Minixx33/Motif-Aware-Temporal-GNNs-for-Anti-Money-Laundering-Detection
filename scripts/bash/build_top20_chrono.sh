#!/bin/bash
# ===========================================================================
# build_top20_chrono.sh -- leak-free top-20 RAT ablation graph (CPU, 1 job):
#
#     sbatch scripts/bash/build_top20_chrono.sh
#
# 1. select_top20_chrono.py: RF importance on rat_natural, chronological TRAIN
#    split only -> results/feature_importance/rat_natural_chrono/top20.json,
#    then builds graphs/HI-Small_Trans_RAT_pristine__top20_chrono
# 2. chronological splits + node-degree fix (same as every other ablation)
# 3. verifies edges/split are identical to rat_natural
# Training is done by train_v2_ablations.sh (top20_chrono is last in its list).
# ===========================================================================
#SBATCH --job-name=top20_chrono
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=14G
#SBATCH --time=12:00:00
#SBATCH --output=scripts/bash/logs/top20_chrono_%j.log
#SBATCH --error=scripts/bash/logs/top20_chrono_%j.err
set -e
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
CONDA_BASE=""
if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then CONDA_BASE="$("${CONDA_EXE}" info --base)"
elif command -v conda > /dev/null 2>&1; then CONDA_BASE="$(conda info --base)"
else for c in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" "/opt/anaconda3" "/opt/miniconda3" "/opt/conda"; do
    [ -f "$c/etc/profile.d/conda.sh" ] && { CONDA_BASE="$c"; break; }; done; fi
[ -n "$CONDA_BASE" ] || { echo "ERROR: conda not found"; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-aml_project}"

SRC="HI-Small_Trans_RAT_pristine"
OUT="${SRC}__top20_chrono"
[ -d "graphs/$SRC" ] && [ -d "splits/$SRC" ] || { echo "ERROR: graphs/ or splits/$SRC missing"; exit 1; }

echo ">>> [$(date +%H:%M:%S)] 1: select top-20 on TRAIN split + build graph"
python scripts/ablations/select_top20_chrono.py --graph "graphs/$SRC" --splits "splits/$SRC" --out_graph "graphs/$OUT"

echo ">>> [$(date +%H:%M:%S)] 2: chronological splits + degree fix"
python scripts/create_splits.py --graph_folder "graphs/$OUT" --split_mode chronological --out_dir "splits/$OUT"
python scripts/analysis/fix_node_degree_leakage.py --graph_dir "graphs/$OUT" --splits_dir "splits/$OUT"

echo ">>> [$(date +%H:%M:%S)] 3: verify identical edges/split to rat_natural"
python - "$OUT" "$SRC" <<'PYEOF'
import sys, torch
p, r = sys.argv[1], sys.argv[2]
ok = torch.equal(torch.load(f"graphs/{p}/timestamps.pt"), torch.load(f"graphs/{r}/timestamps.pt"))
for s in ("train", "val", "test"):
    ok &= torch.equal(torch.load(f"splits/{p}/{s}_edge_idx.pt"), torch.load(f"splits/{r}/{s}_edge_idx.pt"))
print("OK" if ok else "MISMATCH"); sys.exit(0 if ok else 1)
PYEOF
echo "DONE $(date)"
