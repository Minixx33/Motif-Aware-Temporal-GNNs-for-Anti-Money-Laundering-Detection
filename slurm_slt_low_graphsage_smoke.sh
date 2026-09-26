#!/bin/bash
#SBATCH --job-name=aml_slt_smoke
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --qos=gpu-long-iamer-001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=aml_slt_smoke_%j.out
#SBATCH --error=aml_slt_smoke_%j.err

source /opt/miniconda/etc/profile.d/conda.sh
conda activate aml_project

cd /shared/g00112161/Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection

echo "Host: $(hostname)"
echo "Python: $(which python)"
python --version
nvidia-smi

python scripts/training/train_graphsage.py \
  --config configs/smoke/graphsage_smoke.yaml \
  --dataset configs/datasets/slt.yaml \
  --base_config configs/smoke/base_smoke.yaml \
  --intensity low
