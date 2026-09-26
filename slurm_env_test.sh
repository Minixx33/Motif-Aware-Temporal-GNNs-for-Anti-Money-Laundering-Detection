#!/bin/bash
#SBATCH --job-name=aml_env_test
#SBATCH --account=acc-iamer
#SBATCH --partition=gpu
#SBATCH --qos=gpu-long-iamer-001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=aml_env_test_%j.out
#SBATCH --error=aml_env_test_%j.err

source /opt/miniconda/etc/profile.d/conda.sh
conda activate /shared/conda_envs/aml_project

cd /shared/g00112161/Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection

echo "Host: $(hostname)"
echo "Python: $(which python)"
python --version
nvidia-smi

python -c "import sys, torch, torch_geometric, sklearn, pandas, numpy, yaml; print('python:', sys.version); print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('torch_geometric:', torch_geometric.__version__); print('sklearn:', sklearn.__version__); print('pandas:', pandas.__version__); print('numpy:', numpy.__version__); print('yaml ok')"
