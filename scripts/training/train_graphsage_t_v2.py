# train_graphsage_t_v2.py -- fixed GraphSAGE-T (causal_leakfix_v2). See causal_sage.py for
# the full list of fixes. v1 script (train_graphsage_t.py) is untouched.
#
# Usage:
#   python scripts/training/train_graphsage_t_v2.py \
#       --config configs/models/graphsage_t_v2.yaml \
#       --dataset configs/datasets/baseline.yaml \
#       --base_config configs/base.yaml
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from scripts.training.causal_sage import main

if __name__ == "__main__":
    main(temporal=True, default_config="configs/models/graphsage_t_v2.yaml")
