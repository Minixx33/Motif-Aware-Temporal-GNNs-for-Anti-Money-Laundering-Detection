# Motif-Aware Temporal GNNs for Anti-Money Laundering Detection
MLR 570 Course Project

Instruction on how to run code:

All required packages are in the `requirements.txt` file. Recommended to also import the conda environment from `environment_full.yml`.

- All datasets csv files should be in ibm_transactions_datasets
- Make sure that the graph split is in splits/HI-Medium_RAT_low
- To run one of the models, the following commands are needed:

1. **Graphsage**
```
python "scripts/training/train_graphsage.py" \
    --config "configs/models/graphsage.yaml" \
    --dataset "configs/datasets/rat.yaml" \
    --base_config "configs/base.yaml" \
    --intensity "low" \
```

2. **Graphsage-T:**

```
python "scripts/training/train_graphsage_t.py" \
    --config "configs/models/graphsage_t.yaml" \
    --dataset "configs/datasets/rat.yaml" \
    --base_config "configs/base.yaml" \
    --intensity "low" \
```

3. **DyRep:**

```
python "scripts/training/train_dyrep.py" \
    --config "configs/models/dyrep.yaml" \
    --dataset "configs/datasets/rat.yaml" \
    --base_config "configs/base.yaml" \
    --intensity "low" \
```

> [!NOTE]: There are bash scripts in scripts/bash for auto running the models but it may not work
