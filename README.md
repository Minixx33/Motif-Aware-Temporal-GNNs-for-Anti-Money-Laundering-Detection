# Motif-Aware Temporal GNNs for Anti-Money Laundering Detection
MLR 570 Course Project

Instruction on how to run code:

All required packages are in the `requirements.txt` file. Recommended to also import the conda environment from `environment_full.yml`.

- All datasets csv files should be in ibm_transactions_datasets
- Download the full graphs from here: https://drive.google.com/drive/folders/1ZQMZdWmBJ0xpb0u2s3kI0ZjMw6AYYI1u?usp=sharing
- Download the graph split from here: https://drive.google.com/file/d/1OCAxAsK-WrduOqznuZ4qmxd4_S6u_Y9_/view?usp=sharing
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

There is also a bash script called `run_rat_all.sh` in scripts/bash that can be used to run the models. However, it will need some reconfiguration as it specifically works for my conda env but the commands are generalized.

> [!NOTE]: DyRep is not going to run as the split script is not optimized yet.
