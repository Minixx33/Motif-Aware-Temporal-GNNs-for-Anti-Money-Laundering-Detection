# Motif-Aware Temporal GNNs for Anti-Money Laundering Detection

**MLR 570 Course Project**

This project trains Graph Neural Network (GNN) models on the IBM AML synthetic transaction dataset to detect money-laundering edges (transactions). On top of the raw transaction graph, criminology-theory features are injected using **Routine Activity Theory (RAT)** and **Social Learning Theory (SLT)**, alongside structural **graph motif features** (fan-in, fan-out, chain, cycle). The study examines how these features improve detection under severe class imbalance (~0.1% positive rate).

---

## Table of Contents

1. [Models](#1-models)
2. [Repository Structure](#2-repository-structure)
3. [Prerequisites](#3-prerequisites)
4. [Installation](#4-installation)
5. [Data & Pre-built Graphs](#5-data--pre-built-graphs)
6. [End-to-End Pipeline](#6-end-to-end-pipeline)
7. [Running Experiments](#7-running-experiments)
8. [Configuration Reference](#8-configuration-reference)
9. [Outputs](#9-outputs)
10. [Building Graphs from Scratch](#10-building-graphs-from-scratch)
11. [Ablation Studies](#11-ablation-studies)
12. [Analysis & Utilities](#12-analysis--utilities)
13. [Bash Driver Scripts](#13-bash-driver-scripts)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Models

| Model | Type | Script | Notes |
|---|---|---|---|
| **GraphSAGE** | Static edge classifier | `scripts/training/train_graphsage.py` | Baseline GNN |
| **GraphSAGE-T** | Temporal (sinusoidal time encoding) | `scripts/training/train_graphsage_t.py` | Time-aware variant |
| **DyRep** | Temporal event stream | `scripts/training/train_dyrep.py` | Functional; temporal split optimisation ongoing |

A copy of the original **TGN** (Temporal Graph Networks) codebase is included under `tgn/` for reference and potential extension.

---

## 2. Repository Structure

```
.
├── configs/
│   ├── base.yaml                          # Seed, paths, eval settings, rclone sync
│   ├── datasets/
│   │   ├── baseline.yaml                  # No theory injection (HI-Small)
│   │   ├── rat.yaml                       # RAT features (HI-Small, low/medium/high)
│   │   └── slt.yaml                       # SLT features (HI-Small, low/medium/high)
│   └── models/
│       ├── graphsage.yaml
│       ├── graphsage_t.yaml
│       └── dyrep.yaml
│
├── ibm_transcations_datasets/             # Raw IBM CSVs (Git LFS tracked)
│   ├── HI-Small_Trans.csv
│   ├── HI-Small_accounts.csv
│   ├── RAT/                               # Output of rat_injector.py
│   └── SLT/                               # Output of slt_injector.py
│
├── graphs/                                # Static graphs (GraphSAGE / GraphSAGE-T)
│   └── <dataset_name>/
│       ├── edge_index.pt                  # [2, E]
│       ├── edge_attr.pt                   # [E, F_e]
│       ├── x.pt                           # [N, F_n]
│       ├── timestamps.pt                  # [E]
│       ├── y_edge.pt                      # [E] edge labels
│       ├── y_node.pt                      # [N] node labels
│       ├── node_mapping.json
│       ├── edge_attr_cols.json
│       └── graph_stats.json
│
├── graphs_dyrep/                          # DyRep event-stream graphs
│   └── <dataset_name>/
│       ├── src.pt, dst.pt, ts.pt          # Event triplets
│       ├── event_type.pt
│       ├── edge_attr.pt
│       ├── node_features.pt
│       ├── labels.pt, y_node.pt
│       └── *.json
│
├── graphs_tgat/                           # TGAT-format graphs (if used)
│
├── splits/                                # Train/val/test index splits for static graphs
│   └── <dataset_name>/
│       ├── train_edge_idx.pt
│       ├── val_edge_idx.pt
│       ├── test_edge_idx.pt
│       └── split_metadata.json
│
├── splits_dyrep/                          # Chronological splits for DyRep
├── splits_tgat/                           # Splits for TGAT
│
├── results/                               # Per-run metrics, predictions, configs
│   └── <dataset>/<experiment>/<model>/
│       ├── best_model.pt
│       ├── checkpoint.pt                  # Resume checkpoint
│       ├── metrics.json
│       ├── {train,val,test}_pred_probs.pt
│       └── experiment_config.json
│
├── logs/                                  # Stdout/stderr logs and TensorBoard events
│   └── <dataset>/<experiment>/<model>/
│       ├── <experiment>_<timestamp>.txt
│       ├── <experiment>_<timestamp>_errors.txt
│       └── tb/
│
├── scripts/
│   ├── create_splits.py                   # Auto-detects graph type, creates splits
│   ├── rat/                               # RAT theory injector + verification tools
│   │   ├── rat_injector.py
│   │   └── verification/
│   ├── SLT/                               # SLT theory injector
│   │   └── slt_injector.py
│   ├── strain/                            # STRAIN theory injector (experimental)
│   ├── graph/                             # Graph builders
│   │   ├── baseline_graph_builder.py
│   │   ├── motif_graph_builder_static.py
│   │   ├── baseline_dyrep_graph_builder.py
│   │   └── motif_dyrep_graph_builder.py
│   ├── training/                          # Model training entry points
│   │   ├── train_graphsage.py
│   │   ├── train_graphsage_t.py
│   │   └── train_dyrep.py
│   ├── ablations/                         # Ablation graph builders
│   │   ├── rat_ablation_groups.py
│   │   ├── run_ablation.py
│   │   └── run_all_ablation_graphs.py
│   ├── analysis/                          # Evaluation and visualisation
│   │   ├── analyze_dataset.py
│   │   ├── feature_importance.py
│   │   ├── plot_curves_from_saved.py
│   │   └── plot_dyrep_curves_from_saved.py
│   ├── utils/
│   │   ├── config_utils.py                # Config loading, path wiring, Logger
│   │   ├── evaluation_utils.py            # Imbalance-aware metrics
│   │   ├── checkpoint_utils.py            # Save/load checkpoints + rclone sync
│   │   └── parse_graphsage_t_logs.py
│   └── bash/                              # Convenience shell runners
│       ├── run_baseline.sh
│       ├── run_rat_all.sh
│       ├── run_slt_all.sh
│       ├── run_dyrep.sh
│       ├── run_ablations.sh
│       └── run_all.sh
│
├── tgn/                                   # TGN reference codebase (Rossi et al. 2020)
├── aml_project.yml                        # Minimal conda environment (working env)
├── environment_full.yml                   # Full pinned conda environment
├── requirements.txt                       # pip dependencies
└── dyrep_features.py                      # DyRep feature utilities
```

---

## 3. Prerequisites

### Hardware

- **GPU recommended.** Configs target CUDA and were tuned on an RTX 4080 (batch size 8192). Training scripts fall back to CPU if CUDA is unavailable, but a full run on CPU is impractical.
- **RAM:** 16 GB minimum (the RAT and SLT injectors hold the full transaction DataFrame in memory).
- **Disk:** ~5 GB free for datasets, graphs, splits, results, and logs.

### Software

- **Python 3.10**
- **CUDA 12.1** (to match `torch==2.5.1+cu121`; CPU-only builds also work)
- **Git LFS** (the IBM CSV files are LFS-tracked)
- **Conda** (recommended) or a `python3.10` virtualenv

---

## 4. Installation

For AWS Slurm:
- Load/source conda before activating:
  source /opt/miniconda/etc/profile.d/conda.sh
  conda activate aml_project

- Verify that `which python` points to:
  /home/<user>/.conda/envs/aml_project/bin/python

- Use the appropriate Slurm GPU header to avoid Slurm-specific issues.  

### Option A — Conda minimal (recommended, working env)

`aml_project.yml` is the environment used for this project.

```bash
git lfs install
git clone <repo-url>
cd Motif-Aware-Temporal-GNNs-for-Anti-Money-Laundering-Detection

conda env create -f aml_project.yml
conda activate aml_project
pip install -r requirements.txt
```

### Option B — Conda full pinned

Captures every exact package version used during development:

```bash
conda env create -f environment_full.yml
conda activate aml_project
```

### Option C — pip only

```bash
python3.10 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```


> **PyTorch CUDA wheels.** `requirements.txt` pins `torch==2.5.1+cu121`. If the install fails on the torch lines, install them directly from the PyTorch index:
>
> ```bash
> pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
>     --index-url https://download.pytorch.org/whl/cu121
> ```
>
> If `pip install -r requirements.txt` still retries the PyTorch packages from the default PyPI index, install the PyTorch packages first and then install the remaining requirements using a temporary filtered file:
>
> ```bash
> pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
>     --index-url https://download.pytorch.org/whl/cu121
>
> grep -v -E '^(torch|torchvision|torchaudio)==' requirements.txt > requirements_no_torch.txt
> pip install -r requirements_no_torch.txt
> ```
>
> For CPU-only or different CUDA versions, adjust the `+cu121` suffix and index URL (see https://pytorch.org/get-started/previous-versions/).

### Verify

```bash
python -c "import torch, torch_geometric, sklearn, pandas; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

### Key packages

| Package | Version | Purpose |
|---|---|---|
| `python` | 3.10 | Runtime |
| `torch` | 2.5.1+cu121 | Deep learning |
| `torch-geometric` | 2.7.0 | `SAGEConv` and graph utilities |
| `numpy` | 1.26.4 | Arrays |
| `pandas` | 2.3.3 | IBM CSV processing |
| `scikit-learn` | 1.7.2 | Splits, metrics, Random Forest |
| `pyarrow` | 23.0.0 | Parquet I/O in graph builders |
| `tensorboard` | 2.20.0 | Training curves |
| `matplotlib` / `seaborn` | 3.10.6 / 0.13.2 | Plots |
| `umap-learn` | 0.5.3 | RAT feature visualisation |
| `pyyaml` | 6.0.3 | Config loading |
| `tqdm` | 4.67.1 | Progress bars |
| `psutil` | latest | Memory stamps in graph builders |

---

## 5. Data & Pre-built Graphs

### 5.1 IBM transaction dataset

Download the IBM AML synthetic transaction dataset from Kaggle (*IBM Transactions for Anti Money Laundering*) and place the files at:

```
ibm_transcations_datasets/
├── HI-Small_Trans.csv          # ~5M transactions, ~515K nodes, ~0.1% laundering
├── HI-Small_accounts.csv
└── HI-Small_Patterns.txt       # optional
```

> The CSVs are tracked via Git LFS. After cloning, run `git lfs pull` to materialise them.

### 5.2 Pre-built graphs and splits

To skip the graph-building stage, download the pre-built tensors from Google Drive:

- **All graphs:** https://drive.google.com/drive/folders/1ZQMZdWmBJ0xpb0u2s3kI0ZjMw6AYYI1u?usp=sharing

Extract into the repo root. The repo ships static graphs for all RAT and SLT intensities (`graphs/HI-Small_Trans*`), DyRep event graphs (`graphs_dyrep/HI-Small_Trans*`), and 9 RAT ablation variants (`graphs_dyrep/HI-Small_Trans_RAT_medium__*`).

---

## 6. End-to-End Pipeline

```
ibm_transcations_datasets/
  Raw IBM CSVs
       │
       ▼
scripts/rat/rat_injector.py          ← RAT + motif features
scripts/SLT/slt_injector.py          ← SLT peer-exposure features
       │
       ▼  ibm_transcations_datasets/{RAT,SLT}/*.csv
       │
       ▼
scripts/graph/
  baseline_graph_builder.py          ← no theory features
  motif_graph_builder_static.py      ← theory + motif features (GraphSAGE / GraphSAGE-T)
  motif_dyrep_graph_builder.py       ← theory + motif features (DyRep)
       │
       ▼  graphs/<name>/ or graphs_dyrep/<name>/
       │
       ▼
scripts/create_splits.py             ← 60/20/20 stratified or chronological splits
       │
       ▼  splits/<name>/ or splits_dyrep/<name>/
       │
       ▼
scripts/training/
  train_graphsage.py
  train_graphsage_t.py
  train_dyrep.py
       │
       ▼  results/<dataset>/<experiment>/<model>/
          logs/<dataset>/<experiment>/<model>/
```

---

## 7. Running Experiments

All training scripts share the same CLI. **Always run from the project root.**

```bash
python scripts/training/<script>.py \
    --config       configs/models/<model>.yaml \
    --dataset      configs/datasets/<dataset>.yaml \
    --base_config  configs/base.yaml \
    [--intensity   low|medium|high]
```

`--intensity` is required for RAT and SLT datasets; omit it for `baseline`.

### 7.1 GraphSAGE

```bash
# Baseline (no theory)
python scripts/training/train_graphsage.py \
    --config configs/models/graphsage.yaml \
    --dataset configs/datasets/baseline.yaml \
    --base_config configs/base.yaml

# RAT low intensity
python scripts/training/train_graphsage.py \
    --config configs/models/graphsage.yaml \
    --dataset configs/datasets/rat.yaml \
    --base_config configs/base.yaml \
    --intensity low
```

### 7.2 GraphSAGE-T

```bash
python scripts/training/train_graphsage_t.py \
    --config configs/models/graphsage_t.yaml \
    --dataset configs/datasets/rat.yaml \
    --base_config configs/base.yaml \
    --intensity medium
```

### 7.3 SLT dataset

Swap `configs/datasets/rat.yaml` for `configs/datasets/slt.yaml` — all three model scripts work unchanged:

```bash
python scripts/training/train_graphsage.py \
    --config configs/models/graphsage.yaml \
    --dataset configs/datasets/slt.yaml \
    --base_config configs/base.yaml \
    --intensity medium
```

The default SLT config targets `HI-Small_Trans_SLT`. Graphs and splits land in `graphs/HI-Small_Trans_SLT_<intensity>/` and `splits/HI-Small_Trans_SLT_<intensity>/`.

### 7.4 DyRep

> **Note:** DyRep training is implemented but the temporal split pipeline is still being optimised. Use with the DyRep graphs and splits in `graphs_dyrep/` / `splits_dyrep/`.

```bash
python scripts/training/train_dyrep.py \
    --config configs/models/dyrep.yaml \
    --dataset configs/datasets/rat.yaml \
    --base_config configs/base.yaml \
    --intensity medium
```

### 7.5 Resuming interrupted runs

All training scripts write a `checkpoint.pt` to `results/<dataset>/<experiment>/<model>/` at the end of every epoch. Rerunning the same command automatically resumes from the last checkpoint.

### 7.6 CLI flag reference

| Flag | Required | Description |
|---|---|---|
| `--config` | yes | Model YAML (`configs/models/*.yaml`) |
| `--dataset` | yes | Dataset YAML (`configs/datasets/*.yaml`) |
| `--base_config` | no (default: `configs/base.yaml`) | Seed, paths, eval settings |
| `--intensity` | required for RAT and SLT | `low`, `medium`, or `high` |

---

## 8. Configuration Reference

Config loading and path wiring lives in `scripts/utils/config_utils.py::setup_experiment`. It merges `base`, `model`, and `dataset` YAMLs, seeds all RNG sources, and resolves output paths:

- `graph_folder  = <graphs_dir>/<prefix>_<intensity>`
- `split_folder  = <splits_dir>/<prefix>_<intensity>`
- `results_dir   = <results_dir>/<dataset>/seed<seed>_<name>/<model>`
- `logs_dir      = <logs_dir>/<dataset>/seed<seed>_<name>/<model>`

### `configs/base.yaml`

```yaml
experiment:
  seed: 1                    # change to run with a different seed
  name: "seed1_experiment"   # affects output directory names

evaluation:
  threshold: 0.5
  auto_threshold: true       # tunes threshold on the val set
  compute_top_k: true
  top_k_values: [100, 500, 1000]

sync:
  enabled: true              # set false to disable rclone sync to Google Drive
  rclone_remote: "gdrive"
  drive_root: "AML_project_results"
  sync_every_n_epochs: 10
  rclone_path: ""            # set to full path of rclone binary if not on PATH
```

### `configs/models/graphsage.yaml` / `graphsage_t.yaml`

```yaml
model:
  hidden_dim: 128
  num_layers: 2
  aggregator: "mean"         # mean / max / lstm
  dropout: 0.2
  # GraphSAGE-T also has:
  time_encoder: "sinusoidal"
  time_dim: 32

training:
  device: "cuda"
  batch_size: 8192
  eval_batch_size: 16384
  lr: 0.0001                 # GraphSAGE; GraphSAGE-T uses 0.0005
  weight_decay: 0.0001
  epochs: 350
  early_stopping_patience: 25
  gradient_clip: 1.0

loss:
  type: "bce"
  pos_weight: null            # null = auto-computed from train split (capped at 100)
```

### `configs/datasets/rat.yaml` / `slt.yaml`

```yaml
# rat.yaml
dataset:
  theory: "RAT"
  prefix: "HI-Small_Trans_RAT"
  requires_intensity: true

# slt.yaml
dataset:
  theory: "SLT"
  prefix: "HI-Small_Trans_SLT"
  requires_intensity: true
```

---

## 9. Outputs

### Results directory: `results/<dataset>/<experiment>/<model>/`

| File | Contents |
|---|---|
| `best_model.pt` | Model weights at best validation AUPR |
| `checkpoint.pt` | Latest epoch checkpoint (for resuming) |
| `metrics.json` | Train/val/test metrics, best epoch, timing |
| `train_pred_probs.pt` | Per-edge predicted probabilities (train) |
| `val_pred_probs.pt` | Per-edge predicted probabilities (val) |
| `test_pred_probs.pt` | Per-edge predicted probabilities (test) |
| `experiment_config.json` | Full merged config snapshot |

### Logs directory: `logs/<dataset>/<experiment>/<model>/`

| File | Contents |
|---|---|
| `<experiment>_<timestamp>.txt` | Full stdout |
| `<experiment>_<timestamp>_errors.txt` | Stderr |
| `tb/` | TensorBoard events (`Loss/train`, `Val/F1`, `Val/AUPR`, `Time/epoch_seconds`) |

```bash
tensorboard --logdir logs
```

### Metrics

All metrics are imbalance-aware: precision, recall, F1, AUPR (average precision), ROC-AUC, balanced accuracy, MCC, Cohen's kappa, confusion matrix counts, auto-tuned operating threshold, and precision@k / recall@k for k ∈ {100, 500, 1000}.

### rclone sync to Google Drive

If `sync.enabled: true` in `base.yaml` and `rclone` has been configured (`rclone config`), results and logs are pushed to Google Drive every `sync_every_n_epochs` epochs and once at the end. To find your `rclone` path after a conda install:

```bash
# Windows
where rclone
# Linux / macOS
which rclone
```

Set `sync.rclone_path` in `base.yaml` to that path, or disable sync entirely with `sync.enabled: false`.

---

## 10. Building Graphs from Scratch

Only needed if you are not using the pre-built graphs from Drive. All scripts resolve their input/output paths from their own file location, so they work on any OS without path editing as long as the repo layout is intact.

### Step 1 — Theory feature injection

Both injectors read `ibm_transcations_datasets/HI-Small_Trans.csv` and write three intensity variants (`low`, `medium`, `high`). Intensity is set by a percentile threshold on the theory score over laundering-positive rows.

#### RAT injector

Computes Routine Activity Theory sub-scores (offender, target, guardian) and graph motif features (fan-in, fan-out, chain, cycle):

```bash
python scripts/rat/rat_injector.py
```

Output:
```
ibm_transcations_datasets/RAT/
├── HI-Small_Trans_RAT_low.csv
├── HI-Small_Trans_RAT_medium.csv
└── HI-Small_Trans_RAT_high.csv
```

#### SLT injector

Computes Social Learning Theory peer-exposure features (suspicious neighbour ratio, amount share, strong-tie ratio, and temporal lags):

```bash
python scripts/SLT/slt_injector.py
```

Output:
```
ibm_transcations_datasets/SLT/
├── HI-Small_Trans_SLT_low.csv
├── HI-Small_Trans_SLT_medium.csv
└── HI-Small_Trans_SLT_high.csv
```

Optional weight arguments (defaults shown):

```bash
python scripts/SLT/slt_injector.py \
    --w_neighbor 0.30 --w_amount 0.25 --w_strong_tie 0.20 \
    --w_delta 0.15 --w_cum 0.10
```

### Step 2 — Build graph tensors

**Static graphs (GraphSAGE / GraphSAGE-T):**

```bash
# No theory features
python scripts/graph/baseline_graph_builder.py

# RAT features
python scripts/graph/motif_graph_builder_static.py \
    --dataset RAT/HI-Small_Trans_RAT_low.csv

# SLT features
python scripts/graph/motif_graph_builder_static.py \
    --dataset SLT/HI-Small_Trans_SLT_low.csv
```

**DyRep event graphs:**

```bash
python scripts/graph/baseline_dyrep_graph_builder.py

# RAT
python scripts/graph/motif_dyrep_graph_builder.py \
    --dataset RAT/HI-Small_Trans_RAT_medium.csv

# SLT
python scripts/graph/motif_dyrep_graph_builder.py \
    --dataset SLT/HI-Small_Trans_SLT_medium.csv
```

The `--dataset` path is relative to `ibm_transcations_datasets/`. Output lands in `graphs/<dataset_name>/` or `graphs_dyrep/<dataset_name>/` respectively.

### Step 3 — Create splits

`create_splits.py` auto-detects the graph type:

- **Static** (`edge_index.pt` present) → stratified random 60/20/20 split on `y_edge`
- **DyRep** (`src.pt + dst.pt + ts.pt` present) → chronological 60/20/20

```bash
python scripts/create_splits.py \
    --graph_folder graphs/HI-Small_Trans_RAT_low \
    --out_dir splits/HI-Small_Trans_RAT_low \
    --train_ratio 0.60 --val_ratio 0.20 --test_ratio 0.20 \
    --seed 42
```

If `--out_dir` is omitted, it defaults to `splits/<dataset_name>` (or `splits_dyrep/<dataset_name>` for DyRep graphs).

---

## 11. Ablation Studies

### 11.1 RAT feature-group ablations

Ablations strip feature groups from an existing graph's `edge_attr.pt` and write a sibling graph directory. This lets you train on partial feature sets without re-running injection or graph building.

### Defined feature groups (`scripts/ablations/rat_ablation_groups.py`)

`no_struct`, `no_temp`, `no_amount`, `no_burst_pattern`, `no_entity`, `no_rat_scores`, `no_motif`, `no_crossbank`, `top20_features` (positive selection of top-20 RF-important features).

### Build all ablation graphs at once

```bash
cd scripts/ablations
python run_all_ablation_graphs.py
```

By default, reads from `graphs_dyrep/HI-Small_Trans_RAT_medium` and writes sibling folders like `graphs_dyrep/HI-Small_Trans_RAT_medium__no_motif`.

### Build a single ablation

```python
from scripts.ablations.run_ablation import run_ablation
from scripts.ablations.rat_ablation_groups import FULL_FEATURES, NO_MOTIF

keep = [f for f in FULL_FEATURES if f not in NO_MOTIF]
run_ablation(
    full_graph_dir="graphs_dyrep/HI-Small_Trans_RAT_medium",
    output_dir="graphs_dyrep/HI-Small_Trans_RAT_medium__no_motif",
    keep_features=keep,
)
```

### Train on an ablation graph

Point the dataset `prefix` in `configs/datasets/rat.yaml` at the ablation folder name. See `scripts/bash/run_ablations.sh` for the reference pattern.

### 11.2 SLT weight ablations

SLT ablations work differently from RAT ablations — instead of stripping feature groups, they vary the **weight parameters** of the SLT score itself across 5 variants to test which peer-exposure signals matter most.

**Variants and their weights** (`w_neighbor`, `w_amount`, `w_strong_tie`, `w_delta`, `w_cum`):

| Variant | neighbor | amount | strong_tie | delta | cum |
|---|---|---|---|---|---|
| `current` | 0.30 | 0.25 | 0.20 | 0.15 | 0.10 |
| `equal` | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |
| `neighbor_heavy` | 0.40 | 0.20 | 0.15 | 0.15 | 0.10 |
| `amount_heavy` | 0.20 | 0.40 | 0.15 | 0.15 | 0.10 |
| `temporal_heavy` | 0.20 | 0.15 | 0.15 | 0.25 | 0.25 |

**Step 1 — Build the SLT variant graphs.**

Run `slt_injector.py` once per variant with its weights, then build a static graph for each. The `create_slt_ablation_variants.sh` script automates this:

```bash
bash scripts/bash/create_slt_ablation_variants.sh
```

Each variant produces graphs at `graphs/HI-Small_Trans_SLT_<variant>_{low,medium,high}/`.

**Step 2 — Train across all variants, intensities, and seeds.**

`run_slt_ablations.sh` runs GraphSAGE-T with 5 seeds × 3 intensities × 5 variants (75 runs total). It temporarily patches `configs/base.yaml` for each seed/experiment name and restores it on exit:

```bash
bash scripts/bash/run_slt_ablations.sh
```

Logs land in `scripts/bash/logs/slt_ablations_training_<timestamp>.log` and a `.done` marker is created on completion.

> **Note:** `run_slt_ablations.sh` has a hardcoded `PYTHON_EXE` path for the original development machine. Before running, either update that path or replace it with `python` if your conda env is already activated.

---

## 12. Analysis & Utilities

| Script | Purpose |
|---|---|
| `scripts/analysis/analyze_dataset.py` | Quick stats on raw IBM CSVs (nodes, edges, banks, % laundering) |
| `scripts/analysis/feature_importance.py` | Random Forest feature importance ranking on a RAT/SLT CSV |
| `scripts/analysis/plot_curves_from_saved.py` | Load saved `test_pred_probs.pt` + `y_edge.pt` to draw ROC and PR curves |
| `scripts/analysis/plot_dyrep_curves_from_saved.py` | Same for DyRep results |
| `scripts/rat/verification/rat_verify_structure.py` | Sanity-check RAT-injected CSV columns |
| `scripts/rat/verification/rat_plot_distributions.py` | Per-feature histograms |
| `scripts/rat/verification/rat_plot_umap.py` | UMAP projection of RAT feature space |
| `scripts/utils/evaluation_utils.py` | `evaluate_binary_classifier(...)` — shared by all training scripts |
| `scripts/utils/checkpoint_utils.py` | Save/load checkpoints; rclone sync helpers |
| `scripts/utils/parse_graphsage_t_logs.py` | Extract per-epoch metrics from log files |
| `scripts/utils/recover_graphsage_t_metrics.py` | Rebuild `metrics.json` from saved pred probs if a run died mid-save |

**Example — feature importance on RAT-medium:**

```bash
python scripts/analysis/feature_importance.py \
    --csv_path ibm_transcations_datasets/RAT/HI-Small_Trans_RAT_medium.csv \
    --label_col "Is Laundering" \
    --top_k 20 \
    --out_dir results/feature_importance
```

**Example — feature importance on SLT-medium:**

```bash
python scripts/analysis/feature_importance.py \
    --csv_path ibm_transcations_datasets/SLT/HI-Small_Trans_SLT_medium.csv \
    --label_col "Is Laundering" \
    --top_k 20 \
    --out_dir results/feature_importance
```

---

## 13. Bash Driver Scripts

The scripts under `scripts/bash/` chain multiple training runs with timestamped logs. They auto-detect conda across common install locations (Linux/macOS/Windows-GitBash). Override defaults with environment variables:

```bash
CONDA_EXE=/path/to/conda CONDA_ENV=my_env bash scripts/bash/run_rat_all.sh
```

| Script | What it runs |
|---|---|
| `run_baseline.sh` | Baseline dataset (no theory), all models |
| `run_rat_all.sh` | GraphSAGE + GraphSAGE-T on RAT low / medium / high |
| `run_slt_all.sh` | GraphSAGE + GraphSAGE-T on SLT low / medium / high |
| `run_dyrep.sh` | DyRep on baseline + RAT low / medium / high |
| `run_ablations.sh` | DyRep on all 9 RAT-medium ablation graphs |
| `run_all.sh` | All models × all datasets in one sweep |

Always run from the project root:

```bash
bash scripts/bash/run_rat_all.sh
```

A `.done` marker file is created on successful completion (e.g. `logs/RAT_ALL_FINISHED_<timestamp>.done`).

---

## 14. Troubleshooting

**`torch.cuda.is_available()` returns `False`.**
Check the wheel with `pip show torch` — the version should include `+cu121`. Verify your NVIDIA driver supports CUDA 12.1 (`nvidia-smi`). Reinstall via the PyTorch index URL in [§4](#4-installation).

**`FileNotFoundError: graphs/HI-Small_Trans_RAT_low/edge_index.pt`**
Either extract the pre-built graphs from Drive, or run the graph builder. Also make sure `--intensity low` was passed — the resolved path is `graphs/{prefix}_{intensity}`.

**`FileNotFoundError: splits/...`**
Run `scripts/create_splits.py` against the matching graph folder to generate splits for any intensity.

**CUDA out of memory.**
Lower `training.batch_size` and `training.eval_batch_size` in the model YAML. The defaults assume ~16 GB VRAM.

**CPU out of memory during graph build.**
The motif graph builder and injectors hold the full transaction DataFrame in memory. 16 GB RAM should be sufficient for HI-Small.

**`run_*.sh` errors with "could not locate a conda install".**
Set `CONDA_EXE` to the path of your `conda` binary:

```bash
CONDA_EXE=/path/to/conda bash scripts/bash/run_rat_all.sh
```

If your env is not named `aml_project`, also set `CONDA_ENV=<your_env_name>`.

**DyRep training errors or hangs.**
The temporal split script is still being optimised. Use the pre-built DyRep graphs (`graphs_dyrep/HI-Small_Trans*`) and their splits in `splits_dyrep/`.

**A graph builder or injector raises `FileNotFoundError`.**
All scripts resolve paths from their own location using `Path(__file__).resolve().parents[N]`. If a script has been moved out of its expected subdirectory, either move it back or update the `parents[N]` index accordingly.

**rclone sync errors.**
Set `sync.enabled: false` in `configs/base.yaml` to disable. To re-enable, run `rclone config` to set up the `gdrive` remote and set `sync.rclone_path` to the full path of your `rclone` binary.
