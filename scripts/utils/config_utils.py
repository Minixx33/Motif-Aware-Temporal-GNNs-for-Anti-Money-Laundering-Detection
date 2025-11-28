# ========================================================================
# config_utils.py  (FULLY PATCHED VERSION)
# ========================================================================
# Includes:
#   ✓ Seed-aware experiment names
#   ✓ Optional experiment ID
#   ✓ Non-overwriting results/log paths
#   ✓ Backwards-compatible with all existing scripts
# ========================================================================

import os
import sys
import json
import yaml
from datetime import datetime
from typing import Dict, Any, Optional
import torch
import numpy as np


# ========================================================================
# LOGGER
# ========================================================================

class Logger:
    def __init__(self, log_file: str, mode: str = "w"):
        self.terminal = sys.stdout
        self.log = open(log_file, mode, encoding="utf-8")
        sys.stdout = self

        self.stderr_terminal = sys.stderr
        self.stderr_log = open(
            log_file.replace(".txt", "_errors.txt"), mode, encoding="utf-8"
        )
        sys.stderr = self._StderrRedirector(self.stderr_terminal, self.stderr_log)

        self.write("\n" + "=" * 80 + "\n")
        self.write(f"Logging started at: {datetime.now()}\n")
        self.write(f"Log file: {log_file}\n")
        self.write("=" * 80 + "\n\n")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.write("\n" + "=" * 80 + "\n")
        self.write(f"Logging ended at: {datetime.now()}\n")
        self.write("=" * 80 + "\n")

        sys.stdout = self.terminal
        sys.stderr = self.stderr_terminal
        self.log.close()
        self.stderr_log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    class _StderrRedirector:
        def __init__(self, terminal, log_file):
            self.terminal = terminal
            self.log = log_file

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()


def setup_logging(log_dir: str, experiment_name: str, timestamp: bool = True) -> str:
    os.makedirs(log_dir, exist_ok=True)

    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{experiment_name}_{ts}.txt"
    else:
        name = f"{experiment_name}.txt"

    return os.path.join(log_dir, name)


# ========================================================================
# YAML LOADING
# ========================================================================

def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data else {}


def load_and_merge_configs(model_config_path, dataset_config_path, base_config_path):
    return {
        "base": load_yaml(base_config_path),
        "model": load_yaml(model_config_path),
        "dataset": load_yaml(dataset_config_path),
    }


# ========================================================================
# VALIDATION
# ========================================================================

def validate_intensity(dataset_cfg, intensity):
    theory = dataset_cfg["dataset"]["theory"]
    requires = dataset_cfg["dataset"].get("requires_intensity", True)

    if requires and intensity is None:
        allowed = dataset_cfg["dataset"].get(
            "available_intensities", ["low", "medium", "high"]
        )
        raise ValueError(
            f"Dataset {theory} requires intensity. Allowed: {allowed}"
        )

    if intensity:
        allowed = dataset_cfg["dataset"].get(
            "available_intensities", ["low", "medium", "high"]
        )
        if intensity not in allowed:
            raise ValueError(
                f"Invalid intensity '{intensity}'. Allowed: {allowed}"
            )

    return True


def validate_config(base_cfg, model_cfg, dataset_cfg, intensity):
    validate_intensity(dataset_cfg, intensity)

    for key in ["paths", "evaluation"]:
        if key not in base_cfg:
            raise ValueError(f"Missing key in base config: {key}")

    for key in ["model", "training", "loss"]:
        if key not in model_cfg:
            raise ValueError(f"Missing key in model config: {key}")

    if "dataset" not in dataset_cfg:
        raise ValueError("Dataset config missing 'dataset' section")

    print("[INFO] Configuration validation passed.")
    return True


# ========================================================================
# REPRODUCIBILITY
# ========================================================================

def set_seed(seed: int = 42, deterministic: bool = False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    try:
        import random
        random.seed(seed)
    except Exception:
        pass

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[INFO] Seed set to {seed}")


# ========================================================================
# EXPERIMENT NAMING (PATCHED)
# ========================================================================

def build_experiment_name(
    dataset_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    intensity: Optional[str] = None,
    seed: Optional[int] = None,
    exp_id: Optional[int] = None,
    timestamp: bool = False
) -> str:
    """
    NEW VERSION — supports:
       ✓ seed-aware directory naming
       ✓ exp_id (experiment number)
    """

    theory = dataset_cfg["dataset"]["theory"]
    prefix = dataset_cfg["dataset"]["prefix"]
    model_name = model_cfg["model"]["name"].replace(" ", "").replace("-", "_")

    if theory.lower() == "baseline":
        base = prefix
    else:
        base = f"{prefix}_{intensity}"

    # ---- PATCHED LOGIC ----
    if seed is not None and exp_id is not None:
        name = f"{base}_seed{seed}_exp{exp_id}"
    elif seed is not None:
        name = f"{base}_seed{seed}"
    else:
        name = base

    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{name}_{ts}"

    return name


# ========================================================================
# PATHS (UNCHANGED EXCEPT FOR EXPERIMENT NAME USAGE)
# ========================================================================

def build_paths(base_cfg, dataset_cfg, intensity, model_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    root = base_cfg["paths"].get("root", project_root)
    if not os.path.isabs(root):
        root = os.path.abspath(root)

    theory = dataset_cfg["dataset"]["theory"]
    prefix = dataset_cfg["dataset"]["prefix"]

    if theory.lower() == "baseline":
        dataset_name = prefix
    else:
        dataset_name = f"{prefix}_{intensity}"

    if "tgat" in model_name.lower():
        graph_dir_key = "tgat_graphs_dir"
        split_dir_key = "tgat_splits_dir"
    else:
        graph_dir_key = "graphs_dir"
        split_dir_key = "splits_dir"

    graphs_dir = os.path.join(root, base_cfg["paths"][graph_dir_key])
    splits_dir = os.path.join(root, base_cfg["paths"][split_dir_key])
    results_root = os.path.join(root, base_cfg["paths"]["results_dir"])
    logs_root = os.path.join(root, base_cfg["paths"]["logs_dir"])

    graph_folder = os.path.join(graphs_dir, dataset_name)
    split_folder = os.path.join(splits_dir, dataset_name)

    results_dir = os.path.join(results_root, dataset_name, model_name)
    logs_dir = os.path.join(logs_root, dataset_name, model_name)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    return {
        "dataset_name": dataset_name,
        "graph_folder": graph_folder,
        "split_folder": split_folder,
        "results_dir": results_dir,
        "logs_dir": logs_dir,
        "root": root,
    }


# ========================================================================
# SAVE EXPERIMENT CONFIG
# ========================================================================

def save_experiment_config(
    save_dir: str,
    base_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    intensity: Optional[str] = None,
    additional_info: Optional[Dict[str, Any]] = None,
    filename: str = "experiment_config.json"
):
    config = {
        "timestamp": datetime.now().isoformat(),
        "base_config": base_cfg,
        "model_config": model_cfg,
        "dataset_config": dataset_cfg,
        "intensity": intensity,
    }

    if additional_info:
        config["additional_info"] = additional_info

    path = os.path.join(save_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return path


# ========================================================================
# PRINT SUMMARY
# ========================================================================

def print_config_summary(base_cfg, model_cfg, dataset_cfg, intensity, paths):
    print("=" * 80)
    print("EXPERIMENT CONFIG SUMMARY")
    print("=" * 80)

    print("\n[DATASET]")
    print(f"  Name:       {paths['dataset_name']}")
    print(f"  Theory:     {dataset_cfg['dataset']['theory']}")
    print(f"  Intensity:  {intensity}")

    print("\n[MODEL]")
    print(f"  {model_cfg['model']['name']}")
    print(f"  Hidden dim: {model_cfg['model'].get('hidden_dim')}")

    print("\n[TRAINING]")
    t = model_cfg["training"]
    print(f"  LR:         {t.get('lr')}")
    print(f"  Epochs:     {t.get('epochs')}")

    print("\n[PATHS]")
    print(f"  Graphs:     {paths['graph_folder']}")
    print(f"  Splits:     {paths['split_folder']}")
    print(f"  Results:    {paths['results_dir']}")
    print(f"  Logs:       {paths['logs_dir']}")

    print("=" * 80)


# ========================================================================
# SETUP EXPERIMENT (PATCHED)
# ========================================================================

def setup_experiment(
    model_config_path: str,
    dataset_config_path: str,
    intensity: Optional[str] = None,
    base_config_path: str = "configs/base.yaml",
    verbose: bool = True,
    enable_logging: bool = True
):
    configs = load_and_merge_configs(
        model_config_path, dataset_config_path, base_config_path
    )

    base_cfg = configs["base"]
    model_cfg = configs["model"]
    dataset_cfg = configs["dataset"]

    validate_config(base_cfg, model_cfg, dataset_cfg, intensity)

    seed = base_cfg.get("experiment", {}).get("seed", 42)
    exp_id = base_cfg.get("experiment", {}).get("exp_id", None)

    set_seed(seed)

    model_name = model_cfg["model"]["name"].lower().replace(" ", "_")
    paths = build_paths(base_cfg, dataset_cfg, intensity, model_name)

    experiment_name = build_experiment_name(
        dataset_cfg,
        model_cfg,
        intensity,
        seed=seed,
        exp_id=exp_id,
        timestamp=False
    )

    results_dir = os.path.join(paths["results_dir"], experiment_name)
    logs_dir = os.path.join(paths["logs_dir"], experiment_name)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    paths["results_dir"] = results_dir
    paths["logs_dir"] = logs_dir

    logger = None
    log_path = None
    if enable_logging:
        log_path = setup_logging(logs_dir, experiment_name, timestamp=True)
        logger = Logger(log_path)
        print(f"[INFO] Logging to: {log_path}")

    if verbose:
        print_config_summary(base_cfg, model_cfg, dataset_cfg, intensity, paths)
        device = torch.device(
            model_cfg["training"].get("device", "cuda")
            if torch.cuda.is_available() else "cpu"
        )
        print(f"\n[DEVICE] {device}\n")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return {
        "base_cfg": base_cfg,
        "model_cfg": model_cfg,
        "dataset_cfg": dataset_cfg,
        "device": device,
        "paths": paths,
        "experiment_name": experiment_name,
        "intensity": intensity,
        "logger": logger,
        "log_path": log_path,
    }
