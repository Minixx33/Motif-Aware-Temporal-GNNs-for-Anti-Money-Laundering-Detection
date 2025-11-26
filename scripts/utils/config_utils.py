"""
config_utils.py
---------------
Utility functions for loading, merging, and validating YAML configurations
for the AML detection experiments.

Functions:
    - load_yaml: Load a single YAML file
    - load_and_merge_configs: Load base + model + dataset configs
    - validate_config: Validate configuration consistency
    - validate_intensity: Check intensity requirements
    - build_paths: Construct experiment paths
    - set_seed: Set random seeds for reproducibility
    - save_experiment_config: Save complete experiment configuration
    - build_experiment_name: Generate experiment name from configs
    - setup_logging: Setup logging to both file and terminal
    - Logger: Context manager for dual output
"""

import os
import sys
import json
import yaml
from datetime import datetime
from typing import Dict, Any, Optional, List, TextIO
import torch
import numpy as np


# ------------------------------------------------------------
# Logging Utilities
# ------------------------------------------------------------
class Logger:
    """
    Logger class that writes to both terminal and file simultaneously.
    
    Usage:
        logger = Logger("path/to/logfile.txt")
        print("This will go to both terminal and file")
        logger.close()
        
    Or use as context manager:
        with Logger("path/to/logfile.txt") as logger:
            print("This will be logged")
    """
    def __init__(self, log_file: str, mode: str = 'w'):
        """
        Initialize logger.
        
        Args:
            log_file: Path to log file
            mode: File mode ('w' for write, 'a' for append)
        """
        self.terminal = sys.stdout
        self.log = open(log_file, mode, encoding='utf-8')
        sys.stdout = self
        
        # Also capture stderr if needed
        self.stderr_terminal = sys.stderr
        self.stderr_log = open(log_file.replace('.txt', '_errors.txt'), mode, encoding='utf-8')
        sys.stderr = self._StderrRedirector(self.stderr_terminal, self.stderr_log)
        
        # Write header
        self.write(f"\n{'='*80}\n")
        self.write(f"Logging started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.write(f"Log file: {log_file}\n")
        self.write(f"{'='*80}\n\n")
    
    def write(self, message):
        """Write message to both terminal and file"""
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
    
    def flush(self):
        """Flush both outputs"""
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        """Close logger and restore stdout"""
        self.write(f"\n{'='*80}\n")
        self.write(f"Logging ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.write(f"{'='*80}\n")
        
        sys.stdout = self.terminal
        sys.stderr = self.stderr_terminal
        self.log.close()
        self.stderr_log.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False
    
    class _StderrRedirector:
        """Helper class to redirect stderr"""
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


def setup_logging(
    log_dir: str,
    experiment_name: str,
    timestamp: bool = True
) -> str:
    """
    Setup logging to file with optional timestamp.
    
    Args:
        log_dir: Directory to save log file
        experiment_name: Base name for log file
        timestamp: If True, append timestamp to filename
        
    Returns:
        Path to log file
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Build log filename
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{experiment_name}_{ts}.txt"
    else:
        log_filename = f"{experiment_name}.txt"
    
    log_path = os.path.join(log_dir, log_filename)
    
    return log_path


# ------------------------------------------------------------
# YAML Loading
# ------------------------------------------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Dictionary with configuration
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
            return config if config is not None else {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {path}: {e}")


def load_and_merge_configs(
    model_config_path: str,
    dataset_config_path: str,
    base_config_path: str = "configs/base.yaml"
) -> Dict[str, Any]:
    """
    Load and merge base, model, and dataset configurations.
    
    Merge priority: base < model < dataset (later overrides earlier)
    However, we keep them separate in the returned dict for clarity.
    
    Args:
        model_config_path: Path to model config (e.g., configs/models/graphsage.yaml)
        dataset_config_path: Path to dataset config (e.g., configs/datasets/rat.yaml)
        base_config_path: Path to base config (default: configs/base.yaml)
        
    Returns:
        Dictionary with keys: 'base', 'model', 'dataset'
    """
    base_cfg = load_yaml(base_config_path)
    model_cfg = load_yaml(model_config_path)
    dataset_cfg = load_yaml(dataset_config_path)
    
    return {
        "base": base_cfg,
        "model": model_cfg,
        "dataset": dataset_cfg,
    }


# ------------------------------------------------------------
# Configuration Validation
# ------------------------------------------------------------
def validate_intensity(
    dataset_cfg: Dict[str, Any],
    intensity: Optional[str]
) -> bool:
    """
    Validate intensity argument based on dataset requirements.
    
    Args:
        dataset_cfg: Dataset configuration dictionary
        intensity: Intensity level (low/medium/high) or None
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If intensity validation fails
    """
    theory = dataset_cfg["dataset"]["theory"]
    requires_intensity = dataset_cfg["dataset"].get("requires_intensity", True)
    
    # Check if intensity is required but not provided
    if requires_intensity and intensity is None:
        available = dataset_cfg["dataset"].get("available_intensities", ["low", "medium", "high"])
        raise ValueError(
            f"Dataset '{theory}' requires --intensity argument. "
            f"Available values: {available}"
        )
    
    # Check if intensity is provided but not required
    if not requires_intensity and intensity is not None:
        print(f"[WARN] --intensity '{intensity}' provided but dataset '{theory}' "
              f"doesn't require it. It will be ignored.")
        return True
    
    # Check if intensity value is valid
    if intensity is not None:
        available = dataset_cfg["dataset"].get("available_intensities", [])
        if available and intensity not in available:
            raise ValueError(
                f"Invalid intensity '{intensity}' for dataset '{theory}'. "
                f"Available values: {available}"
            )
    
    return True


def validate_config(
    base_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    intensity: Optional[str] = None
) -> bool:
    """
    Validate complete configuration for consistency.
    
    Args:
        base_cfg: Base configuration
        model_cfg: Model configuration
        dataset_cfg: Dataset configuration
        intensity: Intensity level (optional)
        
    Returns:
        True if all validations pass
        
    Raises:
        ValueError: If any validation fails
    """
    # Validate intensity
    validate_intensity(dataset_cfg, intensity)
    
    # Check required fields in base config
    required_base_fields = ["paths", "evaluation"]
    for field in required_base_fields:
        if field not in base_cfg:
            raise ValueError(f"Missing required field in base config: {field}")
    
    # Check required fields in model config
    required_model_fields = ["model", "training", "loss"]
    for field in required_model_fields:
        if field not in model_cfg:
            raise ValueError(f"Missing required field in model config: {field}")
    
    # Check required fields in dataset config
    if "dataset" not in dataset_cfg:
        raise ValueError("Missing 'dataset' field in dataset config")
    
    required_dataset_fields = ["theory", "prefix"]
    for field in required_dataset_fields:
        if field not in dataset_cfg["dataset"]:
            raise ValueError(f"Missing required field in dataset config: {field}")
    
    # Validate model-specific requirements
    model_name = model_cfg["model"].get("name", "Unknown")
    
    # Check if TGAT-specific paths are available when using TGAT
    if "TGAT" in model_name.upper():
        if "tgat_graphs_dir" not in base_cfg["paths"]:
            print("[WARN] TGAT model selected but 'tgat_graphs_dir' not found in base config paths")
    
    print("[INFO] Configuration validation passed")
    return True


# ------------------------------------------------------------
# Path Construction
# ------------------------------------------------------------
def build_paths(
    base_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    intensity: Optional[str],
    model_name: str
) -> Dict[str, str]:
    """
    Build all necessary paths for the experiment.
    
    Args:
        base_cfg: Base configuration
        dataset_cfg: Dataset configuration
        intensity: Intensity level (or None for baseline)
        model_name: Model name (e.g., 'graphsage', 'graphsage_t', 'tgat')
        
    Returns:
        Dictionary with paths:
            - dataset_name: Full dataset name
            - graph_folder: Path to graph tensors
            - split_folder: Path to train/val/test splits
            - results_dir: Path to save results
            - logs_dir: Path to save logs
    """
    # Get script directory as default root
    # Get root directory - go up 2 levels from scripts/utils/ to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    root = base_cfg.get("paths", {}).get("root")

    # If no root specified in config, use project root
    if root:
        # If root is relative (like "."), make it absolute
        if not os.path.isabs(root):
            root = os.path.abspath(root)
    else:
        root = project_root
        
    # Extract dataset info
    theory = dataset_cfg["dataset"]["theory"]
    prefix = dataset_cfg["dataset"]["prefix"]
    
    # Build dataset name
    if theory.lower() == "baseline":
        dataset_name = prefix
    else:
        if intensity is None:
            raise ValueError(f"Intensity required for theory '{theory}'")
        dataset_name = f"{prefix}_{intensity}"
    
    # Determine which graphs directory to use
    model_lower = model_name.lower()
    if "tgat" in model_lower:
        graphs_dir_key = "tgat_graphs_dir"
    else:
        graphs_dir_key = "graphs_dir"
    
    graphs_dir = os.path.join(root, base_cfg["paths"][graphs_dir_key])
    splits_dir = os.path.join(root, base_cfg["paths"]["splits_dir"])
    results_root = os.path.join(root, base_cfg["paths"]["results_dir"])
    logs_root = os.path.join(root, base_cfg["paths"].get("logs_dir", "logs"))
    
    # Build specific paths
    graph_folder = os.path.join(graphs_dir, dataset_name)
    split_folder = os.path.join(splits_dir, dataset_name)
    results_dir = os.path.join(results_root, dataset_name, model_lower)
    logs_dir = os.path.join(logs_root, dataset_name, model_lower)
    
    # Create directories if they don't exist
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


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def set_seed(seed: int = 42, deterministic: bool = False):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, enable deterministic CUDA operations
                      (may reduce performance)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[INFO] Random seed set to {seed} (deterministic mode ON)")
    else:
        print(f"[INFO] Random seed set to {seed}")


# ------------------------------------------------------------
# Experiment Management
# ------------------------------------------------------------
def build_experiment_name(
    dataset_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    intensity: Optional[str] = None,
    timestamp: bool = False
) -> str:
    """
    Build a descriptive experiment name.
    
    Args:
        dataset_cfg: Dataset configuration
        model_cfg: Model configuration
        intensity: Intensity level (optional)
        timestamp: If True, append timestamp
        
    Returns:
        Experiment name string (e.g., 'RAT_medium_GraphSAGE' or 
        'baseline_TGAT_20241126_143022')
    """
    theory = dataset_cfg["dataset"]["theory"]
    model_name = model_cfg["model"]["name"].replace(" ", "").replace("-", "_")
    
    # Build base name
    if theory.lower() == "baseline":
        exp_name = f"{theory}_{model_name}"
    else:
        if intensity:
            exp_name = f"{theory}_{intensity}_{model_name}"
        else:
            exp_name = f"{theory}_{model_name}"
    
    # Add timestamp if requested
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{exp_name}_{ts}"
    
    return exp_name


def save_experiment_config(
    save_dir: str,
    base_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    intensity: Optional[str] = None,
    additional_info: Optional[Dict[str, Any]] = None,
    filename: str = "experiment_config.json"
) -> str:
    """
    Save complete experiment configuration to JSON.
    
    Args:
        save_dir: Directory to save config
        base_cfg: Base configuration
        model_cfg: Model configuration
        dataset_cfg: Dataset configuration
        intensity: Intensity level (optional)
        additional_info: Additional info to save (e.g., device, dataset stats)
        filename: Output filename
        
    Returns:
        Path to saved config file
    """
    experiment_config = {
        "timestamp": datetime.now().isoformat(),
        "base_config": base_cfg,
        "model_config": model_cfg,
        "dataset_config": dataset_cfg,
        "intensity": intensity,
    }
    
    if additional_info:
        experiment_config["additional_info"] = additional_info
    
    config_path = os.path.join(save_dir, filename)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, indent=2)
    
    return config_path


# ------------------------------------------------------------
# Helper: Print Configuration Summary
# ------------------------------------------------------------
def print_config_summary(
    base_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    intensity: Optional[str],
    paths: Dict[str, str]
):
    """
    Print a formatted summary of the experiment configuration.
    
    Args:
        base_cfg: Base configuration
        model_cfg: Model configuration
        dataset_cfg: Dataset configuration
        intensity: Intensity level
        paths: Dictionary of paths from build_paths()
    """
    print("=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)
    
    # Dataset info
    theory = dataset_cfg["dataset"]["theory"]
    print("\n[DATASET]")
    print(f"  Theory:       {theory}")
    print(f"  Intensity:    {intensity if intensity else 'N/A (baseline)'}")
    print(f"  Full name:    {paths['dataset_name']}")
    
    # Model info
    model_name = model_cfg["model"]["name"]
    print(f"\n[MODEL] {model_name}")
    print(f"  Hidden dim:   {model_cfg['model'].get('hidden_dim', 'N/A')}")
    print(f"  Num layers:   {model_cfg['model'].get('num_layers', 'N/A')}")
    print(f"  Dropout:      {model_cfg['model'].get('dropout', 'N/A')}")
    
    # Training info
    training = model_cfg["training"]
    print("\n[TRAINING]")
    print(f"  Device:       {training.get('device', 'cuda')}")
    print(f"  Batch size:   {training.get('batch_size', 'N/A')}")
    print(f"  Learning rate: {training.get('lr', 'N/A')}")
    print(f"  Epochs:       {training.get('epochs', 'N/A')}")
    print(f"  Early stop:   {training.get('early_stopping_patience', 'N/A')} epochs")
    
    # Paths
    print("\n[PATHS]")
    print(f"  Graphs:       {paths['graph_folder']}")
    print(f"  Splits:       {paths['split_folder']}")
    print(f"  Results:      {paths['results_dir']}")
    print(f"  Logs:         {paths['logs_dir']}")
    
    # Experiment settings
    seed = base_cfg.get("experiment", {}).get("seed", 42)
    print("\n[EXPERIMENT]")
    print(f"  Seed:         {seed}")
    print(f"  Exp name:     {base_cfg.get('experiment', {}).get('name', 'default')}")
    
    print("\n" + "=" * 80)


# ------------------------------------------------------------
# Complete Setup Function
# ------------------------------------------------------------
def setup_experiment(
    model_config_path: str,
    dataset_config_path: str,
    intensity: Optional[str] = None,
    base_config_path: str = "configs/base.yaml",
    verbose: bool = True,
    enable_logging: bool = True
) -> Dict[str, Any]:
    """
    Complete experiment setup: load configs, validate, build paths, set seed, setup logging.
    
    This is a convenience function that combines all setup steps.
    
    Args:
        model_config_path: Path to model config
        dataset_config_path: Path to dataset config
        intensity: Intensity level (optional)
        base_config_path: Path to base config
        verbose: If True, print configuration summary
        enable_logging: If True, setup logging to file
        
    Returns:
        Dictionary containing:
            - 'base_cfg': Base configuration
            - 'model_cfg': Model configuration
            - 'dataset_cfg': Dataset configuration
            - 'paths': Dictionary of paths
            - 'device': PyTorch device
            - 'experiment_name': Generated experiment name
            - 'logger': Logger object (if enable_logging=True)
            - 'log_path': Path to log file (if enable_logging=True)
    """
    # Load configs
    configs = load_and_merge_configs(
        model_config_path=model_config_path,
        dataset_config_path=dataset_config_path,
        base_config_path=base_config_path
    )
    
    base_cfg = configs["base"]
    model_cfg = configs["model"]
    dataset_cfg = configs["dataset"]
    
    # Validate
    validate_config(base_cfg, model_cfg, dataset_cfg, intensity)
    
    # Set seed
    seed = base_cfg.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    
    # Build paths
    model_name = model_cfg["model"]["name"].lower().replace(" ", "_").replace("-", "_")
    paths = build_paths(base_cfg, dataset_cfg, intensity, model_name)
    
    # Setup device
    device_str = model_cfg["training"].get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    # Generate experiment name
    experiment_name = build_experiment_name(dataset_cfg, model_cfg, intensity)
    
    # Setup logging if enabled
    logger = None
    log_path = None
    if enable_logging:
        log_path = setup_logging(
            log_dir=paths["logs_dir"],
            experiment_name=experiment_name,
            timestamp=True
        )
        logger = Logger(log_path)
        print(f"[INFO] Logging to: {log_path}")
    
    # Print summary if verbose
    if verbose:
        print_config_summary(base_cfg, model_cfg, dataset_cfg, intensity, paths)
        print(f"\n[DEVICE] {device}\n")
    
    result = {
        "base_cfg": base_cfg,
        "model_cfg": model_cfg,
        "dataset_cfg": dataset_cfg,
        "paths": paths,
        "device": device,
        "experiment_name": experiment_name,
        "intensity": intensity,
    }
    
    if enable_logging:
        result["logger"] = logger
        result["log_path"] = log_path
    
    return result


# ------------------------------------------------------------
# Main (for testing)
# ------------------------------------------------------------
if __name__ == "__main__":
    """Test the config utilities"""
    
    print("Testing config_utils.py\n")
    
    # Test 1: Load individual configs
    print("Test 1: Loading individual configs...")
    try:
        base = load_yaml("configs/base.yaml")
        print(f"   Loaded base config with keys: {list(base.keys())}")
    except Exception as e:
        print(f"   Error loading base config: {e}")
    
    # Test 2: Load and merge configs
    print("\nTest 2: Loading and merging configs...")
    try:
        configs = load_and_merge_configs(
            model_config_path="configs/models/graphsage.yaml",
            dataset_config_path="configs/datasets/rat.yaml"
        )
        print(f"   Merged configs with keys: {list(configs.keys())}")
    except Exception as e:
        print(f"   Error merging configs: {e}")
    
    # Test 3: Validate intensity
    print("\nTest 3: Validating intensity...")
    try:
        dataset_cfg = load_yaml("configs/datasets/rat.yaml")
        validate_intensity(dataset_cfg, "medium")
        print("   Intensity validation passed for 'medium'")
    except Exception as e:
        print(f"   Intensity validation failed: {e}")
    
    # Test 4: Complete setup
    print("\nTest 4: Complete experiment setup...")
    try:
        setup = setup_experiment(
            model_config_path="configs/models/graphsage.yaml",
            dataset_config_path="configs/datasets/rat.yaml",
            intensity="medium",
            verbose=True
        )
        print("   Complete setup successful")
    except Exception as e:
        print(f"   Setup failed: {e}")
    
    print("\n" + "=" * 80)
    print("All tests completed!")