# ========================================================================
# checkpoint_utils.py
#
# Shared helpers for:
#   1) Saving/loading training checkpoints (model, optimizer, RNG state,
#      epoch, early-stopping bookkeeping) so a run can resume after a
#      crash / preemption / outage.
#   2) Syncing the results + logs + checkpoint folders for an experiment
#      to a Google Drive remote via `rclone`, so partial results survive
#      even if the GPU instance dies before the run finishes.
#
# Usage (in a training script):
#
#   from scripts.utils.checkpoint_utils import (
#       save_checkpoint, load_checkpoint, sync_experiment_to_drive,
#   )
#
#   ckpt_path = os.path.join(results_dir, "checkpoint.pt")
#   start_epoch, best_val, best_epoch, patience = 1, -1e9, -1, 0
#
#   state = load_checkpoint(ckpt_path, model, optimizer, device)
#   if state is not None:
#       start_epoch = state["epoch"] + 1
#       best_val = state["best_val"]
#       best_epoch = state["best_epoch"]
#       patience = state["patience"]
#
#   for epoch in range(start_epoch, epochs + 1):
#       ... train / validate ...
#       save_checkpoint(ckpt_path, epoch=epoch, model=model,
#                        optimizer=optimizer, best_val=best_val,
#                        best_epoch=best_epoch, patience=patience)
#       sync_experiment_to_drive(base_cfg, project_root, results_dir, logs_dir)
# ========================================================================

import os
import random
import shutil
import subprocess
from typing import Any, Dict, Optional

import numpy as np
import torch


# ========================================================================
# CHECKPOINT SAVE / LOAD (model + optimizer + RNG + bookkeeping)
# ========================================================================

def save_checkpoint(
    checkpoint_path: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val: float,
    best_epoch: int,
    patience: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a full training checkpoint (atomically) so training can be
    resumed from exactly this point.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val": best_val,
        "best_epoch": best_epoch,
        "patience": patience,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }

    if extra:
        state["extra"] = extra

    # Write to a temp file then replace, so a crash mid-write can't
    # corrupt the last good checkpoint.
    tmp_path = checkpoint_path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    restore_rng: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    If a checkpoint exists at `checkpoint_path`, load model/optimizer
    weights (in place) and return the saved bookkeeping dict so the
    caller can resume the training loop. Returns None if no checkpoint
    is found (i.e. this is a fresh run).
    """
    if not os.path.exists(checkpoint_path):
        return None

    print(f"[RESUME] Found checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(state["model_state"])

    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])

    if restore_rng and "rng_state" in state:
        rng = state["rng_state"]
        try:
            torch.set_rng_state(rng["torch"].cpu().to(torch.uint8) if torch.is_tensor(rng["torch"]) else rng["torch"])
            if rng.get("torch_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        except Exception as e:
            print(f"[RESUME] WARNING: could not fully restore RNG state ({e})")

    print(
        f"[RESUME] Resuming from epoch {state['epoch']} "
        f"(best_val={state.get('best_val')}, best_epoch={state.get('best_epoch')}, "
        f"patience={state.get('patience')})"
    )
    return state


# ========================================================================
# GOOGLE DRIVE SYNC (via rclone)
# ========================================================================

def get_sync_config(base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read the optional `sync:` section from base.yaml. Defaults to
    disabled if not present, so this is a no-op unless configured.

    Expected base.yaml shape:

        sync:
          enabled: true
          rclone_remote: "gdrive"          # name from `rclone config`
          drive_root: "AML_project_results"  # folder inside the remote
          sync_logs: true
    """
    cfg = base_cfg.get("sync", {}) or {}
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "rclone_remote": cfg.get("rclone_remote", "gdrive"),
        "drive_root": cfg.get("drive_root", "AML_project_results"),
        "sync_logs": bool(cfg.get("sync_logs", True)),
        "sync_every_n_epochs": int(cfg.get("sync_every_n_epochs", 1)),
        "extra_args": cfg.get("extra_args", []),
        # Optional explicit path to the rclone executable, for cases where
        # rclone (e.g. installed via `conda install -c conda-forge rclone`)
        # isn't on the PATH of the shell that launches training.
        "rclone_path": cfg.get("rclone_path", None),
    }


def should_sync_this_epoch(base_cfg: Dict[str, Any], epoch: int) -> bool:
    """
    True if `epoch` is a multiple of `sync.sync_every_n_epochs` (always
    True if that's <= 1). Use this to gate per-epoch Drive syncs while
    still checkpointing locally every epoch for resume purposes.
    """
    sync_cfg = get_sync_config(base_cfg)
    if not sync_cfg["enabled"]:
        return False
    n = max(sync_cfg["sync_every_n_epochs"], 1)
    return epoch % n == 0


def _find_rclone(explicit_path: Optional[str] = None) -> Optional[str]:
    """
    Locate the rclone executable. Tries, in order:
      1. An explicit path from `sync.rclone_path` in base.yaml
      2. `rclone` on PATH
      3. Common locations next to the running Python interpreter
         (covers `conda install -c conda-forge rclone`, where rclone
         lands in the env's bin/Scripts/Library\\bin but that folder
         may not be on the PATH of the shell that launched training).
    """
    import sys

    if explicit_path:
        if os.path.isfile(explicit_path):
            return explicit_path
        print(f"[DRIVE SYNC] WARNING: configured rclone_path not found: {explicit_path}")

    on_path = shutil.which("rclone")
    if on_path:
        return on_path

    py_dir = os.path.dirname(sys.executable)
    candidates = [
        os.path.join(py_dir, "rclone.exe"),
        os.path.join(py_dir, "rclone"),
        os.path.join(py_dir, "Scripts", "rclone.exe"),
        os.path.join(py_dir, "Scripts", "rclone"),
        os.path.join(py_dir, "Library", "bin", "rclone.exe"),
        os.path.join(py_dir, "bin", "rclone"),
        os.path.join(py_dir, "..", "Scripts", "rclone.exe"),
        os.path.join(py_dir, "..", "Library", "bin", "rclone.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)

    return None


def _rclone_sync_dir(local_dir: str, remote_target: str, extra_args=None, quiet: bool = True, rclone_path: Optional[str] = None) -> bool:
    """
    Run `rclone sync <local_dir> <remote_target>`.
    Returns True on success, False otherwise (never raises - sync
    failures must not crash a training run).
    """
    if not os.path.isdir(local_dir):
        return False

    rclone_exe = _find_rclone(rclone_path)
    if rclone_exe is None:
        print(
            "[DRIVE SYNC] WARNING: 'rclone' not found on PATH or near the Python "
            "interpreter; skipping sync. Set sync.rclone_path in base.yaml to the "
            "full path of rclone.exe (e.g. find it with `where rclone` / `which rclone` "
            "in the env where you ran `conda install -c conda-forge rclone`)."
        )
        return False

    cmd = [rclone_exe, "sync", local_dir, remote_target, "--create-empty-src-dirs"]
    if quiet:
        cmd.append("-q")
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"[DRIVE SYNC] WARNING: rclone exited {result.returncode} for {local_dir} -> {remote_target}")
            if result.stderr:
                print(f"[DRIVE SYNC] stderr: {result.stderr.strip()[:500]}")
            return False
        return True
    except Exception as e:
        print(f"[DRIVE SYNC] WARNING: rclone sync failed for {local_dir} -> {remote_target}: {e}")
        return False


def sync_experiment_to_drive(
    base_cfg: Dict[str, Any],
    project_root: str,
    results_dir: str,
    logs_dir: Optional[str] = None,
    label: str = "",
) -> None:
    """
    Push `results_dir` (and optionally `logs_dir`) to Google Drive via
    rclone, mirroring the path relative to the project root under
    `<rclone_remote>:<drive_root>/...`.

    Safe to call every epoch: failures are logged and swallowed so they
    never interrupt training. No-op if `sync.enabled` is not true in
    base.yaml.
    """
    sync_cfg = get_sync_config(base_cfg)
    if not sync_cfg["enabled"]:
        return

    remote = sync_cfg["rclone_remote"]
    drive_root = sync_cfg["drive_root"]
    extra_args = sync_cfg["extra_args"]
    rclone_path = sync_cfg["rclone_path"]

    targets = [results_dir]
    if logs_dir and sync_cfg["sync_logs"]:
        targets.append(logs_dir)

    for local_dir in targets:
        rel_path = os.path.relpath(local_dir, project_root).replace(os.sep, "/")
        remote_target = f"{remote}:{drive_root}/{rel_path}"

        ok = _rclone_sync_dir(local_dir, remote_target, extra_args=extra_args, rclone_path=rclone_path)
        if ok:
            tag = f" [{label}]" if label else ""
            print(f"[DRIVE SYNC]{tag} {local_dir} -> {remote_target}")
