"""
utils.py - small helpers every script needs.

What this does:
    set_seed()      fix all random number generators so a run can be repeated exactly
    pick_device()   choose the GPU or CPU ("auto" uses a GPU when one exists)
    make_run_dir()  create runs/<timestamp>_<name>/ and save the config there

Why:
    CLAUDE.md §0.6 requires fixed seeds and one folder per run holding its
    config and metrics. The same code must work on the Windows tower (2 GPUs),
    a laptop (probably no NVIDIA GPU) and the cloud (CPU only).
"""
from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and every GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # does nothing when there is no GPU
    except ImportError:
        pass  # lets the NumPy-only tests run without PyTorch installed


def pick_device(requested: str = "auto") -> str:
    """Return the device string to train on.

    "auto"    -> "cuda:0" if an NVIDIA GPU is available, else "mps" on Apple
                 laptops, else "cpu".
    "cuda:1"  -> the second GPU (the tower runs one experiment per GPU, CLAUDE.md §4).
    "cpu"     -> always the processor.
    CUDA = NVIDIA's platform that lets PyTorch use the GPU.
    """
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"  # Apple-silicon GPU
        return "cpu"

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"--device {requested} asked for a GPU, but PyTorch sees no CUDA GPU. "
                               "Check the PyTorch install (README) or use --device cpu.")
        index = int(requested.split(":")[1]) if ":" in requested else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"--device {requested}: only {torch.cuda.device_count()} GPU(s) found")
    return requested


def make_run_dir(runs_root: str | Path, name: str, cfg: dict | None = None) -> Path:
    """Create runs/<YYYYMMDD-HHMMSS>_<name>/ and save the config inside it as config.yaml.

    The timestamp uses no colons, because Windows does not allow ':' in folder names.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runs_root) / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    if cfg is not None:
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    return run_dir
