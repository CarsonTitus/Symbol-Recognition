"""
config.py - load and check the project's YAML settings files.

What this does:
    load_config("configs/stage_a.yaml") reads the stage file, fills in every
    setting it does not mention from the file named in its "base:" line
    (default.yaml), checks the important values, and returns a plain dict.

Why:
    One settings file per experiment means every run can save exactly what it
    used (reproducibility, CLAUDE.md §0.6), and the reward matrix lives in the
    config as CLAUDE.md §2 requires, not hard-coded in Python.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _merge(base: dict, override: dict) -> dict:
    """Return a copy of `base` with values from `override` laid on top.

    Nested sections are merged key by key, so a stage file can change just
    train.finetune_lr without repeating the whole train section.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict:
    """Read a YAML config (following its "base:" file, if any) and validate it."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # If this file says "base: default.yaml", load that first (relative to this file's folder).
    base_name = cfg.pop("base", None)
    if base_name is not None:
        base_cfg = load_config(path.parent / base_name)
        cfg = _merge(base_cfg, cfg)

    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    """Stop early with a clear message if a key setting is missing or wrong."""
    classes = cfg.get("classes")
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("config 'classes' must be a non-empty list of unique names")

    n = len(classes)
    matrix = cfg.get("reward_matrix")
    if matrix is None or len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError(f"config 'reward_matrix' must be {n}x{n} (one row and column per class)")
    for row in matrix:
        for value in row:
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("every reward_matrix value must be between 0 and 1")

    threshold = cfg.get("confidence_threshold")
    if threshold is None or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("config 'confidence_threshold' must be between 0 and 1")

    if "unknown_symbol" not in classes:
        raise ValueError("config 'classes' must include 'unknown_symbol' (the threshold fallback class)")

    # Every public dataset's class_map must point at one of OUR classes (catches typos like "triangel").
    for name, class_map in _all_class_maps(cfg.get("public_data", {})):
        for source_class, our_class in class_map.items():
            if our_class not in classes:
                raise ValueError(f"public_data {name}: '{source_class}' maps to '{our_class}', "
                                 f"which is not one of {classes}")


def _all_class_maps(public_data: dict):
    """Yield (dataset name, class_map) for every public dataset in the config, including Kaggle ones."""
    for name, settings in public_data.items():
        if name == "kaggle":
            for kaggle_name, kaggle_settings in settings.items():
                yield f"kaggle.{kaggle_name}", kaggle_settings.get("class_map") or {}
        else:
            yield name, settings.get("class_map") or {}
