"""
test_config.py - check that the settings files load and that bad settings are caught.

Run with:  pytest
"""
from pathlib import Path

import pytest

from symbolcnn.config import load_config, validate_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", ["default.yaml", "stage_a.yaml", "stage_b.yaml"])
def test_every_config_loads(name):
    cfg = load_config(CONFIGS / name)
    assert cfg["classes"] == ["no_symbol", "unknown_symbol", "triangle", "quadrilateral", "ellipse"]


def test_stage_file_overrides_only_what_it_lists():
    cfg = load_config(CONFIGS / "stage_b.yaml")
    assert cfg["train"]["finetune_lr"] == pytest.approx(3e-5)   # changed by stage_b.yaml
    assert cfg["train"]["batch_size"] == 64                     # still inherited from default.yaml
    assert cfg["reward_matrix"][2] == [0.0, 0.5, 1.0, 0.0, 0.0]           # still inherited from default.yaml


def test_bad_reward_matrix_is_rejected():
    cfg = load_config(CONFIGS / "default.yaml")
    cfg["reward_matrix"] = [[1.0, 0.0], [0.0, 1.0]]  # 2x2 instead of 5x5
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_bad_threshold_is_rejected():
    cfg = load_config(CONFIGS / "default.yaml")
    cfg["confidence_threshold"] = 1.5
    with pytest.raises(ValueError):
        validate_config(cfg)
