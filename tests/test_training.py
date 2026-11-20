"""
test_training.py - check datasets.py, models.py and train.py on a tiny fake dataset.

Run with:  pytest
Uses the small from-scratch model, so it runs in seconds on any CPU with no downloads.
"""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from symbolcnn.config import load_config
from symbolcnn.datasets import SymbolDataset, balanced_sampler, eval_transform, load_split, train_transform
from symbolcnn.models import (IMAGENET_MEAN, IMAGENET_STD, TinyCNN, count_parameters, create_model,
                              freeze_backbone, unfreeze_all)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["unknown_symbol", "triangle", "quadrilateral", "ellipse"]
_spec = importlib.util.spec_from_file_location("train", REPO_ROOT / "scripts" / "train.py")
train = importlib.util.module_from_spec(_spec)
sys.modules["train"] = train
_spec.loader.exec_module(train)


@pytest.fixture
def processed(tmp_path):
    """A fake data/processed: 8 train + 4 val images per class, plus manifest.csv."""
    rows, rng = [], np.random.default_rng(0)
    for split, count in [("train", 8), ("val", 4)]:
        for label in CLASSES:
            folder = tmp_path / split / label
            folder.mkdir(parents=True)
            for n in range(count):
                Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)).save(folder / f"{n}.png")
                rows.append({"split": split, "label": label, "file": f"{split}/{label}/{n}.png",
                             "source": "fake", "source_class": label, "drawer_id": "", "original": ""})
    pd.DataFrame(rows).to_csv(tmp_path / "manifest.csv", index=False)
    return tmp_path


# ---------------------------------------------------------------- datasets.py
def test_dataset_returns_normalized_tensors(processed):
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    rows = load_split(processed, "train")
    for transform in [train_transform(cfg, IMAGENET_MEAN, IMAGENET_STD), eval_transform(IMAGENET_MEAN, IMAGENET_STD)]:
        image, label = SymbolDataset(rows, processed, CLASSES, transform)[0]
        assert image.shape == (3, 224, 224) and image.dtype == torch.float32
        assert label in (0, 1, 2)


def test_data_fraction_keeps_every_class(processed):
    rows = load_split(processed, "train", fraction=0.25, seed=0)
    assert len(rows) == 8                                   # 25% of 32 (8 images x 4 classes)
    assert sorted(rows["label"].unique()) == sorted(CLASSES)


def test_balanced_sampler_evens_out_classes():
    labels = [0] * 10 + [1] * 1000 + [2] * 100              # very uneven, like our data
    picks = list(balanced_sampler(labels, 3, seed=0))
    share = np.bincount([labels[i] for i in picks], minlength=3) / len(picks)
    assert np.allclose(share, 1 / 3, atol=0.05)             # each class picked about a third of the time


# ---------------------------------------------------------------- models.py
def test_tiny_cnn_outputs_one_score_per_class():
    assert TinyCNN(4)(torch.randn(2, 3, 224, 224)).shape == (2, 4)


def test_freeze_leaves_only_the_classifier_trainable():
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    model = create_model(cfg, pretrained=False)             # MobileNetV3 without downloading weights
    freeze_backbone(model)
    assert count_parameters(model, trainable_only=True) == count_parameters(model.get_classifier())
    unfreeze_all(model)
    assert count_parameters(model, trainable_only=True) == count_parameters(model)


# ---------------------------------------------------------------- train.py
def _write_config(tmp_path, **train_settings):
    config = {"base": str(REPO_ROOT / "configs" / "default.yaml"), "run_name": "test",
              "model": {"name": "tiny_cnn"}, "data": {"num_workers": 0},
              "train": {"batch_size": 4, "frozen_epochs": 1, "finetune_epochs": 2,
                        "mixed_precision": False, **train_settings}}
    path = tmp_path / "test.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_training_run_writes_metrics_and_checkpoints(processed, tmp_path):
    config = _write_config(tmp_path)
    train.main(["--config", str(config), "--device", "cpu",
                "--processed", str(processed), "--runs-dir", str(tmp_path / "runs")])
    run_dir = next((tmp_path / "runs").iterdir())
    assert run_dir.name.endswith("_test_tinycnn")

    with (run_dir / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2                                    # tiny_cnn skips the frozen phase
    assert list(rows[0]) == train.METRIC_COLUMNS
    for row in rows:
        assert 0.0 <= float(row["val_acc"]) <= 1.0
        assert 0.0 <= float(row["val_reward_pct"]) <= 100.0

    for name in ["best.pt", "last.pt", "config.yaml", "summary.json", "what_the_model_sees.png"]:
        assert (run_dir / name).exists(), name
    assert any((run_dir / "tensorboard").iterdir())

    checkpoint = torch.load(run_dir / "best.pt", weights_only=False)
    assert checkpoint["classes"] == CLASSES and checkpoint["model_name"] == "tiny_cnn"
    TinyCNN(4).load_state_dict(checkpoint["model_state"])   # weights load back into a fresh model


def test_early_stopping(processed, tmp_path):
    config = _write_config(tmp_path, finetune_epochs=10, early_stopping_patience=1, finetune_lr=0.0)
    train.main(["--config", str(config), "--device", "cpu",
                "--processed", str(processed), "--runs-dir", str(tmp_path / "runs")])
    run_dir = next((tmp_path / "runs").iterdir())
    with (run_dir / "metrics.csv").open() as f:
        epochs = len(list(csv.DictReader(f)))
    assert epochs == 2       # learning rate 0 -> no improvement after epoch 1 -> stops after epoch 2


def test_overrides_are_saved_in_the_run_config(processed, tmp_path):
    config = _write_config(tmp_path, finetune_epochs=1)
    train.main(["--config", str(config), "--device", "cpu", "--data-fraction", "0.5", "--seed", "3",
                "--processed", str(processed), "--runs-dir", str(tmp_path / "runs")])
    run_dir = next((tmp_path / "runs").iterdir())
    assert run_dir.name.endswith("_test_tinycnn_frac50_seed3")
    saved = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert saved["data"]["fraction"] == 0.5 and saved["seed"] == 3
