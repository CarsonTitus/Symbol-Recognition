"""
test_learning_curve.py - check the GPU sharing and run a tiny learning curve end to end on CPU.

Run with:  pytest
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

from test_training import _write_config, processed  # noqa: F401  (processed is a pytest fixture)

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("learning_curve", REPO_ROOT / "scripts" / "learning_curve.py")
learning_curve = importlib.util.module_from_spec(_spec)
sys.modules["learning_curve"] = learning_curve
_spec.loader.exec_module(learning_curve)


def test_biggest_runs_are_spread_across_gpus():
    queues = learning_curve.assign_to_devices([0.1, 0.25, 0.5, 1.0], ["cuda:0", "cuda:1"])
    assert queues == {"cuda:0": [1.0], "cuda:1": [0.5, 0.25, 0.1]}
    assert learning_curve.assign_to_devices([0.1, 0.5], ["cpu"]) == {"cpu": [0.5, 0.1]}


def test_fraction_tag():
    assert learning_curve.fraction_tag(0.1) == "frac10"
    assert learning_curve.fraction_tag(1.0) == "frac100"


def test_tiny_learning_curve_end_to_end(processed, tmp_path):  # noqa: F811
    config = _write_config(tmp_path)
    learning_curve.main(["--config", str(config), "--fractions", "0.5", "1.0", "--devices", "cpu", "cpu",
                         "--processed", str(processed), "--runs-dir", str(tmp_path / "runs"), "--workers", "0"])
    out = next((tmp_path / "runs").iterdir())
    table = pd.read_csv(out / "learning_curve_val.csv")
    assert table["fraction"].tolist() == [0.5, 1.0]
    assert table["train_images"].tolist() == [20, 40]        # 50% and 100% of 8 images x 5 classes
    assert table["accuracy"].between(0, 1).all()
    assert (out / "learning_curve_val.png").exists()
    assert (out / "logs" / "frac50.log").exists()

    # Re-scoring an existing folder works without training again.
    learning_curve.main(["--from", str(out), "--split", "val", "--devices", "cpu",
                         "--processed", str(processed), "--workers", "0"])
