"""
test_evaluate.py - train a tiny model on fake images, then check evaluate.py writes every output.

Run with:  pytest
Uses the small from-scratch model on CPU, so it takes seconds and downloads nothing.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Reuse the fake data/processed folder and the small training config from test_training.py.
from test_training import _write_config, processed, train  # noqa: F401  (processed is a pytest fixture)

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("evaluate", REPO_ROOT / "scripts" / "evaluate.py")
evaluate = importlib.util.module_from_spec(_spec)
sys.modules["evaluate"] = evaluate
_spec.loader.exec_module(evaluate)


@pytest.fixture
def trained_run(processed, tmp_path):  # noqa: F811
    """A finished (tiny) training run folder."""
    train.main(["--config", str(_write_config(tmp_path)), "--device", "cpu",
                "--processed", str(processed), "--runs-dir", str(tmp_path / "runs")])
    return next((tmp_path / "runs").iterdir())


def test_evaluate_writes_all_outputs(trained_run, processed):  # noqa: F811
    evaluate.main(["--run", str(trained_run), "--split", "val", "--device", "cpu",
                   "--processed", str(processed), "--workers", "0"])
    out = trained_run / "eval" / "val"
    for name in ["metrics.json", "confusion_matrix.png", "confusion_matrix.csv", "confusion_matrix_percent.csv",
                 "per_class.csv", "predictions.csv"]:
        assert (out / name).exists(), name
    for name in ["reward_convergence.png", "training_curves.png"]:
        assert (trained_run / "eval" / name).exists(), name

    results = json.loads((out / "metrics.json").read_text())
    assert results["images"] == 20                          # 4 val images x 5 classes
    plain = results["no threshold"]
    assert 0.0 <= plain["accuracy"] <= 1.0
    assert 0.0 <= plain["reward_percent_of_max"] <= 100.0
    assert sum(map(sum, plain["confusion_matrix"])) == 20  # every image counted once
    # 4 images of each class: 4 earn at most 0.5 (unknown), the other 16 at most 1.0
    # best possible = (16*1.0 + 4*0.5) / 20 = 0.9
    assert plain["max_possible_reward"] == pytest.approx(18 / 20)
    assert len(pd.read_csv(out / "predictions.csv")) == 20


def test_threshold_one_turns_every_answer_into_unknown(trained_run, processed):  # noqa: F811
    # No answer is ever 100% sure, so with threshold 1.0 everything becomes unknown_symbol.
    results = evaluate.run_evaluation(trained_run, "val", "cpu", threshold=1.0, processed=processed, workers=0)
    matrix = results["with threshold 1.0"]["confusion_matrix"]
    assert [row[1] for row in matrix] == [4, 4, 4, 4, 4]          # the whole unknown_symbol column


def test_empty_split_gives_a_helpful_message(trained_run, processed):  # noqa: F811
    with pytest.raises(SystemExit, match="--split val"):
        evaluate.run_evaluation(trained_run, "test", "cpu", processed=processed, workers=0)
