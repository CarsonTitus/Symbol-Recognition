"""
test_predict.py - train a tiny model on fake images, then check predict.py's answers and output line.

Run with:  pytest
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from test_training import _write_config, processed, train  # noqa: F401  (processed is a pytest fixture)

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("predict", REPO_ROOT / "scripts" / "predict.py")
predict = importlib.util.module_from_spec(_spec)
sys.modules["predict"] = predict
_spec.loader.exec_module(predict)

CLASSES = ["no_symbol", "unknown_symbol", "triangle", "quadrilateral", "ellipse"]


@pytest.fixture
def runs(processed, tmp_path):  # noqa: F811
    """A runs/ folder holding one finished (tiny) training run."""
    train.main(["--config", str(_write_config(tmp_path)), "--device", "cpu",
                "--processed", str(processed), "--runs-dir", str(tmp_path / "runs")])
    return tmp_path / "runs"


@pytest.fixture
def photos(tmp_path):
    """A folder with two 'photos' of different sizes (not square, like a phone photo) and a text file."""
    folder = tmp_path / "photos"
    folder.mkdir()
    Image.new("RGB", (400, 300), "white").save(folder / "card.jpg")
    Image.new("L", (120, 200), 128).save(folder / "gray.png")      # grayscale works too
    (folder / "notes.txt").write_text("not an image")               # ignored: not an image file
    return folder


def test_every_photo_gets_an_answer_and_probabilities_add_up(runs, photos):
    results = predict.main(["--run", "latest", "--runs-dir", str(runs), "--image", str(photos),
                            "--device", "cpu"])
    assert [Path(r["image"]).name for r in results] == ["card.jpg", "gray.png"]
    for result in results:
        assert result["answer"] in CLASSES
        assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)


def test_threshold_one_always_answers_unknown(runs, photos):
    results = predict.main(["--run", "latest", "--runs-dir", str(runs), "--image", str(photos / "card.jpg"),
                            "--device", "cpu", "--threshold", "1.0"])
    assert results[0]["answer"] == "unknown_symbol"


def test_output_line_format():
    probs = np.array([0.012, 0.047, 0.941, 0.0, 0.0])
    answer, line = predict.describe("card_photo.jpg", probs, CLASSES, threshold=0.6)
    assert answer == "triangle"
    assert line.startswith("card_photo.jpg -> triangle (94.1%)   [no_symbol 1.2% | unknown_symbol 4.7% |")
    # Not confident enough -> the safety net answers unknown_symbol and says why.
    answer, line = predict.describe("x.jpg", np.array([0.1, 0.1, 0.5, 0.2, 0.1]), CLASSES, threshold=0.6)
    assert answer == "unknown_symbol" and "unsure: best guess triangle 50.0%" in line


def test_latest_picks_the_newest_finished_run(tmp_path):
    for name in ["20261118-100000_a", "20261120-090000_b", "20261121-090000_unfinished"]:
        (tmp_path / name).mkdir()
    for name in ["20261118-100000_a", "20261120-090000_b"]:
        (tmp_path / name / "best.pt").write_bytes(b"")
    assert predict.find_run("latest", tmp_path).name == "20261120-090000_b"
    with pytest.raises(SystemExit):
        predict.find_run("latest", tmp_path / "20261121-090000_unfinished")
