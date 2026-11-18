"""
test_prepare_data.py - check prepare_data.py on a tiny fake copy of every data source.

Run with:  pytest
Checks the promises in CLAUDE.md §5c-5d:
  - no drawer appears in two splits
  - public data never lands in test (test = our own photos only)
  - novel shapes go only to test_novel_shapes/unknown_symbol
  - every saved image is 224x224 RGB, in the right class folder
  - shapes are padded to square, never stretched
"""
import csv
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from symbolcnn.config import load_config
from symbolcnn.imageprep import pad_to_square, to_model_size

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("prepare_data", REPO_ROOT / "scripts" / "prepare_data.py")
prepare_data = importlib.util.module_from_spec(_spec)
sys.modules["prepare_data"] = prepare_data   # @dataclass needs the module registered before it runs
_spec.loader.exec_module(prepare_data)


# ---------------------------------------------------------------- image helpers
def test_pad_to_square_does_not_stretch():
    image = Image.new("RGB", (100, 50), (255, 255, 255))
    image.paste((0, 0, 0), (40, 10, 60, 40))           # a 20x30 black block
    square = pad_to_square(image)
    assert square.size == (100, 100)
    pixels = np.asarray(square)
    assert (pixels[:25] == 255).all()                   # white padding on top...
    assert (pixels[75:] == 255).all()                   # ...and bottom, in the border color
    black_rows = np.where((pixels == 0).all(axis=2).any(axis=1))[0]
    assert len(black_rows) == 30                        # block is still 30 tall, not stretched


def test_to_model_size_makes_224_rgb_from_grayscale():
    gray = Image.fromarray(np.full((28, 28), 200, dtype=np.uint8))   # "L" = 1-channel grayscale
    out = to_model_size(gray, 224)
    assert out.size == (224, 224) and out.mode == "RGB"


# ---------------------------------------------------------------- drawer splitting
def test_drawer_split_never_shares_a_drawer_and_fills_every_split():
    sizes = {f"D{i:02d}": 10 + i for i in range(20)}
    shares = {"train": 0.70, "val": 0.15, "test": 0.15}
    assignment = prepare_data.assign_groups_to_splits(sizes, shares, seed=0)
    assert set(assignment) == set(sizes)                    # every drawer placed exactly once
    per_split = {s: sum(sizes[d] for d, x in assignment.items() if x == s) for s in shares}
    total = sum(sizes.values())
    for split, share in shares.items():
        assert per_split[split] > 0
        assert abs(per_split[split] / total - share) < 0.08  # close to the 70/15/15 target


def test_drawer_split_with_only_three_drawers_still_fills_test():
    assignment = prepare_data.assign_groups_to_splits({"A": 50, "B": 50, "C": 50},
                                                      {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
    assert sorted(assignment.values()) == ["test", "train", "val"]


def test_drawer_split_is_repeatable():
    sizes = {f"D{i}": 5 for i in range(10)}
    shares = {"train": 0.7, "val": 0.15, "test": 0.15}
    assert prepare_data.assign_groups_to_splits(sizes, shares, 1) == prepare_data.assign_groups_to_splits(sizes, shares, 1)


# ---------------------------------------------------------------- end to end on fake data
def _png_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (50, 50), color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def fake_data(tmp_path):
    raw_public, raw_own = tmp_path / "raw_public", tmp_path / "raw_own"

    # Hand-Drawn Shapes: 6 drawers x (triangle, rectangle) x 2 images
    for d in range(6):
        for shape in ["triangle", "rectangle"]:
            folder = raw_public / "hds" / "repo" / "data" / f"user.u{d}" / "images" / shape
            folder.mkdir(parents=True)
            for n in range(2):
                Image.new("RGBA", (70, 70), (255, 255, 255, 255)).save(folder / f"{shape}.{n}.png")

    # Quick, Draw!: 10 triangles and 10 circles (white on black)
    (raw_public / "quickdraw").mkdir(parents=True)
    for category in ["triangle", "circle"]:
        np.save(raw_public / "quickdraw" / f"{category}.npy", np.zeros((10, 28, 28), dtype=np.uint8))

    # Hugging Face: label index 0 (= "1", text only) is used; index 2 (= "3", triangle) is not mapped
    (raw_public / "hf_geometric_shapes" / "data").mkdir(parents=True)
    pd.DataFrame({"image": [{"bytes": _png_bytes((9, 9, 9)), "path": None}] * 4,
                  "label": [0, 0, 2, 2]}).to_parquet(raw_public / "hf_geometric_shapes" / "data" / "train.parquet")

    # Our photos: 6 drawers x 3 classes x 2 photos, plus 2 novel-shape photos
    rows = []
    for d in range(6):
        for label in ["no_symbol", "unknown_symbol", "triangle"]:
            (raw_own / label).mkdir(parents=True, exist_ok=True)
            for n in range(2):
                name = f"D{d}_{label}_{n}.jpg"
                Image.new("RGB", (400, 300), (200, 200, 200)).save(raw_own / label / name)
                rows.append({"filename": name, "class": label, "drawer_id": f"D{d}"})
    (raw_own / "novel_shapes").mkdir()
    for n in range(2):
        Image.new("RGB", (300, 400), (180, 180, 180)).save(raw_own / "novel_shapes" / f"heart_{n}.jpg")
        rows.append({"filename": f"heart_{n}.jpg", "class": "novel_shape", "drawer_id": "D0"})
    with (raw_own / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=prepare_data.METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return raw_public, raw_own, tmp_path / "processed"


def test_prepare_end_to_end(fake_data):
    raw_public, raw_own, processed = fake_data
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    manifest, problems = prepare_data.prepare(cfg, raw_public, raw_own, processed, workers=1)
    assert problems == []

    # Every expected image is there: 24 HDS + 20 Quick Draw + 2 HF + 36 own + 2 novel
    assert len(manifest) == 84
    assert set(manifest[manifest.source == "hf_geometric_shapes"].label) == {"no_symbol"}

    # No drawer in two splits (novel shapes are a separate test set, so they're left out of this check)
    drawers = manifest[(manifest.drawer_id != "") & (manifest.split != "test_novel_shapes")]
    assert (drawers.groupby("drawer_id").split.nunique() == 1).all()

    # Test = our own photos only; novel shapes only in their own folder
    assert set(manifest[manifest.split == "test"].source) == {"own"}
    novel = manifest[manifest.split == "test_novel_shapes"]
    assert len(novel) == 2 and set(novel.label) == {"unknown_symbol"}
    assert set(manifest[manifest.source != "own"].split) <= {"train", "val"}

    # Files exist in <split>/<label>/ and are 224x224 RGB
    for row in manifest.itertuples():
        path = processed / row.file
        assert path.parts[-3:-1] == (row.split, row.label)
        with Image.open(path) as image:
            assert image.size == (224, 224) and image.mode == "RGB"

    # Quick, Draw! was inverted: an all-black drawing becomes all white
    quickdraw_file = processed / manifest[manifest.source == "quickdraw"].iloc[0].file
    assert np.asarray(Image.open(quickdraw_file)).min() == 255


def test_drawers_separated_in_own_photos(fake_data):
    raw_public, raw_own, processed = fake_data
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    manifest, _ = prepare_data.prepare(cfg, raw_public, raw_own, processed, workers=1)
    own = manifest[(manifest.source == "own") & (manifest.split != "test_novel_shapes")]
    assert (own.groupby("drawer_id").split.nunique() == 1).all()
    assert set(own.split) == {"train", "val", "test"}


def test_bad_metadata_is_reported(fake_data):
    raw_public, raw_own, processed = fake_data
    with (raw_own / "metadata.csv").open("a", newline="") as f:
        f.write("missing.jpg,triangle,D1,,,,,,\n")       # file doesn't exist
        f.write("x.jpg,triangel,D1,,,,,,\n")             # typo in class
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    _, problems = prepare_data.prepare(cfg, raw_public, raw_own, processed, workers=1)
    assert any("file not found" in p for p in problems)
    assert any("triangel" in p for p in problems)


def test_refuses_to_delete_a_folder_it_did_not_make(tmp_path):
    (tmp_path / "processed").mkdir()
    (tmp_path / "processed" / "important.txt").write_text("keep me")
    with pytest.raises(RuntimeError):
        prepare_data.clear_output_folder(tmp_path / "processed")
    assert (tmp_path / "processed" / "important.txt").exists()
