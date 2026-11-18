"""
test_download_public.py - check the download script's tricky parts without using the internet.

Run with:  pytest
"""
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest

from symbolcnn.config import load_config, validate_config

REPO_ROOT = Path(__file__).resolve().parents[1]

# scripts/ is not a package, so load download_public.py directly from its file.
_spec = importlib.util.spec_from_file_location("download_public", REPO_ROOT / "scripts" / "download_public.py")
download_public = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(download_public)


def test_npy_header_is_read_correctly():
    # Build a real .npy file in memory: 5 fake "drawings" of 784 pixels each.
    drawings = np.arange(5 * 784, dtype=np.uint8).reshape(5, 784)
    buffer = io.BytesIO()
    np.save(buffer, drawings)
    file_bytes = buffer.getvalue()

    data_start, shape = download_public.parse_npy_header(file_bytes[:1024])
    assert shape == (5, 784)
    # The pixel data must start exactly where the header says.
    recovered = np.frombuffer(file_bytes[data_start:], dtype=np.uint8).reshape(5, 784)
    assert np.array_equal(recovered, drawings)


def test_non_npy_file_is_rejected():
    with pytest.raises(ValueError):
        download_public.parse_npy_header(b"<html>not found</html>")


def test_quickdraw_filenames_have_no_spaces():
    assert download_public.quickdraw_filename("smiley face") == "smiley_face.npy"


def test_quickdraw_excludes_triangle_like_categories():
    # CLAUDE.md §5a: these would teach the model that triangles are "unknown".
    class_map = load_config(REPO_ROOT / "configs" / "default.yaml")["public_data"]["quickdraw"]["class_map"]
    for triangle_like in ["mountain", "tent", "pizza"]:
        assert triangle_like not in class_map
    assert class_map["triangle"] == "triangle"


def test_class_map_typo_is_caught():
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    cfg["public_data"]["hds"]["class_map"]["triangle"] = "triangel"
    with pytest.raises(ValueError):
        validate_config(cfg)
