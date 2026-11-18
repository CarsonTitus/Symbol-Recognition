"""
download_public.py - download the public shape datasets into data/raw_public/.

What this does:
    Downloads four kinds of public data listed in configs/default.yaml (public_data):
      hds        Hand-Drawn Shapes: about 27,000 drawings by 37 people     (git clone, ~270 MB)
      quickdraw  Google Quick, Draw!: the first 3000 doodles of 12 shape
                 categories (only the part of each file we need, ~2 MB each)
      hf         Hugging Face "0-ma/geometric-shapes": synthetic images,
                 used for "no symbol" examples                             (~25 MB)
      kaggle     Two Kaggle datasets. Needs a Kaggle API token (README);
                 without one this part is skipped with a message, not an error.
    Each dataset goes in its own folder with a _download_info.json file
    recording when it was downloaded, from where, and its license.

Why:
    Public data gives the model thousands of examples of "shapes in general"
    before it sees our own photos (CLAUDE.md §5a). This script only DOWNLOADS;
    scripts/prepare_data.py later sorts the images into our three classes.

Usage (same in bash and PowerShell):
    python scripts/download_public.py                        # everything
    python scripts/download_public.py --sources hds quickdraw
    python scripts/download_public.py --force                # delete and download again

Safe to run again: datasets that are already downloaded are skipped.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np

from symbolcnn.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
INFO_FILE = "_download_info.json"           # marks a finished download
QUICKDRAW_SIDE = 28                          # Quick, Draw! bitmaps are 28x28 pixels
QUICKDRAW_BYTES_PER_DRAWING = QUICKDRAW_SIDE * QUICKDRAW_SIDE   # 784 bytes, one byte per pixel


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def already_downloaded(folder: Path) -> bool:
    """A dataset counts as downloaded only if its info file was written at the end."""
    return (folder / INFO_FILE).exists()


def write_info(folder: Path, info: dict) -> None:
    """Record what was downloaded, when, and its license, next to the data."""
    info["downloaded_at"] = datetime.now().isoformat(timespec="seconds")
    with (folder / INFO_FILE).open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def start_fresh(folder: Path, raw_public: Path) -> None:
    """Delete a half-finished or --force'd download. Refuses to delete anything outside data/raw_public."""
    if folder.exists():
        if raw_public.resolve() not in folder.resolve().parents:
            raise RuntimeError(f"refusing to delete {folder}: it is not inside {raw_public}")
        shutil.rmtree(folder)
    folder.mkdir(parents=True)


def http_get_bytes(url: str, first_byte: int | None = None, last_byte: int | None = None) -> bytes:
    """Download a URL, or just bytes first_byte..last_byte of it (an HTTP "Range" request)."""
    request = urllib.request.Request(url, headers={"User-Agent": "symbolcnn-downloader"})
    if first_byte is not None:
        request.add_header("Range", f"bytes={first_byte}-{last_byte}")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


# --------------------------------------------------------------------------
# Hand-Drawn Shapes (HDS)
# --------------------------------------------------------------------------
def download_hds(cfg: dict, raw_public: Path) -> str:
    folder = raw_public / "hds"
    start_fresh(folder, raw_public)
    url = cfg["public_data"]["hds"]["git_url"]
    print(f"  cloning {url} (about 270 MB)...")
    # --depth 1 = only the latest version of the files, not the full history
    subprocess.run(["git", "clone", "--depth", "1", url, str(folder / "repo")], check=True)

    # Layout: repo/data/user.<drawer id>/images/<shape>/<file>.png
    images = list((folder / "repo" / "data").glob("user.*/images/*/*.png"))
    drawers = sorted({p.parts[-4] for p in images})
    write_info(folder, {"source": url, "license": "CC BY 4.0",
                        "images": len(images), "drawers": len(drawers)})
    return f"{len(images)} images from {len(drawers)} drawers"


# --------------------------------------------------------------------------
# Quick, Draw!
# --------------------------------------------------------------------------
def parse_npy_header(start_of_file: bytes) -> tuple[int, tuple]:
    """Read a NumPy .npy file's header (its first few hundred bytes).

    Returns (where the pixel data starts, the array's shape).
    Format: 6-byte magic "\\x93NUMPY", 2 version bytes, a header length,
    then a text header like "{'descr': '|u1', 'fortran_order': False, 'shape': (123170, 784), }".
    """
    if start_of_file[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    major_version = start_of_file[6]
    if major_version == 1:
        header_length = int.from_bytes(start_of_file[8:10], "little")
        header_start = 10
    else:  # versions 2 and 3 use a 4-byte length
        header_length = int.from_bytes(start_of_file[8:12], "little")
        header_start = 12
    data_start = header_start + header_length
    header = ast.literal_eval(start_of_file[header_start:data_start].decode("latin1"))
    if header["descr"] != "|u1" or header["fortran_order"]:
        raise ValueError(f"unexpected .npy format: {header}")
    return data_start, header["shape"]


def quickdraw_filename(category: str) -> str:
    """'smiley face' -> 'smiley_face.npy' (no spaces in our file names)."""
    return category.replace(" ", "_") + ".npy"


def download_quickdraw(cfg: dict, raw_public: Path) -> str:
    settings = cfg["public_data"]["quickdraw"]
    cap = int(settings["per_category_cap"])
    folder = raw_public / "quickdraw"
    start_fresh(folder, raw_public)

    counts = {}
    for category in settings["class_map"]:
        url = f"{settings['base_url']}/{urllib.parse.quote(category)}.npy"
        # Each full file is ~100 MB, but drawings are stored one after another,
        # so we download only the header plus the first `cap` drawings.
        data_start, shape = parse_npy_header(http_get_bytes(url, 0, 1023))
        count = min(cap, shape[0])
        last_byte = data_start + count * QUICKDRAW_BYTES_PER_DRAWING - 1
        raw = http_get_bytes(url, 0, last_byte)
        if len(raw) != last_byte + 1:
            raise RuntimeError(f"{category}: expected {last_byte + 1} bytes, got {len(raw)}")

        drawings = np.frombuffer(raw[data_start:], dtype=np.uint8)
        drawings = drawings.reshape(count, QUICKDRAW_SIDE, QUICKDRAW_SIDE)   # still white-on-black
        np.save(folder / quickdraw_filename(category), drawings)
        counts[category] = count
        print(f"  {category:12s} {count} drawings (of {shape[0]:,} available)")

    write_info(folder, {"source": settings["base_url"], "license": "CC BY 4.0",
                        "per_category_cap": cap, "drawings": counts,
                        "note": "first N drawings of each category; white strokes on black"})
    return f"{sum(counts.values())} drawings in {len(counts)} categories"


# --------------------------------------------------------------------------
# Hugging Face geometric shapes
# --------------------------------------------------------------------------
def download_hf(cfg: dict, raw_public: Path) -> str:
    from huggingface_hub import snapshot_download

    repo_id = cfg["public_data"]["hf_geometric_shapes"]["repo_id"]
    folder = raw_public / "hf_geometric_shapes"
    start_fresh(folder, raw_public)
    # The images are stored in 3 Parquet files (a compact table format) under data/.
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=folder,
                      allow_patterns=["data/*.parquet", "README.md"])
    files = sorted(p.name for p in (folder / "data").glob("*.parquet"))
    write_info(folder, {"source": f"https://huggingface.co/datasets/{repo_id}",
                        "license": "MIT", "files": files})
    return f"{len(files)} parquet files"


# --------------------------------------------------------------------------
# Kaggle
# --------------------------------------------------------------------------
def kaggle_api_or_none():
    """Return a logged-in Kaggle API object, or None (with a message) if no token is set up."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        # Hide the library's long "how to log in" text; we print a short message instead.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            api.authenticate()
        return api
    except SystemExit:
        # The kaggle library calls exit() when no token exists; catch that so the rest still runs.
        print("  no Kaggle API token found - skipping Kaggle datasets.")
        print("  To add one: README section 'Kaggle API token', then run with --sources kaggle")
        return None


def download_kaggle(cfg: dict, raw_public: Path, force: bool) -> str:
    api = kaggle_api_or_none()
    if api is None:
        return "SKIPPED (no Kaggle token)"

    results = []
    for name, settings in cfg["public_data"]["kaggle"].items():
        folder = raw_public / "kaggle" / name
        if already_downloaded(folder) and not force:
            results.append(f"{name}: already present")
            continue
        start_fresh(folder, raw_public)
        print(f"  downloading {settings['slug']}...")
        api.dataset_download_files(settings["slug"], path=str(folder), unzip=True, quiet=False)
        files = sum(1 for p in folder.rglob("*") if p.is_file())
        write_info(folder, {"source": f"https://www.kaggle.com/datasets/{settings['slug']}",
                            "license": "see data/SOURCES.md", "files": files})
        results.append(f"{name}: {files} files")
    return "; ".join(results)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
DOWNLOADERS = {
    "hds": ("hds", download_hds),
    "quickdraw": ("quickdraw", download_quickdraw),
    "hf": ("hf_geometric_shapes", download_hf),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the public shape datasets into data/raw_public/.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--sources", nargs="+", default=["hds", "quickdraw", "hf", "kaggle"],
                        choices=["hds", "quickdraw", "hf", "kaggle"])
    parser.add_argument("--force", action="store_true", help="delete and re-download even if already present")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_public = REPO_ROOT / cfg["paths"]["raw_public"]
    raw_public.mkdir(parents=True, exist_ok=True)

    summary, failed = {}, False
    for source in args.sources:
        print(f"\n[{source}]")
        try:
            if source == "kaggle":
                summary[source] = download_kaggle(cfg, raw_public, args.force)
                continue
            folder_name, download = DOWNLOADERS[source]
            if already_downloaded(raw_public / folder_name) and not args.force:
                summary[source] = "already present (use --force to download again)"
                print(f"  {summary[source]}")
                continue
            summary[source] = download(cfg, raw_public)
        except Exception as error:   # keep going so one broken source doesn't block the others
            summary[source] = f"FAILED: {error}"
            failed = True
        print(f"  -> {summary[source]}")

    print("\nSummary:")
    for source, result in summary.items():
        print(f"  {source:10s} {result}")
    print(f"Data folder: {raw_public}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
