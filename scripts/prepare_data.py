"""
prepare_data.py - build the training folders from the raw downloads and our photos.

What this does:
    1. Collects images from every source:
         data/raw_public/hds, quickdraw, hf_geometric_shapes, kaggle/*   (public)
         data/raw_own/<class>/ + data/raw_own/metadata.csv               (our photos)
       and gives each one of OUR labels using the class_map settings in
       configs/default.yaml. Source classes with no mapping are skipped.
    2. Splits them into train / val / test (CLAUDE.md §5c):
         - our photos are split BY DRAWER (70/15/15): all photos by one
           person go into one split, so the test score can't be inflated by
           the model memorising someone's handwriting;
         - Hand-Drawn Shapes is also split by drawer (train/val only);
         - other public data is split randomly (train/val only);
         - public data NEVER goes in test: test = our own photos only.
       Photos whose metadata class is "novel_shape" go to test_novel_shapes/.
    3. Converts every image to 224x224 RGB (pad to square, never stretch;
       Quick, Draw! inverted to dark lines on a light background) and saves
       it as data/processed/<split>/<label>/<source>_<number>.png.
    4. Writes data/processed/manifest.csv: one row per image with its split,
       label, source, original location and drawer. Training reads this file.

Why:
    The model needs one consistent, honest set of folders (CLAUDE.md §5c-5d).
    The script always rebuilds data/processed from scratch, and with the same
    seed it always makes the same split, so results are reproducible.

Usage (same in bash and PowerShell):
    python scripts/prepare_data.py
    python scripts/prepare_data.py --workers 40           # tower: use more CPU cores
    python scripts/prepare_data.py --limit-per-class 200  # quick small trial run
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from symbolcnn.config import load_config
from symbolcnn.imageprep import load_photo, to_model_size

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
NOVEL_CLASS = "novel_shape"            # metadata class for never-trained-on shapes (CLAUDE.md §5c)
NOVEL_FOLDER = "novel_shapes"          # data/raw_own/novel_shapes/
MANIFEST = "manifest.csv"
MARKER = ".made_by_prepare_data"       # proves data/processed was made by this script
METADATA_COLUMNS = ["filename", "class", "drawer_id", "pen", "surface",
                    "distance_m", "angle_deg", "lighting", "notes"]


@dataclass
class Item:
    """One image waiting to be processed."""
    source: str           # hds, quickdraw, hf_geometric_shapes, kaggle.<name>, own
    source_class: str     # the class name in the original dataset
    label: str            # OUR class (config "classes"), e.g. triangle or ellipse
    drawer_id: str        # who drew it ("" if unknown)
    original: str         # where it came from, for the manifest
    kind: str             # how to load it: "file", "array" or "bytes"
    payload: object       # a Path, a NumPy array, or the image file's bytes
    invert: bool = False  # True for Quick, Draw! (white strokes on black)
    split: str = ""       # filled in by the splitting step


# --------------------------------------------------------------------------
# Step 1: collect images from each source
# --------------------------------------------------------------------------
def collect_hds(cfg: dict, raw_public: Path) -> list[Item]:
    class_map = cfg["public_data"]["hds"]["class_map"]
    items = []
    # Layout: hds/repo/data/user.<drawer>/images/<shape>/<file>.png
    for path in sorted((raw_public / "hds" / "repo" / "data").glob("user.*/images/*/*.png")):
        shape = path.parent.name
        if shape in class_map:
            drawer = path.parts[-4].removeprefix("user.")
            items.append(Item("hds", shape, class_map[shape], f"hds:{drawer}",
                              str(path.relative_to(raw_public)), "file", path))
    return items


def collect_quickdraw(cfg: dict, raw_public: Path) -> list[Item]:
    class_map = cfg["public_data"]["quickdraw"]["class_map"]
    items = []
    for category, label in class_map.items():
        file = raw_public / "quickdraw" / (category.replace(" ", "_") + ".npy")  # same rule as download_public.py
        if not file.exists():
            continue
        for index, drawing in enumerate(np.load(file)):
            items.append(Item("quickdraw", category, label, "", f"{file.name}#{index}",
                              "array", drawing, invert=True))
    return items


def collect_hf(cfg: dict, raw_public: Path) -> list[Item]:
    class_map = cfg["public_data"]["hf_geometric_shapes"]["class_map"]
    items = []
    for parquet in sorted((raw_public / "hf_geometric_shapes" / "data").glob("*.parquet")):
        table = pd.read_parquet(parquet)
        for row_number, (image, label_index) in enumerate(zip(table["image"], table["label"])):
            # The dataset card names labels "1".."6" for indexes 0..5 ("1" = text only, no shape).
            name = str(int(label_index) + 1)
            if name in class_map:
                items.append(Item("hf_geometric_shapes", name, class_map[name], "",
                                  f"{parquet.name}#{row_number}", "bytes", image["bytes"]))
    return items


def collect_kaggle(cfg: dict, raw_public: Path) -> list[Item]:
    items = []
    for name, settings in cfg["public_data"].get("kaggle", {}).items():
        class_map = {k.lower(): v for k, v in (settings.get("class_map") or {}).items()}
        folder = raw_public / "kaggle" / name
        if not folder.exists() or not class_map:
            continue
        # Optional "every_nth": keep only every Nth image of each class.
        # Four Shapes is cut from ONE video per shape, so neighbouring frames are
        # near-identical copies. Keeping every Nth frame keeps the variety (each
        # rotation/position) without thousands of duplicates, which would also
        # leak into validation and make the val score look better than it is.
        every_nth = int(settings.get("every_nth") or 1)
        seen_per_class: dict[str, int] = {}
        # Kaggle image datasets are usually folders named after the class.
        # Files are numbered 0.png, 1.png...; sort by number so "every Nth" follows video order.
        def frame_order(p: Path):
            return (str(p.parent), int(p.stem) if p.stem.isdigit() else 0, p.name)
        for path in sorted(folder.rglob("*"), key=frame_order):
            shape = path.parent.name.lower()
            if path.suffix.lower() in IMAGE_EXTENSIONS and shape in class_map:
                count = seen_per_class.get(shape, 0)
                seen_per_class[shape] = count + 1
                if count % every_nth != 0:
                    continue  # skip near-duplicate frame
                items.append(Item(f"kaggle.{name}", shape, class_map[shape], "",
                                  str(path.relative_to(raw_public)), "file", path))
    return items


def collect_own(cfg: dict, raw_own: Path) -> tuple[list[Item], list[Item], list[str]]:
    """Read data/raw_own/metadata.csv. Returns (normal photos, novel-shape photos, problems)."""
    metadata = raw_own / "metadata.csv"
    if not metadata.exists():
        return [], [], []

    classes = cfg["classes"]
    items, novel, problems = [], [], []
    with metadata.open(newline="", encoding="utf-8-sig") as f:   # utf-8-sig: tolerate Excel's CSVs
        reader = csv.DictReader(f)
        missing = [c for c in ("filename", "class", "drawer_id") if c not in (reader.fieldnames or [])]
        if missing:
            return [], [], [f"metadata.csv is missing column(s): {missing}"]
        for line, row in enumerate(reader, start=2):              # line 1 is the header
            filename, label, drawer = (row["filename"] or "").strip(), (row["class"] or "").strip(), \
                                      (row["drawer_id"] or "").strip()
            if label not in classes and label != NOVEL_CLASS:
                problems.append(f"metadata.csv line {line}: class '{label}' is not one of "
                                f"{classes + [NOVEL_CLASS]}")
                continue
            if not drawer:
                problems.append(f"metadata.csv line {line}: drawer_id is empty")
                continue
            folder = NOVEL_FOLDER if label == NOVEL_CLASS else label
            path = raw_own / folder / filename
            if not path.exists():
                problems.append(f"metadata.csv line {line}: file not found: {path.relative_to(raw_own)}")
                continue
            item = Item("own", label, label, drawer, str(path.relative_to(raw_own)), "file", path)
            if label == NOVEL_CLASS:
                item.label, item.split = "unknown_symbol", "test_novel_shapes"   # should be called unknown
                novel.append(item)
            else:
                items.append(item)
    return items, novel, problems


# --------------------------------------------------------------------------
# Step 2: split into train / val / test
# --------------------------------------------------------------------------
def assign_groups_to_splits(group_sizes: dict[str, int], shares: dict[str, float], seed: int) -> dict[str, str]:
    """Put whole groups (drawers) into splits, aiming for the target share of IMAGES per split.

    1. Shuffle the drawers (with a fixed seed, so it's repeatable).
    2. If there are enough drawers, first give one to each of the smaller
       splits (test, val), so none of them is left empty.
    3. Give each remaining drawer, biggest first, to the split that is
       furthest below its target number of images.
    A drawer is only ever placed once, so no drawer can appear in two splits.
    """
    groups = sorted(group_sizes)
    random.Random(seed).shuffle(groups)
    total = sum(group_sizes.values())
    split_names = list(shares)
    assignment, images_in = {}, {s: 0 for s in split_names}

    if len(groups) >= len(split_names):
        for split in sorted(split_names, key=lambda s: shares[s])[:-1]:   # every split but the largest
            group = groups.pop(0)
            assignment[group] = split
            images_in[split] += group_sizes[group]

    for group in sorted(groups, key=lambda g: -group_sizes[g]):          # stable: ties keep shuffled order
        neediest = max(split_names, key=lambda s: shares[s] * total - images_in[s])
        assignment[group] = neediest
        images_in[neediest] += group_sizes[group]
    return assignment


def split_by_drawer(items: list[Item], shares: dict[str, float], seed: int) -> None:
    sizes: dict[str, int] = {}
    for item in items:
        sizes[item.drawer_id] = sizes.get(item.drawer_id, 0) + 1
    assignment = assign_groups_to_splits(sizes, shares, seed)
    for item in items:
        item.split = assignment[item.drawer_id]


def split_randomly(items: list[Item], val_share: float, seed: int) -> None:
    rng = random.Random(seed)
    for item in items:
        item.split = "val" if rng.random() < val_share else "train"


def limit_per_class(items: list[Item], limit: int, seed: int) -> list[Item]:
    """Keep at most `limit` random items per (source, source class). For quick trial runs."""
    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)
    kept, counts = [], {}
    for item in shuffled:
        key = (item.source, item.source_class)
        if counts.get(key, 0) < limit:
            counts[key] = counts.get(key, 0) + 1
            kept.append(item)
    return kept


# --------------------------------------------------------------------------
# Step 3: convert and save images (runs in parallel worker processes)
# --------------------------------------------------------------------------
def process_one(task: tuple) -> str | None:
    """Load one image, convert it to size x size RGB, save it. Returns an error message or None."""
    kind, payload, invert, out_path, size, original = task
    try:
        if kind == "file":
            image = load_photo(payload)
        elif kind == "array":
            pixels = 255 - payload if invert else payload    # white-on-black -> dark-on-light
            image = Image.fromarray(pixels)
        else:  # "bytes"
            image = Image.open(io.BytesIO(payload))
            image.load()
        to_model_size(image, size).save(out_path)
        return None
    except Exception as error:   # one bad file shouldn't stop the other 90,000
        return f"{original}: {error}"


def clear_output_folder(processed: Path) -> None:
    """Delete the old data/processed, but only if this script made it (it has the marker file)."""
    if processed.exists() and any(processed.iterdir()):
        if not (processed / MARKER).exists():
            raise RuntimeError(f"{processed} has files that prepare_data.py did not make; "
                               "refusing to delete them. Move them somewhere else first.")
        shutil.rmtree(processed)
    processed.mkdir(parents=True, exist_ok=True)
    (processed / MARKER).write_text("This folder is rebuilt by scripts/prepare_data.py.\n")


# --------------------------------------------------------------------------
# The whole pipeline
# --------------------------------------------------------------------------
def prepare(cfg: dict, raw_public: Path, raw_own: Path, processed: Path,
            workers: int = 1, limit: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Run steps 1-4. Returns (manifest table, list of problems)."""
    seed = cfg["seed"]
    prep = cfg["prepare"]

    # Step 1: collect
    hds = collect_hds(cfg, raw_public)
    other_public = collect_quickdraw(cfg, raw_public) + collect_hf(cfg, raw_public) + collect_kaggle(cfg, raw_public)
    own, novel, problems = collect_own(cfg, raw_own)
    if limit:
        hds, other_public = limit_per_class(hds, limit, seed), limit_per_class(other_public, limit, seed)
        own, novel = limit_per_class(own, limit, seed), limit_per_class(novel, limit, seed)

    # Step 2: split
    public_shares = {"train": 1 - prep["public_val_share"], "val": prep["public_val_share"]}
    split_by_drawer(hds, public_shares, seed)
    split_randomly(other_public, prep["public_val_share"], seed)
    split_by_drawer(own, prep["own_split_shares"], seed)
    if own and len({i.drawer_id for i in own}) < 3:
        problems.append("fewer than 3 drawers in our own photos: train/val/test can't all get a "
                        "different person, so the test score would not be honest. Add more drawers.")

    # Step 3: convert and save
    clear_output_folder(processed)
    all_items = hds + other_public + own + novel
    rows, tasks = [], []
    for number, item in enumerate(all_items):
        relative = Path(item.split) / item.label / f"{item.source.replace('.', '_')}_{number:06d}.png"
        (processed / relative.parent).mkdir(parents=True, exist_ok=True)
        tasks.append((item.kind, item.payload, item.invert, processed / relative, cfg["image"]["size"], item.original))
        rows.append({"split": item.split, "label": item.label, "file": relative.as_posix(),
                     "source": item.source, "source_class": item.source_class,
                     "drawer_id": item.drawer_id, "original": item.original})

    progress = dict(total=len(tasks), desc="Preparing images", unit="img")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            errors = list(tqdm(pool.map(process_one, tasks, chunksize=256), **progress))
    else:
        errors = [process_one(task) for task in tqdm(tasks, **progress)]

    # Step 4: manifest (rows whose image failed are left out)
    manifest = pd.DataFrame([row for row, error in zip(rows, errors) if error is None],
                            columns=["split", "label", "file", "source", "source_class", "drawer_id", "original"])
    manifest.to_csv(processed / MANIFEST, index=False)
    problems += [f"could not process {e}" for e in errors if e is not None]
    return manifest, problems


def print_summary(manifest: pd.DataFrame, classes: list[str]) -> None:
    print("\nImages per split and class:")
    table = pd.crosstab(manifest["split"], manifest["label"]).reindex(columns=classes, fill_value=0)
    print(table.to_string())
    print("\nImages per source:")
    print(pd.crosstab(manifest["source"], manifest["split"]).to_string())
    if "test" not in set(manifest["split"]):
        print("\nNOTE: no test images yet. The test set is our own photos only "
              "(data/raw_own + metadata.csv, see data/README.md).")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build data/processed/ (train/val/test) from the raw data.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--workers", type=int, default=None, help="parallel processes (default: config prepare.workers)")
    parser.add_argument("--limit-per-class", type=int, default=None,
                        help="keep at most N images per source class (quick trial runs)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    manifest, problems = prepare(cfg, REPO_ROOT / paths["raw_public"], REPO_ROOT / paths["raw_own"],
                                 REPO_ROOT / paths["processed"],
                                 workers=args.workers or cfg["prepare"]["workers"], limit=args.limit_per_class)
    print_summary(manifest, cfg["classes"])
    print(f"\nWrote {len(manifest)} images and {MANIFEST} to {REPO_ROOT / paths['processed']}")
    if problems:
        print(f"\nPROBLEMS ({len(problems)}):")
        for problem in problems[:50]:
            print(f"  - {problem}")
        return 1
    return 0


if __name__ == "__main__":   # required on Windows for the parallel worker processes
    sys.exit(main())
