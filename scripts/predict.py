"""
predict.py - ask the trained model what shape is in a photo (the terminal demo, CLAUDE.md §10).

What this does:
    1. Loads a trained model (best.pt from a run folder, or the newest run with --run latest).
    2. Prepares each photo EXACTLY like the training images (src/symbolcnn/imageprep.py):
       turn it the right way up, pad to a square, resize to 224x224.
    3. Runs the model, which gives one probability per class (they add up to 100%).
    4. Prints one line per photo, e.g.
         card_photo.jpg -> triangle (94.1%)   [no_symbol 1.2% | unknown_symbol 4.7% | triangle 94.1% | ...]
       If the model's top probability is below the confidence threshold (0.6 by default,
       CLAUDE.md §3), the answer becomes unknown_symbol and the line says "unsure".

Crop first: the model names the outline of the MAIN thing in the image, so crop the
photo so the card or object fills most of the frame (this is what "Step 1" will do in the app).

Usage (same in bash and PowerShell):
    python scripts/predict.py --run latest --image card_photo.jpg
    python scripts/predict.py --run runs/20261120-082954_stage_a_five_classes --image photos/
    python scripts/predict.py --run latest --image a.jpg b.jpg --device cpu --threshold 0
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from symbolcnn.datasets import eval_transform
from symbolcnn.imageprep import load_photo, to_model_size
from symbolcnn.models import normalization
from symbolcnn.reward import apply_confidence_threshold
from symbolcnn.utils import pick_device

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Reuse load_checkpoint() from evaluate.py (a script, so it is loaded by file path).
_spec = importlib.util.spec_from_file_location("evaluate", REPO_ROOT / "scripts" / "evaluate.py")
evaluate = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("evaluate", evaluate)
_spec.loader.exec_module(evaluate)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify photos with a trained model.")
    parser.add_argument("--run", required=True, help="run folder, or 'latest' for the newest run in runs/")
    parser.add_argument("--image", required=True, nargs="+", help="one or more image files and/or folders")
    parser.add_argument("--checkpoint", default="best.pt", help="file inside the run folder")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0 or cuda:1")
    parser.add_argument("--threshold", type=float, default=None,
                        help="confidence threshold (default: the run's config; 0 switches it off)")
    parser.add_argument("--runs-dir", default=str(REPO_ROOT / "runs"), help="where to look for --run latest")
    return parser.parse_args(argv)


def find_run(run: str, runs_dir: Path, checkpoint: str = "best.pt") -> Path:
    """'latest' -> the newest folder in runs/ that has a finished checkpoint; otherwise the given folder."""
    if run != "latest":
        return Path(run)
    # Folder names start with the date and time (YYYYMMDD-HHMMSS), so sorting by name sorts by age.
    finished = sorted(d for d in Path(runs_dir).iterdir() if (d / checkpoint).exists())
    if not finished:
        raise SystemExit(f"No finished runs (with {checkpoint}) in {runs_dir}. Train one with scripts/train.py.")
    return finished[-1]


def list_images(paths: list[str]) -> list[Path]:
    """Expand folders into the image files inside them (sorted); keep files as given."""
    images = []
    for path in map(Path, paths):
        if path.is_dir():
            images += sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        elif path.exists():
            images.append(path)
        else:
            print(f"Not found, skipped: {path}")
    return images


@torch.no_grad()   # no learning here, so skip the gradient bookkeeping
def classify(model, cfg: dict, image_path: Path, device: str) -> np.ndarray:
    """Probabilities for one photo, one per class, in config "classes" order."""
    picture = to_model_size(load_photo(image_path), cfg["image"]["size"])   # same steps as training data
    mean, std = normalization(model)
    tensor = eval_transform(mean, std)(image=np.asarray(picture))["image"]   # scale pixels, make a tensor
    scores = model(tensor.unsqueeze(0).to(device))                            # unsqueeze: a batch of 1
    return scores.float().softmax(dim=1)[0].cpu().numpy()                     # scores -> probabilities


def describe(name: str, probs: np.ndarray, classes: list[str], threshold: float) -> tuple[str, str]:
    """Build the output line. Returns (answer, line)."""
    unknown = classes.index("unknown_symbol")
    answer = classes[apply_confidence_threshold(probs[None, :], threshold, unknown)[0]]
    top = int(probs.argmax())
    all_probs = " | ".join(f"{c} {p:.1%}" for c, p in zip(classes, probs))
    if answer == classes[top]:
        headline = f"{answer} ({probs[top]:.1%})"
    else:   # the safety net kicked in: the model's best guess was not confident enough
        headline = f"{answer}  (unsure: best guess {classes[top]} {probs[top]:.1%} < threshold {threshold:.0%})"
    return answer, f"{name} -> {headline}   [{all_probs}]"


def main(argv=None) -> list[dict]:
    args = parse_args(argv)
    run_dir = find_run(args.run, Path(args.runs_dir), args.checkpoint)
    model, cfg = evaluate.load_checkpoint(run_dir / args.checkpoint)
    device = pick_device(args.device)
    model.to(device)
    classes = cfg["classes"]
    threshold = cfg["confidence_threshold"] if args.threshold is None else args.threshold
    print(f"Model: {run_dir.name}/{args.checkpoint} on {device}, confidence threshold {threshold}")

    results = []
    for image_path in list_images(args.image):
        try:
            probs = classify(model, cfg, image_path, device)
        except OSError as error:            # not an image, or a damaged file
            print(f"{image_path.name} -> could not read ({error})")
            continue
        answer, line = describe(image_path.name, probs, classes, threshold)
        print(line)
        results.append({"image": str(image_path), "answer": answer,
                        "probabilities": dict(zip(classes, probs.tolist()))})
    if not results:
        print("No images classified.")
    return results


if __name__ == "__main__":
    main()
