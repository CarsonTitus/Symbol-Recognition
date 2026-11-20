"""
export_tflite.py - convert a trained model into a phone model file (TFLite / LiteRT) and check it.

What this does (CLAUDE.md §12):
    1. Loads best.pt from a training run (--run latest = the newest run in runs/).
    2. Wraps the model so the phone's job is as simple as possible:
         input   one 224x224 RGB image, pixel values 0-1, laid out [1, 224, 224, 3]
                 (height, width, color: the order Android bitmaps already use)
         output  one probability per class, adding up to 1
       The color scaling ("normalization") and the softmax (scores -> probabilities)
       happen INSIDE the model file, so the app can't get them wrong.
    3. Converts it with Google's litert-torch converter into a .tflite file.
       TFLite is Google's format for running AI on phones; it is now called LiteRT.
    4. Checks the phone model against the original PyTorch model on 100 validation
       images: both must give the same answers (CLAUDE.md §12: verify before trusting it).
    5. Writes into exports/<run name>/:
         symbolcnn.tflite     the phone model
         labels.txt           class names, one per line, in output order
         export_report.json   the check results and the model's settings
       and, with --android, copies the model and labels into the Android app
       (android/app/src/main/assets/), ready to build.

Why a separate Python environment:
    The converter needs an older PyTorch than our training environment, so it lives
    in .venv-export (see README "Export the model for the phone"). Training is unaffected.

Usage (bash, from the repo root):
    .venv-export/bin/python scripts/export_tflite.py --run latest --android
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from symbolcnn.datasets import load_split
from symbolcnn.models import create_model, normalization

REPO_ROOT = Path(__file__).resolve().parents[1]
ANDROID_ASSETS = REPO_ROOT / "android" / "app" / "src" / "main" / "assets"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained model to TFLite for the phone.")
    parser.add_argument("--run", required=True, help="run folder, or 'latest' for the newest run in runs/")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--check-images", type=int, default=100, help="validation images used for the check")
    parser.add_argument("--processed", default=None, help="data folder (default: config paths.processed)")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "exports"))
    parser.add_argument("--android", action="store_true", help="also copy the model into android/app assets")
    return parser.parse_args(argv)


def find_run(run: str, checkpoint: str) -> Path:
    """'latest' -> newest folder in runs/ with a finished checkpoint (names start with date-time)."""
    if run != "latest":
        return Path(run)
    finished = sorted(d for d in (REPO_ROOT / "runs").iterdir() if (d / checkpoint).exists())
    if not finished:
        raise SystemExit("No finished runs in runs/. Train one with scripts/train.py first.")
    return finished[-1]


class PhoneModel(nn.Module):
    """The trained model plus the steps the phone would otherwise have to do itself.

    [1, 224, 224, 3] pixels 0-1  ->  reorder to [1, 3, 224, 224] (PyTorch's layout)
                                 ->  normalize with the model's mean/std
                                 ->  model  ->  softmax  ->  [1, classes] probabilities
    """

    def __init__(self, model: nn.Module, mean, std):
        super().__init__()
        self.model = model
        # register_buffer: saved with the model as fixed numbers (not learned)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = pixels.permute(0, 3, 1, 2)          # height-width-color -> color-height-width
        x = (x - self.mean) / self.std
        return self.model(x).softmax(dim=1)


def load_model(path: Path):
    """Rebuild the trained model from best.pt (same as evaluate.load_checkpoint)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)   # our own file, so this is safe
    cfg = checkpoint["config"]
    model = create_model(cfg, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), cfg


def check_images(processed: Path, count: int, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """`count` validation images spread evenly over the classes, as [N, 224, 224, 3] floats 0-1."""
    rows = load_split(processed, "val")
    per_class = max(1, count // rows["label"].nunique())
    rows = rows.groupby("label").sample(n=per_class, random_state=seed)   # the same number from every class
    images = [cv2.cvtColor(cv2.imread(str(processed / f)), cv2.COLOR_BGR2RGB) for f in rows["file"]]
    return np.stack(images).astype(np.float32) / 255.0, rows["label"].tolist()


def main(argv=None) -> int:
    args = parse_args(argv)
    run_dir = find_run(args.run, args.checkpoint)
    model, cfg = load_model(run_dir / args.checkpoint)
    classes = cfg["classes"]
    mean, std = normalization(model)
    phone_model = PhoneModel(model, mean, std).eval()
    size = cfg["image"]["size"]
    print(f"Exporting {run_dir.name}/{args.checkpoint}: classes {classes}, input {size}x{size}")

    # --- Convert. litert_torch traces the model with an example input of the right shape.
    import litert_torch   # imported here: only the export environment has it
    example = (torch.rand(1, size, size, 3),)
    edge_model = litert_torch.convert(phone_model, example)

    out = Path(args.out_dir) / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "symbolcnn.tflite"
    edge_model.export(str(model_path))
    (out / "labels.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    print(f"Saved {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")

    # --- Check: same answers as PyTorch on real validation images?
    processed = Path(args.processed) if args.processed else REPO_ROOT / cfg["paths"]["processed"]
    images, labels = check_images(processed, args.check_images)
    with torch.no_grad():
        torch_probs = phone_model(torch.from_numpy(images)).numpy()
    start = time.perf_counter()
    phone_probs = np.concatenate([np.asarray(edge_model(images[i:i + 1])) for i in range(len(images))])
    ms_per_image = 1000 * (time.perf_counter() - start) / len(images)

    same_answer = float((torch_probs.argmax(1) == phone_probs.argmax(1)).mean())
    biggest_gap = float(np.abs(torch_probs - phone_probs).max())
    correct = float((phone_probs.argmax(1) == np.array([classes.index(l) for l in labels])).mean())
    report = {"run": run_dir.name, "checkpoint": args.checkpoint, "classes": classes,
              "input": f"[1, {size}, {size}, 3] RGB floats 0-1", "output": "[1, classes] probabilities",
              "confidence_threshold": cfg["confidence_threshold"], "check_images": len(images),
              "same_answer_as_pytorch": same_answer, "largest_probability_difference": biggest_gap,
              "phone_model_accuracy_on_check_images": correct, "ms_per_image_on_tower_cpu": ms_per_image}
    (out / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Check on {len(images)} validation images:")
    print(f"  same answer as PyTorch:        {same_answer:.1%}")
    print(f"  largest probability difference: {biggest_gap:.5f}")
    print(f"  phone model accuracy:          {correct:.1%}")
    print(f"  speed on the tower's CPU:      {ms_per_image:.1f} ms per image")
    if same_answer < 1.0 or biggest_gap > 1e-3:
        print("WARNING: the phone model does not match PyTorch closely. Don't use it until this is explained.")
        return 1

    if args.android:
        ANDROID_ASSETS.mkdir(parents=True, exist_ok=True)
        for name in ["symbolcnn.tflite", "labels.txt"]:
            shutil.copy2(out / name, ANDROID_ASSETS / name)
        (ANDROID_ASSETS / "model_info.json").write_text(json.dumps(
            {"run": run_dir.name, "confidence_threshold": cfg["confidence_threshold"], "input_size": size},
            indent=2), encoding="utf-8")
        print(f"Copied the model into {ANDROID_ASSETS.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
