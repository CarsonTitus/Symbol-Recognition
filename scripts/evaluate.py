"""
evaluate.py - measure how good a trained model is (CLAUDE.md §8, Lesson 36 and 41).

What this does:
    1. Loads a trained model from a run folder (runs/<timestamp>_<name>/best.pt).
    2. Runs it on one split of data/processed, with NO random changes to the images:
           test               our own photos (the honest final score)
           val                validation images (use this until we have our own photos)
           test_novel_shapes  shapes never seen in training (should be called unknown_symbol)
    3. Scores the answers two ways:
           "no threshold"   = the model's top answer, always
           "with threshold" = if the top answer's confidence is below the config's
                              confidence_threshold (0.6), say unknown_symbol instead (CLAUDE.md §3)
    4. Prints the results and saves into runs/<run>/eval/<split>/:
           metrics.json             all the numbers below, in one file
           confusion_matrix.png/csv true vs predicted counts and row percentages (required, §2)
           per_class.csv            precision, recall, F1 per class
           predictions.csv          one row per image: file, true label, prediction, confidence
           error_gallery.png        the most confidently WRONG images (for the failures discussion)
       and into runs/<run>/eval/:
           reward_convergence.png   average validation reward per epoch (required, §2)
           training_curves.png      loss and accuracy per epoch, train vs validation

Why:
    The submitted scope promises a confusion matrix, a reward convergence graph and
    90% test accuracy. The reward is reported raw AND as a percentage of the best
    possible, because a correct unknown_symbol only earns 0.5 (CLAUDE.md §2).

How to run (from the repo root, with the virtual environment active):
    python scripts/evaluate.py --run runs/<timestamp>_stage_a --split val
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from symbolcnn.datasets import SymbolDataset, eval_transform, load_split
from symbolcnn.metrics import (accuracy, confusion_matrix, novel_shape_summary, per_class_stats,
                               row_percentages)
from symbolcnn.models import create_model, normalization
from symbolcnn.plots import (confusion_matrix_figure, image_grid, reward_convergence_figure, save_figure,
                             training_curves_figure)
from symbolcnn.reward import (apply_confidence_threshold, average_reward, max_possible_average_reward,
                              reward_matrix_from_config, reward_percent_of_max)
from symbolcnn.utils import pick_device

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["test", "val", "test_novel_shapes"]


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained symbol classifier.")
    parser.add_argument("--run", required=True, help="run folder, e.g. runs/20261118-111521_stage_a")
    parser.add_argument("--checkpoint", default="best.pt", help="file inside the run folder (best.pt or last.pt)")
    parser.add_argument("--split", default="test", choices=SPLITS)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0 or cuda:1")
    parser.add_argument("--threshold", type=float, default=None,
                        help="confidence threshold (default: confidence_threshold in the run's config)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=None, help="image-loading processes (default: config)")
    parser.add_argument("--processed", default=None, help="data folder (default: config paths.processed)")
    parser.add_argument("--errors", type=int, default=32, help="how many mistakes to show in the error gallery")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# Loading the model and running it
# --------------------------------------------------------------------------
def load_checkpoint(path: Path):
    """Rebuild the model saved by train.py. Returns (model, config saved with it)."""
    # weights_only=False: the checkpoint also holds the run's config (a plain dict).
    # Only load checkpoints we made ourselves; this setting can run code hidden in a file.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = create_model(cfg, pretrained=False)   # no download: we load our own trained weights next
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, cfg


@torch.no_grad()   # no learning here, so skip the gradient bookkeeping
def predict_probabilities(model, loader, device: str) -> np.ndarray:
    """Run every image through the model. Returns probabilities, shape (images, classes)."""
    model.to(device)
    use_amp = device.startswith("cuda")   # 16-bit math on the GPU: faster, same answers in practice
    all_probs = []
    for images, _ in tqdm(loader, desc="  predicting", leave=False):
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=use_amp):
            scores = model(images.to(device, non_blocking=True))
        all_probs.append(scores.float().softmax(dim=1).cpu())   # scores -> probabilities that add up to 1
    return torch.cat(all_probs).numpy()


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score(y_true, y_pred, classes: list[str], reward_matrix: np.ndarray) -> dict:
    """Every number for one set of predictions."""
    counts = confusion_matrix(y_true, y_pred, len(classes))
    return {
        "accuracy": accuracy(y_true, y_pred),
        "average_reward": average_reward(y_true, y_pred, reward_matrix),
        "max_possible_reward": max_possible_average_reward(y_true, reward_matrix),
        "reward_percent_of_max": reward_percent_of_max(y_true, y_pred, reward_matrix),
        "per_class": per_class_stats(counts, classes),
        "confusion_matrix": counts.tolist(),
    }


def print_report(name: str, result: dict, classes: list[str]) -> None:
    """The terminal summary (what we show at the Lesson 36 checkpoint)."""
    print(f"\n=== {name} ===")
    print(f"Accuracy:        {result['accuracy'] * 100:.2f}%")
    print(f"Average reward:  {result['average_reward']:.4f}  (best possible {result['max_possible_reward']:.4f})")
    print(f"Reward % of max: {result['reward_percent_of_max']:.2f}%")
    print("\nConfusion matrix (rows = true label, columns = predicted):")
    print(pd.DataFrame(result["confusion_matrix"], index=classes, columns=classes).to_string())
    print("\nPer class (recall = that class's accuracy):")
    table = pd.DataFrame(result["per_class"]).set_index("class")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))


# --------------------------------------------------------------------------
# Pictures
# --------------------------------------------------------------------------
def save_error_gallery(rows: pd.DataFrame, probs: np.ndarray, y_true, y_pred, classes, processed: Path,
                       count: int, path: Path) -> int:
    """Grid of the most confidently WRONG images. Returns how many were shown."""
    wrong = np.flatnonzero(y_pred != y_true)
    if wrong.size == 0:
        return 0
    confidence = probs[wrong, y_pred[wrong]]
    worst = wrong[np.argsort(-confidence)][:count]          # most confident mistakes first
    images, captions = [], []
    for i in worst:
        image = cv2.imread(str(processed / rows.loc[i, "file"]), cv2.IMREAD_COLOR)
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        captions.append(f"true: {classes[y_true[i]]}\npred: {classes[y_pred[i]]} ({probs[i, y_pred[i]]:.0%})")
    figure = image_grid(np.stack(images), captions,
                        title="Most confident mistakes (true label / model's answer and confidence)")
    save_figure(figure, path)
    return len(worst)


def save_training_plots(run_dir: Path, reward_matrix: np.ndarray, processed: Path, classes) -> None:
    """Reward convergence graph and train-vs-validation curves, read from the run's metrics.csv."""
    metrics_file = run_dir / "metrics.csv"
    if not metrics_file.exists():
        return
    history = pd.read_csv(metrics_file)
    out = run_dir / "eval"
    # The best possible reward depends on the mix of classes in the validation set.
    val_labels = [classes.index(label) for label in load_split(processed, "val")["label"]]
    best = max_possible_average_reward(val_labels, reward_matrix)
    save_figure(reward_convergence_figure(history["epoch"].tolist(), history["val_reward"].tolist(), best,
                                          phases=history["phase"].tolist(),
                                          title=f"Reward convergence: {run_dir.name}"),
                out / "reward_convergence.png")
    save_figure(training_curves_figure(history["epoch"], history["train_loss"], history["val_loss"],
                                       history["train_acc"], history["val_acc"]),
                out / "training_curves.png")


# --------------------------------------------------------------------------
# The whole evaluation
# --------------------------------------------------------------------------
def run_evaluation(run_dir: Path, split: str, device: str, checkpoint: str = "best.pt",
                   threshold: float | None = None, processed: Path | None = None, batch_size: int = 128,
                   workers: int | None = None, errors: int = 32) -> dict:
    model, cfg = load_checkpoint(run_dir / checkpoint)
    classes = cfg["classes"]
    reward_matrix = reward_matrix_from_config(cfg)
    threshold = cfg["confidence_threshold"] if threshold is None else threshold
    processed = Path(processed) if processed else REPO_ROOT / cfg["paths"]["processed"]

    rows = load_split(processed, split)
    if rows.empty:
        raise SystemExit(f"No images in split '{split}' of {processed}. The test set is our own photos only, "
                         "so it stays empty until we add some (data/README.md). Meanwhile use --split val.")
    mean, std = normalization(model)
    dataset = SymbolDataset(rows, processed, classes, eval_transform(mean, std))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=cfg["data"]["num_workers"] if workers is None else workers,
                        pin_memory=device.startswith("cuda"))
    print(f"Evaluating {run_dir.name}/{checkpoint} on '{split}' ({len(rows)} images) on {device}")
    probs = predict_probabilities(model, loader, device)

    y_true = np.asarray(dataset.labels)
    unknown = classes.index("unknown_symbol")
    predictions = {"no threshold": probs.argmax(axis=1),
                   f"with threshold {threshold}": apply_confidence_threshold(probs, threshold, unknown)}

    out = run_dir / "eval" / split
    out.mkdir(parents=True, exist_ok=True)
    results = {"run": run_dir.name, "checkpoint": checkpoint, "split": split, "images": len(rows),
               "threshold": threshold}
    for name, y_pred in predictions.items():
        result = score(y_true, y_pred, classes, reward_matrix)
        print_report(name, result, classes)
        if split == "test_novel_shapes":
            result["novel_shapes"] = novel_shape_summary(y_pred, classes)
            print(f"Never-seen shapes called unknown_symbol: {result['novel_shapes']['called_unknown_symbol']:.1%}"
                  f", called triangle (false alarm): {result['novel_shapes']['called_triangle']:.1%}")
        results[name] = result

    # Files for the presentation. The confusion matrix and gallery use the plain (no threshold) answers.
    plain = predictions["no threshold"]
    plain_result = results["no threshold"]
    counts = np.asarray(plain_result["confusion_matrix"])
    percents = row_percentages(counts)
    pd.DataFrame(counts, index=classes, columns=classes).to_csv(out / "confusion_matrix.csv")
    pd.DataFrame(percents.round(2), index=classes, columns=classes).to_csv(out / "confusion_matrix_percent.csv")
    save_figure(confusion_matrix_figure(counts, percents, classes,
                                        f"{run_dir.name} on {split}: accuracy {plain_result['accuracy']:.1%}"),
                out / "confusion_matrix.png")
    pd.DataFrame(plain_result["per_class"]).to_csv(out / "per_class.csv", index=False)
    table = pd.DataFrame({"file": rows["file"], "true": [classes[i] for i in y_true],
                          "predicted": [classes[i] for i in plain], "confidence": probs.max(axis=1)})
    for k, name in enumerate(classes):
        table[f"prob_{name}"] = probs[:, k]
    table.to_csv(out / "predictions.csv", index=False)
    shown = save_error_gallery(rows, probs, y_true, plain, classes, processed, errors, out / "error_gallery.png")
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    save_training_plots(run_dir, reward_matrix, processed, classes)

    print(f"\nSaved results to {out}" + (f" ({shown} mistakes in error_gallery.png)" if shown else ""))
    print(f"Reward convergence graph and training curves: {run_dir / 'eval'}")
    return results


def main(argv=None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run)
    if not (run_dir / args.checkpoint).exists():
        sys.exit(f"Not found: {run_dir / args.checkpoint}. Give the run folder made by train.py, "
                 "e.g. --run runs/20261118-111521_stage_a")
    run_evaluation(run_dir, args.split, pick_device(args.device), args.checkpoint, args.threshold,
                   args.processed, args.batch_size, args.workers, args.errors)


# Required on Windows: image-loading worker processes re-import this file (CLAUDE.md §0.3).
if __name__ == "__main__":
    main()
