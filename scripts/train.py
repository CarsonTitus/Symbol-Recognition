"""
train.py - train the symbol classifier and record everything about the run.

What this does:
    1. Loads the settings (e.g. configs/stage_a.yaml) and creates a run folder
       runs/<date-time>_<name>/ holding a copy of the exact settings used.
    2. Builds the model (pretrained MobileNetV3 by default) and the image loaders
       for data/processed/train and data/processed/val.
    3. Trains in two phases (CLAUDE.md §7):
         frozen    only the new final layer learns (backbone frozen), lr 1e-3
         finetune  every layer learns, lr 1e-4 lowered gradually (cosine schedule)
    4. After every epoch (one full pass over the training images) it scores the
       validation images with the reward rubric and writes one row to metrics.csv:
       losses, accuracies, average reward, reward % of max, learning rate, time.
    5. Saves best.pt (highest validation reward so far) and last.pt (latest epoch),
       and stops early if validation reward hasn't improved for 5 epochs.

Watching it train:
    - The terminal prints one summary line per epoch.
    - TensorBoard (live charts in a web browser):  tensorboard --logdir runs
      shows loss, accuracy and the reward convergence graph, plus snapshots of
      the exact images being trained on and the validation images it gets wrong.
    - runs/<run>/what_the_model_sees.png is the first training batch, saved as a picture.

Usage (same in bash and PowerShell):
    python scripts/train.py --config configs/stage_a.yaml --device cuda:0
    python scripts/train.py --config configs/stage_a.yaml --device cuda:1 --data-fraction 0.25
    python scripts/train.py --config configs/stage_a.yaml --model tiny_cnn      # from-scratch comparison
    python scripts/train.py --config configs/stage_a.yaml --device cpu --max-batches 5   # quick smoke test
Press Ctrl+C to stop early; best.pt and last.pt are kept.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from symbolcnn.config import load_config
from symbolcnn.datasets import (SymbolDataset, balanced_sampler, denormalize, eval_transform,
                                load_split, train_transform)
from symbolcnn.models import (TinyCNN, count_parameters, create_model, freeze_backbone,
                              normalization, unfreeze_all)
from symbolcnn.plots import image_grid, save_figure
from symbolcnn.reward import average_reward, reward_matrix_from_config, reward_percent_of_max
from symbolcnn.utils import make_run_dir, pick_device, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
METRIC_COLUMNS = ["epoch", "phase", "lr", "train_loss", "train_acc", "val_loss", "val_acc",
                  "val_reward", "val_reward_pct", "epoch_seconds"]


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the symbol classifier.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "stage_a.yaml"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0 or cuda:1")
    parser.add_argument("--data-fraction", type=float, default=None, help="share of training data (learning curve)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", default=None, help="override model.name, e.g. tiny_cnn")
    parser.add_argument("--name", default=None, help="run name (default: from the config)")
    parser.add_argument("--workers", type=int, default=None, help="image-loading processes (default: config)")
    parser.add_argument("--init", default=None, help="start from this checkpoint (Stage B: the best Stage A best.pt)")
    parser.add_argument("--max-batches", type=int, default=None, help="only N batches per epoch (quick tests)")
    parser.add_argument("--processed", default=None, help="data folder (default: config paths.processed)")
    parser.add_argument("--runs-dir", default=None, help="where run folders go (default: config paths.runs)")
    return parser.parse_args(argv)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Copy command-line overrides into the config, so the saved config.yaml shows what really ran."""
    cfg = copy.deepcopy(cfg)
    if args.data_fraction is not None:
        cfg["data"]["fraction"] = args.data_fraction
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.model is not None:
        cfg["model"]["name"] = args.model
    if args.workers is not None:
        cfg["data"]["num_workers"] = args.workers
    if args.init is not None:
        cfg.setdefault("stage_b", {})["init_checkpoint"] = args.init
    return cfg


def run_name(cfg: dict, args: argparse.Namespace) -> str:
    name = args.name or cfg.get("run_name", "train")
    if cfg["model"]["name"] == "tiny_cnn":
        name += "_tinycnn"
    if cfg["data"]["fraction"] < 1.0:
        name += f"_frac{round(cfg['data']['fraction'] * 100)}"
    if cfg["seed"] != 0:
        name += f"_seed{cfg['seed']}"
    return name


# --------------------------------------------------------------------------
# One epoch of training / one pass of validation
# --------------------------------------------------------------------------
def seed_worker(worker_id: int) -> None:
    """Give each image-loading process its own, repeatable random seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, log) -> dict:
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for step, (images, labels) in enumerate(tqdm(loader, desc=f"  epoch {log['epoch']} train", leave=False)):
        if log["max_batches"] and step >= log["max_batches"]:
            break
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Mixed precision: on a GPU, do most math in 16-bit numbers (faster, less memory).
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=use_amp):
            scores = model(images)              # one score per class for each image ("logits")
            loss = criterion(scores, labels)    # how wrong the scores are

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()           # work out how each weight should change
        scaler.step(optimizer)                  # change the weights a little
        scaler.update()

        total_loss += loss.item() * len(labels)
        correct += (scores.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
        log["global_step"] += 1

        # Snapshot of the exact images being trained on (cheap: a few times per epoch).
        if log["global_step"] == 1 or log["global_step"] % log["images_every"] == 0:
            pictures = denormalize(images[:32], log["mean"], log["std"])
            captions = [log["classes"][i] for i in labels[:32].tolist()]
            figure = image_grid(pictures, captions, title=f"Training batch at step {log['global_step']}")
            log["writer"].add_figure("train/what_the_model_sees", figure, log["global_step"])
            if log["global_step"] == 1:
                save_figure(image_grid(pictures, captions, title="First training batch (after augmentation)"),
                            log["run_dir"] / "what_the_model_sees.png")
    return {"loss": total_loss / max(seen, 1), "acc": correct / max(seen, 1)}


@torch.no_grad()   # no learning during validation, so skip the gradient bookkeeping
def validate(model, loader, criterion, device, use_amp, max_batches, keep_mistakes: int) -> dict:
    model.eval()
    total_loss, all_probs, all_labels, mistakes = 0.0, [], [], []
    for step, (images, labels) in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=use_amp):
            scores = model(images)
            loss = criterion(scores, labels)
        probs = scores.float().softmax(dim=1)         # scores -> probabilities that add up to 1
        total_loss += loss.item() * len(labels)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

        # Remember the most confident WRONG answers, to look at later.
        confidence, predicted = probs.max(dim=1)
        for i in torch.nonzero(predicted != labels).flatten().tolist():
            mistakes.append((confidence[i].item(), images[i].cpu(), labels[i].item(), predicted[i].item()))
        mistakes = sorted(mistakes, key=lambda m: -m[0])[:keep_mistakes]

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    return {"loss": total_loss / len(labels), "acc": float((probs.argmax(1) == labels).mean()),
            "probs": probs, "labels": labels, "mistakes": mistakes}


def save_checkpoint(path: Path, model, cfg: dict, epoch: int, metrics: dict) -> None:
    torch.save({"model_state": model.state_dict(), "model_name": cfg["model"]["name"],
                "classes": cfg["classes"], "image_size": cfg["image"]["size"],
                "config": cfg, "epoch": epoch, "metrics": metrics}, path)


# --------------------------------------------------------------------------
# The whole training run
# --------------------------------------------------------------------------
def run_training(cfg: dict, device: str, processed: Path, runs_root: Path, name: str,
                 max_batches: int | None = None) -> Path:
    set_seed(cfg["seed"])
    classes, train_cfg = cfg["classes"], cfg["train"]
    run_dir = make_run_dir(runs_root, name, cfg)
    print(f"Run folder: {run_dir}")

    # ---- model
    model = create_model(cfg)
    init = (cfg.get("stage_b") or {}).get("init_checkpoint")
    if init:
        model.load_state_dict(torch.load(init, map_location="cpu", weights_only=False)["model_state"])
        print(f"Started from checkpoint {init}")
    mean, std = normalization(model)
    model.to(device)

    # ---- data
    fraction = cfg["data"]["fraction"]
    train_rows = load_split(processed, "train", fraction, cfg["seed"])
    val_rows = load_split(processed, "val")
    if len(train_rows) == 0 or len(val_rows) == 0:
        raise SystemExit(f"No train/val images in {processed}. Run scripts/prepare_data.py first.")
    print(f"Training images: {len(train_rows)} ({fraction:.0%} of train) "
          f"{train_rows['label'].value_counts().reindex(classes, fill_value=0).to_dict()}")
    print(f"Validation images: {len(val_rows)}")

    train_set = SymbolDataset(train_rows, processed, classes, train_transform(cfg, mean, std))
    val_set = SymbolDataset(val_rows, processed, classes, eval_transform(mean, std))
    workers = cfg["data"]["num_workers"]
    loader_settings = dict(num_workers=workers, pin_memory=device.startswith("cuda"),
                           persistent_workers=workers > 0, worker_init_fn=seed_worker)
    batch_size = train_cfg["batch_size"]
    sampler = balanced_sampler(train_set.labels, len(classes), cfg["seed"]) if cfg["data"]["weighted_sampler"] else None
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, shuffle=sampler is None,
                              drop_last=len(train_set) > batch_size,   # avoid a last batch of 1 image
                              generator=torch.Generator().manual_seed(cfg["seed"]), **loader_settings)
    val_loader = DataLoader(val_set, batch_size=batch_size * 2, shuffle=False, **loader_settings)

    # ---- training tools
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])
    use_amp = bool(train_cfg["mixed_precision"]) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    reward_matrix = reward_matrix_from_config(cfg)
    writer = SummaryWriter(run_dir / "tensorboard")
    log = {"writer": writer, "run_dir": run_dir, "classes": classes, "mean": mean, "std": std,
           "global_step": 0, "images_every": train_cfg["log_images_every"], "max_batches": max_batches, "epoch": 0}

    phases = [("frozen", train_cfg["frozen_epochs"], train_cfg["frozen_lr"]),
              ("finetune", train_cfg["finetune_epochs"], train_cfg["finetune_lr"])]
    if isinstance(model, TinyCNN):   # nothing pretrained to protect, so no frozen phase
        phases = [phases[1]]
    phases = [p for p in phases if p[1] > 0]
    total_epochs = sum(p[1] for p in phases)
    print(f"Model: {cfg['model']['name']} ({count_parameters(model):,} parameters) on {device}"
          f"{' with mixed precision' if use_amp else ''}; up to {total_epochs} epochs\n")

    with (run_dir / "metrics.csv").open("w", newline="") as f:
        csv.writer(f).writerow(METRIC_COLUMNS)

    best = {"val_reward": -1.0}
    epochs_without_improvement, patience = 0, train_cfg["early_stopping_patience"]
    stop = False
    try:
        for phase, phase_epochs, learning_rate in phases:
            if phase == "frozen":
                freeze_backbone(model)
            else:
                unfreeze_all(model)
            # AdamW = the optimizer that changes the weights; it only gets the layers allowed to learn.
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=learning_rate, weight_decay=train_cfg["weight_decay"])
            scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase_epochs)
                         if phase == "finetune" and train_cfg["scheduler"] == "cosine" else None)
            print(f"Phase '{phase}': {count_parameters(model, trainable_only=True):,} trainable parameters, "
                  f"lr {learning_rate:g}")

            for _ in range(phase_epochs):
                log["epoch"] += 1
                epoch = log["epoch"]
                started = time.time()
                lr = optimizer.param_groups[0]["lr"]
                train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp, log)
                val = validate(model, val_loader, criterion, device, use_amp, max_batches,
                               train_cfg["mistakes_to_show"])
                if scheduler:
                    scheduler.step()

                predictions = val["probs"].argmax(1)
                row = {"epoch": epoch, "phase": phase, "lr": lr,
                       "train_loss": train_stats["loss"], "train_acc": train_stats["acc"],
                       "val_loss": val["loss"], "val_acc": val["acc"],
                       "val_reward": average_reward(val["labels"], predictions, reward_matrix),
                       "val_reward_pct": reward_percent_of_max(val["labels"], predictions, reward_matrix),
                       "epoch_seconds": time.time() - started}
                with (run_dir / "metrics.csv").open("a", newline="") as f:
                    csv.writer(f).writerow([row[c] for c in METRIC_COLUMNS])

                # TensorBoard charts
                writer.add_scalars("loss", {"train": row["train_loss"], "val": row["val_loss"]}, epoch)
                writer.add_scalars("accuracy", {"train": row["train_acc"], "val": row["val_acc"]}, epoch)
                writer.add_scalar("reward/val_average", row["val_reward"], epoch)
                writer.add_scalar("reward/val_percent_of_max", row["val_reward_pct"], epoch)
                writer.add_scalar("learning_rate", lr, epoch)
                if val["mistakes"]:
                    _, images, true, predicted = zip(*val["mistakes"])
                    captions = [f"true: {classes[t]}\npred: {classes[p]} ({c:.0%})"
                                for (c, _, t, p) in val["mistakes"]]
                    writer.add_figure("val/most_confident_mistakes",
                                      image_grid(denormalize(torch.stack(images), mean, std), captions,
                                                 title=f"Most confident mistakes, epoch {epoch}"), epoch)
                writer.flush()

                # Checkpoints and early stopping
                save_checkpoint(run_dir / "last.pt", model, cfg, epoch, row)
                improved = row["val_reward"] > best["val_reward"]
                if improved:
                    best = dict(row)
                    save_checkpoint(run_dir / "best.pt", model, cfg, epoch, row)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                print(f"Epoch {epoch:2d}/{total_epochs} [{phase:8s}] "
                      f"train loss {row['train_loss']:.3f} acc {row['train_acc']:6.1%} | "
                      f"val loss {row['val_loss']:.3f} acc {row['val_acc']:6.1%} "
                      f"reward {row['val_reward']:.3f} ({row['val_reward_pct']:5.1f}% of max) | "
                      f"{row['epoch_seconds']:.0f}s{'  *best*' if improved else ''}")

                if epochs_without_improvement >= patience:
                    print(f"\nStopping early: validation reward hasn't improved for {patience} epochs.")
                    stop = True
                    break
            if stop:
                break
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C. best.pt and last.pt hold the progress so far.")
    finally:
        writer.close()

    (run_dir / "summary.json").write_text(json.dumps({"best": best}, indent=2))
    if best["val_reward"] >= 0:
        print(f"\nBest epoch {best['epoch']}: val accuracy {best['val_acc']:.1%}, "
              f"val reward {best['val_reward']:.3f} ({best['val_reward_pct']:.1f}% of max)")
    print(f"Saved: {run_dir / 'best.pt'}")
    print(f"See the charts:  tensorboard --logdir {runs_root}")
    return run_dir


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args)
    device = pick_device(args.device)
    processed = Path(args.processed) if args.processed else REPO_ROOT / cfg["paths"]["processed"]
    runs_root = Path(args.runs_dir) if args.runs_dir else REPO_ROOT / cfg["paths"]["runs"]
    run_training(cfg, device, processed, runs_root, run_name(cfg, args), args.max_batches)
    return 0


if __name__ == "__main__":   # required on Windows for the image-loading worker processes
    sys.exit(main())
