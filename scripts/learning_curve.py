"""
learning_curve.py - how much does the model improve when it gets more training data?

What this does (CLAUDE.md §2 "different dataset sizes", §8 item 6):
    1. Trains the same model several times, each time on a different share
       ("fraction") of the training images: 10%, 25%, 50% and 100% by default.
       Every class keeps the same share, so the class balance never changes.
    2. Runs the trainings side by side, one queue per GPU (e.g. cuda:0 and cuda:1).
       Each training is an ordinary `scripts/train.py` run in its own process,
       with its terminal output saved to logs/frac<N>.log.
    3. Scores every finished run's best checkpoint with evaluate.py and writes:
         learning_curve.csv   one row per fraction: training images, accuracy, reward
         learning_curve.png   accuracy and reward % of max vs number of training images

Why:
    It shows whether more data still helps. If the curve is still climbing at
    100%, collecting more photos is worth it. If it has flattened, more data of
    the same kind won't help much.

Where things go:
    runs/<date-time>_learning_curve/
        frac10/ frac25/ ...    hold the four normal training run folders
        logs/                  terminal output of each training
        learning_curve.csv/png the results

Usage (same in bash and PowerShell):
    python scripts/learning_curve.py --devices cuda:0 cuda:1           # train all, then score on val
    python scripts/learning_curve.py --from runs/<..>_learning_curve --split test
        re-scores finished runs without retraining (e.g. once our own test photos exist)

Note: until our own photos exist, the test split is empty, so the default --split is val.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd

from symbolcnn.config import load_config
from symbolcnn.datasets import load_split
from symbolcnn.plots import learning_curve_figure, save_figure

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"

# evaluate.py is a script, not part of the symbolcnn package, so load it by file path.
_spec = importlib.util.spec_from_file_location("evaluate", REPO_ROOT / "scripts" / "evaluate.py")
evaluate = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("evaluate", evaluate)
_spec.loader.exec_module(evaluate)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on 10/25/50/100%% of the data and compare.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "stage_a.yaml"))
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--devices", nargs="+", default=["cuda:0"],
                        help="one training queue per device, e.g. --devices cuda:0 cuda:1")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="which images to score on")
    parser.add_argument("--from", dest="from_dir", default=None,
                        help="an earlier learning-curve folder: skip training, just score its runs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None, help="image-loading processes per training")
    parser.add_argument("--processed", default=None, help="data folder (default: config paths.processed)")
    parser.add_argument("--runs-dir", default=None, help="where the learning-curve folder goes (default: runs/)")
    parser.add_argument("--max-batches", type=int, default=None, help="only N batches per epoch (quick tests)")
    return parser.parse_args(argv)


def fraction_tag(fraction: float) -> str:
    """0.25 -> 'frac25' (used for folder and log names)."""
    return f"frac{round(fraction * 100)}"


def assign_to_devices(fractions: list[float], devices: list[str]) -> dict[str, list[float]]:
    """Share the trainings between the GPUs so they finish at about the same time.

    Training time grows with the amount of data, so we hand out the biggest
    fractions first, each to the GPU with the least work so far.
    Example with 2 GPUs: [1.0] on one, [0.5, 0.25, 0.1] (= 0.85) on the other.
    """
    queues = {device: [] for device in devices}
    for fraction in sorted(fractions, reverse=True):
        lightest = min(devices, key=lambda d: sum(queues[d]))
        queues[lightest].append(fraction)
    return queues


def train_command(args, fraction: float, device: str, runs_dir: Path) -> list[str]:
    """The `python scripts/train.py ...` command for one fraction."""
    command = [sys.executable, str(TRAIN_SCRIPT), "--config", args.config, "--device", device,
               "--data-fraction", str(fraction), "--runs-dir", str(runs_dir)]
    for flag, value in [("--seed", args.seed), ("--workers", args.workers),
                        ("--processed", args.processed), ("--max-batches", args.max_batches)]:
        if value is not None:
            command += [flag, str(value)]
    return command


def run_queue(args, device: str, fractions: list[float], out: Path, failures: list) -> None:
    """Train each fraction in turn on one device (runs in its own thread)."""
    for fraction in fractions:
        tag = fraction_tag(fraction)
        log_path = out / "logs" / f"{tag}.log"
        print(f"[{device}] training on {fraction:.0%} of the data  (log: {log_path})", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            finished = subprocess.run(train_command(args, fraction, device, out / tag),
                                      stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
        if finished.returncode != 0:
            failures.append(tag)
            print(f"[{device}] {tag} FAILED (exit code {finished.returncode}); see {log_path}", flush=True)
        else:
            print(f"[{device}] {tag} done", flush=True)


def train_all(args, out: Path) -> None:
    """Start one thread per GPU; each thread works through its own list of fractions."""
    (out / "logs").mkdir(parents=True, exist_ok=True)
    queues = assign_to_devices(args.fractions, args.devices)
    (out / "plan.json").write_text(json.dumps({"config": args.config, "queues": queues}, indent=2))
    failures: list[str] = []
    threads = [threading.Thread(target=run_queue, args=(args, device, fractions, out, failures))
               for device, fractions in queues.items() if fractions]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()              # wait until every GPU has finished its list
    if failures:
        print(f"Some trainings failed: {', '.join(failures)}. The rest are still scored below.")


def score_all(out: Path, split: str, device: str, processed: Path | None, workers: int | None) -> pd.DataFrame:
    """Evaluate the best checkpoint of every finished run and collect one row per fraction."""
    rows = []
    for tag_dir in sorted(out.glob("frac*")):
        run_dirs = [d for d in tag_dir.iterdir() if (d / "best.pt").exists()]
        if not run_dirs:
            print(f"Skipping {tag_dir.name}: no finished run inside.")
            continue
        run_dir = max(run_dirs)                          # newest, if a fraction was ever re-run
        cfg = load_config(run_dir / "config.yaml")
        data_dir = Path(processed) if processed else REPO_ROOT / cfg["paths"]["processed"]
        fraction = cfg["data"]["fraction"]
        results = evaluate.run_evaluation(run_dir, split, device, processed=data_dir, workers=workers)
        plain = results["no threshold"]
        thresholded = results[f"with threshold {results['threshold']}"]
        best = json.loads((run_dir / "summary.json").read_text())["best"]
        rows.append({
            "fraction": fraction,
            "train_images": len(load_split(data_dir, "train", fraction, cfg["seed"])),
            "best_epoch": best["epoch"],
            "accuracy": plain["accuracy"],
            "average_reward": plain["average_reward"],
            "reward_percent_of_max": plain["reward_percent_of_max"],
            "accuracy_with_threshold": thresholded["accuracy"],
            "reward_percent_of_max_with_threshold": thresholded["reward_percent_of_max"],
            "run": str(run_dir.relative_to(out)),
        })
    return pd.DataFrame(rows).sort_values("fraction").reset_index(drop=True) if rows else pd.DataFrame()


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.from_dir:
        out = Path(args.from_dir)
    else:
        runs_root = Path(args.runs_dir) if args.runs_dir else REPO_ROOT / load_config(args.config)["paths"]["runs"]
        out = runs_root / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_learning_curve"
        out.mkdir(parents=True)
        print(f"Learning curve folder: {out}")
        train_all(args, out)

    table = score_all(out, args.split, args.devices[0], args.processed, args.workers)
    if table.empty:
        raise SystemExit(f"No finished runs found in {out}.")
    table.to_csv(out / f"learning_curve_{args.split}.csv", index=False)
    save_figure(learning_curve_figure(table, args.split), out / f"learning_curve_{args.split}.png")

    print(f"\nLearning curve on '{args.split}':")
    print(table[["fraction", "train_images", "best_epoch", "accuracy", "reward_percent_of_max"]]
          .to_string(index=False, formatters={"fraction": "{:.0%}".format, "accuracy": "{:.2%}".format,
                                              "reward_percent_of_max": "{:.2f}".format}))
    print(f"Saved: {out / f'learning_curve_{args.split}.csv'} and .png")
    return 0


if __name__ == "__main__":   # required on Windows for the image-loading worker processes
    sys.exit(main())
