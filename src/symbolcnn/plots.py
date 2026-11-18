"""
plots.py - pictures and charts (matplotlib).

What this does:
    image_grid()   a grid of images with a caption under each, e.g. "true: triangle"
                   or "true: triangle / pred: unknown_symbol (71%)". Used to show
                   what the model is training on, and which images it gets wrong.

Why:
    Seeing the actual images is the quickest way to spot problems (bad labels,
    augmentation that goes too far, a class the model keeps confusing).
    confusion_matrix_figure()    true vs predicted grid with counts and row percentages
    reward_convergence_figure()  average validation reward per epoch (CLAUDE.md §2)
    training_curves_figure()     loss and accuracy per epoch, train vs validation
    learning_curve_figure()      accuracy and reward vs amount of training data
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")   # draw to files, no window needed (works on servers and over SSH)
import matplotlib.pyplot as plt  # noqa: E402  (must come after choosing the backend)
import numpy as np  # noqa: E402


def image_grid(images: np.ndarray, captions: list[str], columns: int = 8, title: str = ""):
    """images: (N, H, W, 3) uint8. Returns a matplotlib Figure (save it or log it to TensorBoard)."""
    count = len(images)
    rows = max(1, int(np.ceil(count / columns)))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 1.9, rows * 2.1), squeeze=False)
    for i, axis in enumerate(axes.flat):
        axis.axis("off")
        if i < count:
            axis.imshow(images[i])
            axis.set_title(captions[i], fontsize=7)
    if title:
        figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    return figure


def save_figure(figure, path) -> None:
    figure.savefig(path, dpi=110)
    plt.close(figure)


def confusion_matrix_figure(counts: np.ndarray, percents: np.ndarray, classes: list[str], title: str):
    """Confusion matrix picture: each cell shows the count and the row percentage.

    Rows = true label, columns = predicted label. Darker = larger share of that row,
    so a good model has a dark diagonal from top-left to bottom-right.
    """
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(percents, cmap="Blues", vmin=0, vmax=100)
    figure.colorbar(image, ax=axis, label="% of the true class (row)")
    axis.set_xticks(range(len(classes)), classes, rotation=20)
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    for row in range(len(classes)):
        for col in range(len(classes)):
            color = "white" if percents[row, col] > 60 else "black"   # readable on dark cells
            axis.text(col, row, f"{counts[row, col]}\n({percents[row, col]:.1f}%)",
                      ha="center", va="center", color=color, fontsize=10)
    axis.set_title(title, fontsize=10)
    figure.tight_layout()
    return figure


def reward_convergence_figure(epochs, rewards, max_reward: float, phases=None, title: str = ""):
    """Average validation reward per epoch, with a dashed line at the best possible reward.

    phases: optional list (one per epoch) like "frozen"/"finetune". A dotted vertical
    line is drawn wherever the phase changes (e.g. backbone unfrozen, or Stage A -> Stage B).
    """
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, rewards, marker="o", label="average validation reward")
    axis.axhline(max_reward, linestyle="--", color="gray", label=f"best possible ({max_reward:.3f})")
    if phases is not None:
        for i in range(1, len(phases)):
            if phases[i] != phases[i - 1]:
                boundary = (epochs[i - 1] + epochs[i]) / 2
                axis.axvline(boundary, linestyle=":", color="black")
                axis.text(boundary, axis.get_ylim()[0], f" {phases[i]}", va="bottom", fontsize=8)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Average reward per image")
    axis.set_title(title or "Reward convergence", fontsize=10)
    axis.legend(loc="lower right")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def training_curves_figure(epochs, train_loss, val_loss, train_acc, val_acc):
    """Loss and accuracy per epoch, train vs validation, side by side.

    A growing gap (train keeps improving, validation doesn't) is the sign of
    overfitting: the model memorizing training images.
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.plot(epochs, train_loss, marker="o", label="train")
    left.plot(epochs, val_loss, marker="o", label="validation")
    left.set_title("Loss (lower is better)", fontsize=10)
    right.plot(epochs, np.asarray(train_acc) * 100, marker="o", label="train")
    right.plot(epochs, np.asarray(val_acc) * 100, marker="o", label="validation")
    right.set_title("Accuracy % (higher is better)", fontsize=10)
    for axis in (left, right):
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def learning_curve_figure(table, split: str):
    """Accuracy and reward % of max against the number of training images (CLAUDE.md §8 item 6).

    table: one row per fraction with columns fraction, train_images, accuracy,
    reward_percent_of_max. Each point is labeled with its fraction (10%, 25%, ...).
    A curve that is still rising at the right means more data would still help.
    """
    figure, axis = plt.subplots(figsize=(7, 4))
    x = table["train_images"]
    axis.plot(x, table["accuracy"] * 100, marker="o", label="accuracy %")
    axis.plot(x, table["reward_percent_of_max"], marker="s", label="reward % of max")
    for images, accuracy, fraction in zip(x, table["accuracy"] * 100, table["fraction"]):
        axis.annotate(f"{fraction:.0%}", (images, accuracy), textcoords="offset points", xytext=(0, 8),
                      ha="center", fontsize=8)
    axis.axhline(90, linestyle="--", color="gray", label="90% target")
    axis.set_xscale("log")      # 10% -> 100% is a 10x jump, so a log scale spaces the points evenly
    axis.set_xlabel("Training images (log scale)")
    axis.set_ylabel("%")
    axis.set_title(f"Learning curve (scored on {split})", fontsize=10)
    axis.legend(loc="lower right")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure
