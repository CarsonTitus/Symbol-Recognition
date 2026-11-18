"""
plots.py - pictures and charts (matplotlib).

What this does:
    image_grid()   a grid of images with a caption under each, e.g. "true: triangle"
                   or "true: triangle / pred: unknown_symbol (71%)". Used to show
                   what the model is training on, and which images it gets wrong.

Why:
    Seeing the actual images is the quickest way to spot problems (bad labels,
    augmentation that goes too far, a class the model keeps confusing).
    More charts (confusion matrix, reward convergence) will be added here for evaluate.py.
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
