"""
metrics.py - the numbers that describe how good the model is (CLAUDE.md §8).

What this does:
    confusion_matrix()     4x4 table: row = TRUE label, column = PREDICTED label,
                           each cell = how many images. The diagonal is the correct answers.
    row_percentages()      the same table, but each row turned into percentages
                           ("of all real triangles, 93% were called triangle").
    per_class_stats()      precision, recall and F1 for every class, plus per-class accuracy.
    novel_shape_summary()  for the never-seen-shapes test: what share was called
                           unknown_symbol (good) vs a named shape (a false alarm).

The three key words, in plain language (for the triangle class):
    precision = when the model SAYS "triangle", how often is it right?   (low = false alarms)
    recall    = of all REAL triangles, how many did it catch?            (low = missed symbols)
    F1        = one number that balances the two (their "harmonic mean")
    For a single class, recall is the same as that class's accuracy.

Labels are class INDEXES in the order of config "classes"
(0 = unknown_symbol, 1 = triangle, 2 = quadrilateral, 3 = ellipse).
Only NumPy is used, so the unit tests run fast without PyTorch.
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    """Count of images for every (true label, predicted label) pair. Shape (classes, classes)."""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    # np.add.at adds 1 to matrix[true, pred] once for every image (repeats are counted).
    np.add.at(matrix, (np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    return matrix


def row_percentages(matrix: np.ndarray) -> np.ndarray:
    """Each row as percentages of that row's total (a row with no images stays 0)."""
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(100.0 * matrix, totals, out=np.zeros(matrix.shape, dtype=float), where=totals > 0)


def _safe_divide(top: float, bottom: float) -> float:
    return float(top / bottom) if bottom else 0.0


def per_class_stats(matrix: np.ndarray, classes: list[str]) -> list[dict]:
    """Precision, recall, F1 and image count for each class, read straight off the confusion matrix.

    For class k:
        true positives  = matrix[k, k]           (really k, called k)
        all called k    = column k summed        (everything the model labeled k)
        all really k    = row k summed           (every image whose true label is k)
    """
    stats = []
    for k, name in enumerate(classes):
        hits = matrix[k, k]
        precision = _safe_divide(hits, matrix[:, k].sum())
        recall = _safe_divide(hits, matrix[k, :].sum())
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        stats.append({"class": name, "images": int(matrix[k, :].sum()),
                      "precision": precision, "recall": recall, "f1": f1})
    return stats


def accuracy(y_true, y_pred) -> float:
    """Share of images whose prediction equals the true label (0.0 if there are none)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if y_true.size else 0.0


def novel_shape_summary(y_pred, classes: list[str]) -> dict:
    """For never-seen shapes (true label is always unknown_symbol): share of each prediction.

    "called_unknown_symbol" is the good outcome. "called_a_named_shape" (triangle,
    quadrilateral or ellipse) is a false alarm an operator would see in the glasses
    (CLAUDE.md §5c).
    """
    y_pred = np.asarray(y_pred, dtype=int)
    total = len(y_pred)
    summary = {f"called_{name}": _safe_divide((y_pred == k).sum(), total) for k, name in enumerate(classes)}
    named = [name for name in classes if name != "unknown_symbol"]
    summary["called_a_named_shape"] = sum(summary[f"called_{name}"] for name in named)
    return summary | {"images": total}
