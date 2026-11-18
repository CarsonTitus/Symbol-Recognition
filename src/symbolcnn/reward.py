"""
reward.py - score the model's predictions with the submitted reward rubric.

What this does:
    The CS471 scope (CLAUDE.md §2) promises a reward for every image:
        correct "no symbol"                      -> +1
        "symbol detected but cannot identify"    -> +0.5
        "symbol detected: triangle" (correct)    -> +1
        anything incorrect                       -> +0
    These numbers live in a 3x3 "reward matrix" in configs/default.yaml:
    row = the TRUE label, column = the model's PREDICTED label.

    This file turns lists of true and predicted labels into:
        - the reward for each image
        - the average reward per image (plotted per epoch = reward convergence graph)
        - the best average reward possible for that same set of images
        - the average as a percentage of that best possible value

Why "percent of max":
    A true unknown_symbol can earn at most 0.5, so a perfect model scores below
    1.0 whenever unknown symbols are present. Percent-of-max makes runs on
    different datasets comparable (CLAUDE.md §2 notes).

Labels are class INDEXES (0, 1, 2) in the order of config "classes":
    0 = no_symbol, 1 = unknown_symbol, 2 = triangle
Only NumPy is used, so the unit tests run fast without PyTorch.
"""
from __future__ import annotations

import numpy as np


def reward_matrix_from_config(cfg: dict) -> np.ndarray:
    """Read the reward matrix from a loaded config as a NumPy array of shape (classes, classes)."""
    matrix = np.asarray(cfg["reward_matrix"], dtype=float)
    n = len(cfg["classes"])
    if matrix.shape != (n, n):
        raise ValueError(f"reward matrix must be {n}x{n}, got {matrix.shape}")
    return matrix


def per_image_rewards(y_true, y_pred, matrix: np.ndarray) -> np.ndarray:
    """Reward earned by each image: look up matrix[true label, predicted label]."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    return matrix[y_true, y_pred]


def average_reward(y_true, y_pred, matrix: np.ndarray) -> float:
    """Average reward per image (0.0 if there are no images)."""
    rewards = per_image_rewards(y_true, y_pred, matrix)
    return float(rewards.mean()) if rewards.size else 0.0


def max_possible_average_reward(y_true, matrix: np.ndarray) -> float:
    """Best average reward any model could get on these images.

    For each image, the best it can earn is the largest value in its TRUE label's
    row of the matrix (e.g. 1.0 for a triangle, 0.5 for an unknown symbol).
    """
    y_true = np.asarray(y_true, dtype=int)
    if y_true.size == 0:
        return 0.0
    best_per_class = matrix.max(axis=1)        # best reward for each true class
    return float(best_per_class[y_true].mean())


def reward_percent_of_max(y_true, y_pred, matrix: np.ndarray) -> float:
    """Average reward as a percentage (0-100) of the best possible average reward."""
    best = max_possible_average_reward(y_true, matrix)
    if best == 0.0:
        return 0.0
    return 100.0 * average_reward(y_true, y_pred, matrix) / best


def apply_confidence_threshold(probabilities, threshold: float, unknown_index: int) -> np.ndarray:
    """Turn class probabilities into predictions, with the safety-net rule from CLAUDE.md §3.

    probabilities: array of shape (images, classes); each row sums to 1.
    Normally the prediction is the most likely class. If that class's probability
    is below `threshold`, predict unknown_symbol instead of guessing.
    Use threshold=0.0 for "no threshold" results.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = probabilities.argmax(axis=1)
    top_confidence = probabilities.max(axis=1)
    predictions[top_confidence < threshold] = unknown_index
    return predictions
