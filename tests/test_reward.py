"""
test_reward.py - prove the reward math matches the submitted rubric (CLAUDE.md §2).

Run with:  pytest
Every expected number here was worked out by hand, so the team can check them on paper.
Label indexes: 0 = no_symbol, 1 = unknown_symbol, 2 = triangle.
"""
from pathlib import Path

import numpy as np
import pytest

from symbolcnn.config import load_config
from symbolcnn.reward import (
    apply_confidence_threshold,
    average_reward,
    max_possible_average_reward,
    per_image_rewards,
    reward_matrix_from_config,
    reward_percent_of_max,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NO, UNK, TRI = 0, 1, 2


@pytest.fixture
def matrix():
    """The reward matrix exactly as stored in configs/default.yaml."""
    return reward_matrix_from_config(load_config(REPO_ROOT / "configs" / "default.yaml"))


def test_matrix_matches_submitted_rubric(matrix):
    # rows = true label, columns = predicted label
    expected = np.array([
        [1.0, 0.0, 0.0],   # true no_symbol
        [0.0, 0.5, 0.0],   # true unknown_symbol
        [0.0, 0.5, 1.0],   # true triangle
    ])
    assert np.array_equal(matrix, expected)


def test_every_rubric_case(matrix):
    cases = [
        (NO, NO, 1.0),    # correctly says no symbol           -> +1
        (TRI, TRI, 1.0),  # detected and identified            -> +1
        (TRI, UNK, 0.5),  # detected but cannot identify       -> +0.5
        (UNK, UNK, 0.5),  # detected but cannot identify       -> +0.5
        (NO, TRI, 0.0),   # false alarm                        -> 0
        (NO, UNK, 0.0),   # false alarm                        -> 0
        (TRI, NO, 0.0),   # missed a triangle                  -> 0
        (UNK, NO, 0.0),   # missed a symbol                    -> 0
        (UNK, TRI, 0.0),  # called a non-triangle a triangle   -> 0
    ]
    for true, pred, expected in cases:
        assert per_image_rewards([true], [pred], matrix)[0] == expected, (true, pred)


def test_average_and_percent_of_max(matrix):
    y_true = [NO, UNK, TRI, TRI]
    y_pred = [NO, UNK, UNK, NO]
    # rewards: 1.0 + 0.5 + 0.5 + 0.0 = 2.0  ->  average 2.0 / 4 = 0.5
    assert average_reward(y_true, y_pred, matrix) == pytest.approx(0.5)
    # best possible: 1.0 + 0.5 + 1.0 + 1.0 = 3.5  ->  3.5 / 4 = 0.875
    assert max_possible_average_reward(y_true, matrix) == pytest.approx(0.875)
    # 0.5 / 0.875 = 57.142857...%
    assert reward_percent_of_max(y_true, y_pred, matrix) == pytest.approx(100 * 0.5 / 0.875)


def test_perfect_model_scores_100_percent_even_with_unknowns(matrix):
    y_true = [NO, UNK, UNK, TRI]
    # average is (1 + 0.5 + 0.5 + 1) / 4 = 0.75, below 1.0 because of the unknowns...
    assert average_reward(y_true, y_true, matrix) == pytest.approx(0.75)
    # ...but it is still 100% of the best possible score
    assert reward_percent_of_max(y_true, y_true, matrix) == pytest.approx(100.0)


def test_empty_input_does_not_crash(matrix):
    assert average_reward([], [], matrix) == 0.0
    assert reward_percent_of_max([], [], matrix) == 0.0


def test_mismatched_lengths_raise(matrix):
    with pytest.raises(ValueError):
        per_image_rewards([NO, TRI], [NO], matrix)


def test_confidence_threshold_falls_back_to_unknown():
    probs = np.array([
        [0.05, 0.05, 0.90],  # confident triangle          -> triangle
        [0.30, 0.20, 0.50],  # top is 0.50 < 0.6, unsure    -> unknown_symbol
        [0.70, 0.10, 0.20],  # confident no_symbol          -> no_symbol
        [0.40, 0.00, 0.60],  # exactly 0.6 is NOT below     -> triangle
    ])
    preds = apply_confidence_threshold(probs, threshold=0.6, unknown_index=UNK)
    assert preds.tolist() == [TRI, UNK, NO, TRI]


def test_threshold_zero_means_plain_argmax():
    probs = np.array([[0.34, 0.33, 0.33], [0.1, 0.2, 0.7]])
    preds = apply_confidence_threshold(probs, threshold=0.0, unknown_index=UNK)
    assert preds.tolist() == [NO, TRI]
