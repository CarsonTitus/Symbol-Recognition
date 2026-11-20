"""
test_reward.py - prove the reward math matches the submitted rubric (CLAUDE.md §2).

Run with:  pytest
Every expected number here was worked out by hand, so the team can check them on paper.
Label indexes: 0 = unknown_symbol, 1 = triangle, 2 = quadrilateral, 3 = ellipse.
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
UNK, TRI, QUAD, ELL = 0, 1, 2, 3


@pytest.fixture
def matrix():
    """The reward matrix exactly as stored in configs/default.yaml."""
    return reward_matrix_from_config(load_config(REPO_ROOT / "configs" / "default.yaml"))


def test_matrix_matches_submitted_rubric(matrix):
    # rows = true label, columns = predicted label
    expected = np.array([
        # pred: unk  tri  quad ell
        [0.5, 0.0, 0.0, 0.0],   # true unknown_symbol
        [0.5, 1.0, 0.0, 0.0],   # true triangle
        [0.5, 0.0, 1.0, 0.0],   # true quadrilateral
        [0.5, 0.0, 0.0, 1.0],   # true ellipse
    ])
    assert np.array_equal(matrix, expected)


def test_every_rubric_case(matrix):
    cases = [
        (TRI, TRI, 1.0),    # detected and identified          -> +1
        (QUAD, QUAD, 1.0),  # detected and identified          -> +1
        (ELL, ELL, 1.0),    # detected and identified          -> +1
        (TRI, UNK, 0.5),    # detected but cannot identify     -> +0.5
        (QUAD, UNK, 0.5),   # detected but cannot identify     -> +0.5
        (ELL, UNK, 0.5),    # detected but cannot identify     -> +0.5
        (UNK, UNK, 0.5),    # correctly not one of the shapes  -> +0.5
        (UNK, TRI, 0.0),    # false alarm                      -> 0
        (UNK, ELL, 0.0),    # false alarm                      -> 0
        (TRI, ELL, 0.0),    # named the WRONG shape            -> 0
        (QUAD, TRI, 0.0),   # named the WRONG shape            -> 0
    ]
    for true, pred, expected in cases:
        assert per_image_rewards([true], [pred], matrix)[0] == expected, (true, pred)


def test_average_and_percent_of_max(matrix):
    y_true = [UNK, TRI, QUAD, ELL]
    y_pred = [UNK, UNK, QUAD, TRI]
    # rewards: 0.5 + 0.5 + 1.0 + 0.0 = 2.0  ->  average 2.0 / 4 = 0.5
    assert average_reward(y_true, y_pred, matrix) == pytest.approx(0.5)
    # best possible: 0.5 + 1.0 + 1.0 + 1.0 = 3.5  ->  3.5 / 4 = 0.875
    assert max_possible_average_reward(y_true, matrix) == pytest.approx(0.875)
    # 0.5 / 0.875 = 57.142857...%
    assert reward_percent_of_max(y_true, y_pred, matrix) == pytest.approx(100 * 0.5 / 0.875)


def test_perfect_model_scores_100_percent_even_with_unknowns(matrix):
    y_true = [UNK, UNK, TRI, ELL]
    # average is (0.5 + 0.5 + 1 + 1) / 4 = 0.75, below 1.0 because of the unknowns...
    assert average_reward(y_true, y_true, matrix) == pytest.approx(0.75)
    # ...but it is still 100% of the best possible score
    assert reward_percent_of_max(y_true, y_true, matrix) == pytest.approx(100.0)


def test_empty_input_does_not_crash(matrix):
    assert average_reward([], [], matrix) == 0.0
    assert reward_percent_of_max([], [], matrix) == 0.0


def test_mismatched_lengths_raise(matrix):
    with pytest.raises(ValueError):
        per_image_rewards([UNK, TRI], [UNK], matrix)


def test_confidence_threshold_falls_back_to_unknown():
    probs = np.array([
        # unk   tri   quad  ell
        [0.05, 0.90, 0.03, 0.02],  # confident triangle          -> triangle
        [0.20, 0.50, 0.20, 0.10],  # top is 0.50 < 0.6, unsure    -> unknown_symbol
        [0.10, 0.10, 0.10, 0.70],  # confident ellipse           -> ellipse
        [0.40, 0.00, 0.60, 0.00],  # exactly 0.6 is NOT below     -> quadrilateral
    ])
    preds = apply_confidence_threshold(probs, threshold=0.6, unknown_index=UNK)
    assert preds.tolist() == [TRI, UNK, ELL, QUAD]


def test_threshold_zero_means_plain_argmax():
    probs = np.array([[0.26, 0.25, 0.25, 0.24], [0.1, 0.2, 0.0, 0.7]])
    preds = apply_confidence_threshold(probs, threshold=0.0, unknown_index=UNK)
    assert preds.tolist() == [UNK, ELL]
