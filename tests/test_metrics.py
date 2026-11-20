"""
test_metrics.py - check the confusion matrix, precision/recall/F1 and novel-shape numbers by hand.

Run with:  pytest
Labels: 0 = no_symbol, 1 = unknown_symbol, 2 = triangle.
"""
import numpy as np
import pytest

from symbolcnn.metrics import (accuracy, confusion_matrix, novel_shape_summary, per_class_stats,
                               row_percentages)

CLASSES = ["no_symbol", "unknown_symbol", "triangle"]

# 6 images: 2 of each true class. Mistakes: one unknown called triangle, one triangle called unknown.
Y_TRUE = [0, 0, 1, 1, 2, 2]
Y_PRED = [0, 0, 1, 2, 2, 1]


def test_confusion_matrix_counts_true_rows_and_predicted_columns():
    expected = np.array([[2, 0, 0],
                         [0, 1, 1],
                         [0, 1, 1]])
    assert (confusion_matrix(Y_TRUE, Y_PRED, 3) == expected).all()
    assert confusion_matrix(Y_TRUE, Y_PRED, 3).sum() == len(Y_TRUE)   # every image counted once


def test_row_percentages_and_empty_rows():
    percents = row_percentages(confusion_matrix(Y_TRUE, Y_PRED, 3))
    assert percents[0].tolist() == [100.0, 0.0, 0.0]
    assert percents[2].tolist() == [0.0, 50.0, 50.0]
    # A class with no images gives a row of zeros instead of a divide-by-zero error.
    assert row_percentages(confusion_matrix([0], [0], 3))[2].tolist() == [0.0, 0.0, 0.0]


def test_precision_recall_f1():
    stats = {row["class"]: row for row in per_class_stats(confusion_matrix(Y_TRUE, Y_PRED, 3), CLASSES)}
    # Triangle: called "triangle" twice, right once -> precision 0.5; caught 1 of 2 real ones -> recall 0.5.
    assert stats["triangle"]["precision"] == pytest.approx(0.5)
    assert stats["triangle"]["recall"] == pytest.approx(0.5)
    assert stats["triangle"]["f1"] == pytest.approx(0.5)
    assert stats["no_symbol"]["precision"] == stats["no_symbol"]["recall"] == stats["no_symbol"]["f1"] == 1.0
    assert stats["triangle"]["images"] == 2


def test_class_never_predicted_has_zero_precision_not_an_error():
    stats = per_class_stats(confusion_matrix([0, 2], [0, 0], 3), CLASSES)
    assert stats[2]["precision"] == 0.0 and stats[2]["recall"] == 0.0 and stats[2]["f1"] == 0.0


def test_accuracy():
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(4 / 6)
    assert accuracy([], []) == 0.0


def test_novel_shape_summary():
    summary = novel_shape_summary([1, 1, 1, 2], CLASSES)   # 3 called unknown, 1 false alarm
    assert summary["called_unknown_symbol"] == pytest.approx(0.75)
    assert summary["called_triangle"] == pytest.approx(0.25)
    assert summary["called_no_symbol"] == 0.0
    assert summary["called_a_named_shape"] == pytest.approx(0.25)
    assert summary["images"] == 4


def test_novel_shape_false_alarms_count_every_named_shape():
    classes = ["no_symbol", "unknown_symbol", "triangle", "quadrilateral", "ellipse"]
    summary = novel_shape_summary([1, 2, 3, 4], classes)   # 1 good answer, 3 false alarms
    assert summary["called_a_named_shape"] == pytest.approx(0.75)
