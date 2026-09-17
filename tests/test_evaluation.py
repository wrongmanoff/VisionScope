import numpy as np
import pytest

from src.evaluation.metrics import (
    classification_metrics,
    segmentation_metrics,
    regression_metrics,
)


def test_classification_metrics():
    result = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0])
    assert result["accuracy"] == 0.75
    assert 0 <= result["f1"] <= 1


def test_segmentation_metrics():
    a = np.array([[1, 1], [0, 0]])
    b = np.array([[1, 0], [0, 0]])
    result = segmentation_metrics(a, b)
    assert np.isclose(result["iou"], 0.5)


def test_regression_metrics():
    result = regression_metrics(np.array([1, 2, 3]), np.array([1, 3, 5]))
    assert np.isclose(result["mae"], 1.0)
    assert np.isclose(result["rmse"], np.sqrt(5 / 3))


def test_metric_shape_validation():
    with pytest.raises(ValueError):
        regression_metrics(np.zeros((2,)), np.zeros((3,)))
