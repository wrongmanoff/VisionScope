import numpy as np
import pytest

from src.reconstruction.disparity import (
    compute_disparity_sad,
    normalize_disparity,
)
from src.reconstruction.depth import disparity_to_depth, depth_error_metrics
from src.reconstruction.point_cloud import depth_to_point_cloud


def test_disparity_synthetic_shift():
    # A textured pattern shifted by 4 px gives a measurable disparity.
    rng = np.random.default_rng(10)
    left = (rng.random((64, 100)) * 255).astype(np.uint8)
    right = np.zeros_like(left)
    right[:, :-4] = left[:, 4:]

    disparity = compute_disparity_sad(
        left, right, max_disparity=16, block_size=5
    )
    interior = disparity[10:-10, 20:-10]
    assert np.median(interior[interior > 0]) == pytest.approx(4.0, abs=1.0)


def test_depth_formula():
    disparity = np.full((2, 2), 10.0, dtype=np.float32)
    depth = disparity_to_depth(disparity, focal_length=800, baseline=0.1)
    np.testing.assert_allclose(depth, 8.0)


def test_depth_metrics():
    gt = np.array([[2.0, 4.0], [0.0, 8.0]])
    est = np.array([[3.0, 3.0], [0.0, 10.0]])
    metrics = depth_error_metrics(est, gt)
    assert metrics["mae"] == pytest.approx(4.0 / 3.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.0))


def test_point_cloud_projection():
    depth = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    points = depth_to_point_cloud(depth, focal_length=100, cx=0, cy=0)
    assert points.shape == (4, 3)
    np.testing.assert_allclose(points[0], [0.0, 0.0, 2.0])
    np.testing.assert_allclose(points[1], [0.02, 0.0, 2.0])


def test_invalid_disparity_parameters():
    image = np.zeros((20, 20), dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_disparity_sad(image, image, max_disparity=20)
