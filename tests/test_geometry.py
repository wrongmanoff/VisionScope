import numpy as np
import pytest

from src.geometry.camera import create_camera_matrix, projection_matrix, project_3d_points
from src.geometry.homography import estimate_homography_dlt, project_points, reprojection_errors
from src.geometry.ransac import ransac_homography


def _synthetic_correspondences():
    src = np.array(
        [[0, 0], [100, 0], [100, 80], [0, 80], [30, 20], [70, 60], [10, 50], [90, 10]],
        dtype=np.float64,
    )
    H_true = np.array(
        [[1.1, 0.08, 20.0], [0.03, 1.05, 15.0], [0.0004, 0.0002, 1.0]],
        dtype=np.float64,
    )
    dst = project_points(src, H_true)
    return src, dst, H_true


def test_dlt_recovers_homography():
    src, dst, H_true = _synthetic_correspondences()
    H = estimate_homography_dlt(src, dst)
    projected = project_points(src, H)
    np.testing.assert_allclose(projected, dst, atol=1e-6)
    np.testing.assert_allclose(H / H[2, 2], H_true, atol=1e-6)


def test_ransac_rejects_outliers():
    src, dst, _ = _synthetic_correspondences()
    rng = np.random.default_rng(7)
    outlier_src = rng.uniform(0, 100, size=(12, 2))
    outlier_dst = rng.uniform(0, 200, size=(12, 2))

    src_all = np.vstack([src, outlier_src])
    dst_all = np.vstack([dst, outlier_dst])

    H, mask, _ = ransac_homography(
        src_all, dst_all, threshold=1.0, iterations=500, seed=3
    )

    assert mask.shape == (len(src_all),)
    assert int(mask[: len(src)].sum()) >= 7
    assert int(mask[len(src) :].sum()) <= 2
    assert np.median(reprojection_errors(src, dst, H)) < 1.0


def test_camera_projection():
    K = create_camera_matrix(800, 800, 320, 240)
    P = projection_matrix(K)
    points = np.array([[0, 0, 5], [1, 0, 5], [0, 1, 5]], dtype=float)
    projected = project_3d_points(points, P)
    np.testing.assert_allclose(projected, [[320, 240], [480, 240], [320, 400]])


def test_homography_rejects_too_few_points():
    with pytest.raises(ValueError):
        estimate_homography_dlt(
            np.zeros((3, 2)),
            np.zeros((3, 2)),
        )
