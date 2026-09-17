import numpy as np
import cv2
import pytest

from src.motion.optical_flow import compute_dense_optical_flow, flow_magnitude
from src.motion.klt import track_features_klt


def synthetic_pair():
    a = np.zeros((100, 120), dtype=np.uint8)
    cv2.rectangle(a, (30, 30), (60, 60), 255, -1)
    b = np.zeros_like(a)
    cv2.rectangle(b, (34, 30), (64, 60), 255, -1)
    return a, b


def test_dense_flow_shape():
    a, b = synthetic_pair()
    flow = compute_dense_optical_flow(a, b)
    assert flow.shape == (100, 120, 2)
    assert np.isfinite(flow).all()


def test_flow_magnitude():
    flow = np.zeros((2, 3, 2), dtype=float)
    flow[..., 0] = 3
    flow[..., 1] = 4
    assert np.allclose(flow_magnitude(flow), 5)


def test_klt_tracks_points():
    a, b = synthetic_pair()
    p0, p1 = track_features_klt(a, b)
    assert p0.ndim == 2 and p1.ndim == 2
    assert p0.shape[1] == 2 and p1.shape[1] == 2
    assert len(p0) == len(p1)


def test_invalid_flow_window():
    a, b = synthetic_pair()
    with pytest.raises(ValueError):
        compute_dense_optical_flow(a, b, winsize=4)
