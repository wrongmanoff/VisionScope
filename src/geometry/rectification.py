"""Stereo image rectification utilities."""

from __future__ import annotations

import cv2
import numpy as np


def stereo_rectify(
    K1: np.ndarray,
    D1: np.ndarray,
    K2: np.ndarray,
    D2: np.ndarray,
    image_size: tuple[int, int],
    R: np.ndarray,
    T: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute stereo rectification transforms with OpenCV.

    Returns R1, R2, P1, P2, Q. Calibration is required; this function does not
    invent camera parameters.
    """
    K1 = np.asarray(K1, dtype=np.float64)
    K2 = np.asarray(K2, dtype=np.float64)
    D1 = np.asarray(D1, dtype=np.float64)
    D2 = np.asarray(D2, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(-1)

    if K1.shape != (3, 3) or K2.shape != (3, 3):
        raise ValueError("K1 and K2 must have shape (3, 3).")
    if R.shape != (3, 3) or T.shape != (3,):
        raise ValueError("R must be (3, 3) and T must contain 3 values.")
    if len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError("image_size must be (width, height), both positive.")

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, tuple(map(int, image_size)), R, T
    )
    return R1, R2, P1, P2, Q
