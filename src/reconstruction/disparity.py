"""Stereo disparity estimation with classical block matching."""

from __future__ import annotations

import cv2
import numpy as np


def _validate_stereo(left: np.ndarray, right: np.ndarray) -> None:
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("Stereo block matching expects grayscale images.")
    if left.shape != right.shape:
        raise ValueError("Left and right images must have identical dimensions.")


def compute_disparity_sad(
    left: np.ndarray,
    right: np.ndarray,
    max_disparity: int = 64,
    block_size: int = 9,
) -> np.ndarray:
    """Estimate left-image disparity using SAD block matching.

    For every valid pixel, candidate horizontal disparities are evaluated using
    the sum of absolute differences over a square window. A disparity d means
    the corresponding point in the right image is approximately x-d.
    """
    _validate_stereo(left, right)

    if max_disparity <= 0:
        raise ValueError("max_disparity must be positive.")
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd integer >= 3.")
    if max_disparity >= left.shape[1]:
        raise ValueError("max_disparity must be smaller than image width.")

    left_f = left.astype(np.float32)
    right_f = right.astype(np.float32)
    half = block_size // 2

    disparity = np.zeros(left.shape, dtype=np.float32)
    best_cost = np.full(left.shape, np.inf, dtype=np.float32)

    # Keep a one-pixel-valid border and a right-image search margin.
    y_start, y_end = half, left.shape[0] - half
    x_start, x_end = max(half, max_disparity + half), left.shape[1] - half

    for d in range(max_disparity):
        if d == 0:
            shifted = right_f
        else:
            shifted = np.zeros_like(right_f)
            shifted[:, d:] = right_f[:, :-d]

        diff = np.abs(left_f - shifted)

        # Box filter computes the windowed SAD efficiently.
        cost = cv2.boxFilter(
            diff,
            ddepth=-1,
            ksize=(block_size, block_size),
            normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        )

        valid = np.zeros_like(cost, dtype=bool)
        valid[y_start:y_end, x_start:x_end] = True
        improved = valid & (cost < best_cost)

        disparity[improved] = float(d)
        best_cost[improved] = cost[improved]

    # Pixels for which no valid comparison existed remain zero.
    return disparity


def compute_disparity_sgbm(
    left: np.ndarray,
    right: np.ndarray,
    min_disparity: int = 0,
    num_disparities: int = 64,
    block_size: int = 5,
) -> np.ndarray:
    """Optional OpenCV SGBM baseline for algorithm comparison."""
    _validate_stereo(left, right)

    if num_disparities <= 0 or num_disparities % 16 != 0:
        raise ValueError("num_disparities must be a positive multiple of 16.")
    if block_size <= 0 or block_size % 2 == 0:
        raise ValueError("block_size must be a positive odd integer.")

    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher.compute(left, right).astype(np.float32) / 16.0


def normalize_disparity(disparity: np.ndarray) -> np.ndarray:
    """Normalize disparity to an 8-bit image for visualization."""
    d = np.asarray(disparity, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError("disparity must be a 2D array.")
    valid = np.isfinite(d) & (d > 0)
    out = np.zeros(d.shape, dtype=np.uint8)
    if np.any(valid):
        lo, hi = float(d[valid].min()), float(d[valid].max())
        if hi > lo:
            out[valid] = np.clip(
                (d[valid] - lo) * 255.0 / (hi - lo), 0, 255
            ).astype(np.uint8)
        else:
            out[valid] = 255
    return out
