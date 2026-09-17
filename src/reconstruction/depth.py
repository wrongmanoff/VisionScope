"""Stereo depth reconstruction utilities."""

from __future__ import annotations

import numpy as np


def disparity_to_depth(
    disparity: np.ndarray,
    focal_length: float,
    baseline: float,
) -> np.ndarray:
    """Compute depth Z = fB/d in the same distance unit as baseline."""
    d = np.asarray(disparity, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError("disparity must be a 2D array.")
    if focal_length <= 0:
        raise ValueError("focal_length must be positive.")
    if baseline <= 0:
        raise ValueError("baseline must be positive.")

    depth = np.zeros_like(d, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    depth[valid] = (focal_length * baseline) / d[valid]
    return depth


def depth_error_metrics(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    """Calculate MAE and RMSE over valid positive ground-truth pixels."""
    est = np.asarray(estimated, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)

    if est.shape != gt.shape:
        raise ValueError("Estimated and ground-truth depth must have the same shape.")

    valid = np.isfinite(est) & np.isfinite(gt) & (gt > 0)
    if not np.any(valid):
        raise ValueError("No valid ground-truth depth pixels are available.")

    error = est[valid] - gt[valid]
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }
