"""RANSAC-based robust homography estimation."""

from __future__ import annotations

import numpy as np

from src.geometry.homography import estimate_homography_dlt, reprojection_errors


def ransac_homography(
    points_src: np.ndarray,
    points_dst: np.ndarray,
    threshold: float = 3.0,
    iterations: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Estimate a homography robustly and return H, inlier mask, and iterations.

    The minimal sample contains four correspondences. The returned mask marks
    correspondences whose reprojection error is <= threshold.
    """
    src = np.asarray(points_src, dtype=np.float64)
    dst = np.asarray(points_dst, dtype=np.float64)

    if src.ndim != 2 or src.shape[1] != 2 or dst.shape != src.shape:
        raise ValueError("Point arrays must both have shape (N, 2).")
    if len(src) < 4:
        raise ValueError("RANSAC homography requires at least four correspondences.")
    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    if iterations <= 0:
        raise ValueError("iterations must be positive.")

    rng = np.random.default_rng(seed)
    best_H = None
    best_mask = None
    best_count = -1
    best_error = np.inf

    for _ in range(iterations):
        indices = rng.choice(len(src), size=4, replace=False)

        try:
            H = estimate_homography_dlt(src[indices], dst[indices])
            errors = reprojection_errors(src, dst, H)
        except ValueError:
            continue

        mask = errors <= threshold
        count = int(mask.sum())
        total_error = float(errors[mask].sum()) if count else np.inf

        if count > best_count or (count == best_count and total_error < best_error):
            best_H = H
            best_mask = mask
            best_count = count
            best_error = total_error

    if best_H is None or best_mask is None or best_count < 4:
        raise ValueError("RANSAC could not find a valid homography with sufficient inliers.")

    # Refit using all inliers from the best consensus set.
    refined_H = estimate_homography_dlt(src[best_mask], dst[best_mask])
    refined_errors = reprojection_errors(src, dst, refined_H)
    refined_mask = refined_errors <= threshold

    if int(refined_mask.sum()) >= 4:
        refined_H = estimate_homography_dlt(src[refined_mask], dst[refined_mask])
        refined_errors = reprojection_errors(src, dst, refined_H)
        refined_mask = refined_errors <= threshold

    return refined_H, refined_mask, iterations
