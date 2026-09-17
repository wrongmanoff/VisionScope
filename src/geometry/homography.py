"""Homography estimation using the normalized Direct Linear Transform (DLT)."""

from __future__ import annotations

import numpy as np


def _as_points(points: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2).")
    if arr.shape[0] < 4:
        raise ValueError("At least four point correspondences are required.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_distance = np.mean(np.linalg.norm(shifted, axis=1))

    if mean_distance < 1e-12:
        raise ValueError("Point set is degenerate: all points are identical.")

    scale = np.sqrt(2.0) / mean_distance
    T = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    homogeneous = np.column_stack([points, np.ones(len(points))])
    normalized = (T @ homogeneous.T).T
    return normalized[:, :2], T


def estimate_homography_dlt(
    points_src: np.ndarray,
    points_dst: np.ndarray,
) -> np.ndarray:
    """Estimate H such that x_dst ~ H x_src using normalized DLT."""
    src = _as_points(points_src, "points_src")
    dst = _as_points(points_dst, "points_dst")

    if src.shape != dst.shape:
        raise ValueError("Source and destination point arrays must have the same shape.")

    src_n, T_src = _normalize_points(src)
    dst_n, T_dst = _normalize_points(dst)

    A_rows = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        A_rows.extend(
            [
                [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u],
                [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v],
            ]
        )

    A = np.asarray(A_rows, dtype=np.float64)
    if np.linalg.matrix_rank(A) < 8:
        raise ValueError("Point correspondences are degenerate for homography estimation.")

    _, _, Vt = np.linalg.svd(A)
    H_normalized = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(T_dst) @ H_normalized @ T_src
    if abs(H[2, 2]) < 1e-12:
        H /= np.linalg.norm(H)
    else:
        H /= H[2, 2]

    return H


def project_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Project 2D points through a homography."""
    pts = _as_points(points, "points")
    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("H must have shape (3, 3).")

    homogeneous = np.column_stack([pts, np.ones(len(pts))])
    projected_h = (matrix @ homogeneous.T).T
    scale = projected_h[:, 2]

    if np.any(np.abs(scale) < 1e-12):
        raise ValueError("Homography maps a point to infinity.")

    return projected_h[:, :2] / scale[:, None]


def reprojection_errors(
    points_src: np.ndarray,
    points_dst: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """Return Euclidean reprojection error for each correspondence."""
    src = _as_points(points_src, "points_src")
    dst = _as_points(points_dst, "points_dst")
    if src.shape != dst.shape:
        raise ValueError("Source and destination point arrays must have the same shape.")

    projected = project_points(src, H)
    return np.linalg.norm(projected - dst, axis=1)
