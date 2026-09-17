"""Pinhole camera geometry utilities."""

from __future__ import annotations

import numpy as np


def create_camera_matrix(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Create the intrinsic camera matrix K."""
    if fx <= 0 or fy <= 0:
        raise ValueError("Focal lengths fx and fy must be positive.")
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def projection_matrix(
    K: np.ndarray,
    R: np.ndarray | None = None,
    t: np.ndarray | None = None,
) -> np.ndarray:
    """Construct P = K[R|t] for a pinhole camera."""
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError("K must have shape (3, 3).")

    R = np.eye(3, dtype=np.float64) if R is None else np.asarray(R, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64) if t is None else np.asarray(t, dtype=np.float64).reshape(-1)

    if R.shape != (3, 3):
        raise ValueError("R must have shape (3, 3).")
    if t.shape != (3,):
        raise ValueError("t must have shape (3,).")

    return K @ np.column_stack([R, t])


def project_3d_points(
    points_3d: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """Project 3D points to image coordinates using P."""
    points = np.asarray(points_3d, dtype=np.float64)
    matrix = np.asarray(P, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_3d must have shape (N, 3).")
    if matrix.shape != (3, 4):
        raise ValueError("P must have shape (3, 4).")

    homogeneous = np.column_stack([points, np.ones(len(points))])
    projected_h = (matrix @ homogeneous.T).T

    if np.any(np.abs(projected_h[:, 2]) < 1e-12):
        raise ValueError("A 3D point projects onto the camera plane.")

    return projected_h[:, :2] / projected_h[:, 2, None]
