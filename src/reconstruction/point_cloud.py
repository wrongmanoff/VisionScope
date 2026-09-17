"""Convert a disparity/depth map into a simple stereo point cloud."""

from __future__ import annotations

import numpy as np


def depth_to_point_cloud(
    depth: np.ndarray,
    focal_length: float,
    cx: float,
    cy: float,
    max_depth: float | None = None,
) -> np.ndarray:
    """Return XYZ points using the pinhole equations.

    X = (u-cx)Z/f, Y = (v-cy)Z/f, Z = depth.
    """
    z = np.asarray(depth, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError("depth must be a 2D array.")
    if focal_length <= 0:
        raise ValueError("focal_length must be positive.")

    valid = np.isfinite(z) & (z > 0)
    if max_depth is not None:
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")
        valid &= z <= max_depth

    v, u = np.nonzero(valid)
    zv = z[v, u]
    x = (u.astype(np.float32) - cx) * zv / focal_length
    y = (v.astype(np.float32) - cy) * zv / focal_length

    return np.column_stack((x, y, zv)).astype(np.float32)


def save_point_cloud_ply(points: np.ndarray, path: str) -> None:
    """Save XYZ points as a minimal ASCII PLY file."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
