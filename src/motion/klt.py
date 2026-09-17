"""KLT feature tracking using the classical Lucas-Kanade formulation."""

import cv2
import numpy as np


def track_features_klt(
    prev: np.ndarray,
    curr: np.ndarray,
    max_corners: int = 200,
    quality_level: float = 0.01,
    min_distance: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Track Shi-Tomasi corners between two frames using Lucas-Kanade.

    Returns:
        previous_points, current_points with shape (N, 2).
    """
    if prev is None or curr is None or prev.size == 0 or curr.size == 0:
        raise ValueError("Both frames are required and must be non-empty.")
    if prev.shape[:2] != curr.shape[:2]:
        raise ValueError("Frames must have identical height and width.")

    if prev.ndim == 3:
        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    if curr.ndim == 3:
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(
        prev, maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance
    )
    if corners is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev, curr, corners, None,
        winSize=(21, 21), maxLevel=3
    )
    if next_pts is None or status is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    good = status.ravel().astype(bool)
    return corners.reshape(-1, 2)[good], next_pts.reshape(-1, 2)[good]
