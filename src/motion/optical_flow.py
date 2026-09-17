"""Classical optical-flow analysis.

The implementation exposes Farneback dense optical flow as a library
baseline and provides NumPy post-processing for interpretable motion
statistics.
"""

import cv2
import numpy as np


def _validate_pair(prev: np.ndarray, curr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if prev is None or curr is None:
        raise ValueError("Both frames are required.")
    if prev.size == 0 or curr.size == 0:
        raise ValueError("Frames must not be empty.")
    if prev.shape[:2] != curr.shape[:2]:
        raise ValueError("Frames must have identical height and width.")
    if prev.ndim == 3:
        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    if curr.ndim == 3:
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    if prev.ndim != 2 or curr.ndim != 2:
        raise ValueError("Frames must be grayscale or BGR images.")
    return prev.astype(np.uint8), curr.astype(np.uint8)


def compute_dense_optical_flow(
    prev: np.ndarray,
    curr: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
) -> np.ndarray:
    """Compute dense Farneback optical flow as an HxWx2 float array."""
    prev, curr = _validate_pair(prev, curr)
    if not (0.0 < pyr_scale < 1.0):
        raise ValueError("pyr_scale must be between 0 and 1.")
    if levels < 1 or winsize < 3 or iterations < 1:
        raise ValueError("levels, winsize, and iterations must be positive.")
    if winsize % 2 == 0:
        raise ValueError("winsize must be odd.")
    return cv2.calcOpticalFlowFarneback(
        prev, curr, None, pyr_scale, levels, winsize,
        iterations, 5, 1.2, 0
    )


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """Return per-pixel motion magnitude."""
    flow = np.asarray(flow)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError("flow must have shape (H, W, 2).")
    return np.linalg.norm(flow, axis=-1)


def flow_statistics(flow: np.ndarray) -> dict[str, float]:
    """Return interpretable mean/max motion statistics."""
    mag = flow_magnitude(flow)
    return {
        "mean_magnitude": float(np.mean(mag)),
        "max_magnitude": float(np.max(mag)),
        "moving_pixel_ratio": float(np.mean(mag > 1.0)),
    }
