"""Benchmark selected VisionScope algorithms on reproducible synthetic inputs."""

from __future__ import annotations

import time
import numpy as np
import cv2


def make_shift_pair(
    height: int = 160,
    width: int = 220,
    shift: int = 6,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic textured stereo-like pair with known shift."""
    if height < 40 or width < 40:
        raise ValueError("height and width must be at least 40.")
    if shift < 1 or shift >= width // 3:
        raise ValueError("shift must be positive and smaller than one third of width.")

    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    cv2.rectangle(base, (25, 35), (75, 105), 220, -1)
    cv2.circle(base, (145, 80), 28, 60, -1)
    cv2.line(base, (20, 130), (190, 125), 180, 3)

    left = base
    right = np.zeros_like(base)
    right[:, :-shift] = left[:, shift:]
    right[:, -shift:] = left[:, -1:]
    return left, right


def time_callable(func, *args, repeats: int = 3, **kwargs):
    """Time a callable repeatedly and return median runtime in seconds."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1.")
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        times.append(time.perf_counter() - start)
    return result, float(np.median(times))


def disparity_quality(disparity: np.ndarray, expected: float) -> dict[str, float]:
    """Compare valid disparity pixels against a known synthetic shift."""
    d = np.asarray(disparity, dtype=float)
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return {"valid_ratio": 0.0, "mae": float("inf"), "rmse": float("inf")}
    err = d[valid] - expected
    return {
        "valid_ratio": float(np.mean(valid)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }
