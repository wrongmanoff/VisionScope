"""Harris corner detection."""

import cv2
import numpy as np


def harris_corners(
    image: np.ndarray,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    threshold_ratio: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Harris response and a binary corner mask."""
    if image.ndim != 2:
        raise ValueError("harris_corners expects a grayscale image.")
    if image.dtype != np.uint8:
        raise ValueError("harris_corners expects an 8-bit image.")
    if block_size <= 0 or ksize <= 0 or ksize % 2 == 0:
        raise ValueError("block_size must be positive and ksize must be odd.")
    if not 0 < k < 0.25:
        raise ValueError("k must be between 0 and 0.25.")
    if not 0 < threshold_ratio <= 1:
        raise ValueError("threshold_ratio must be in (0, 1].")

    response = cv2.cornerHarris(
        np.float32(image), block_size, ksize, k
    )
    threshold = response.max() * threshold_ratio
    mask = (response > threshold).astype(np.uint8) * 255
    return response, mask


def detect_harris(
    image: np.ndarray,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    threshold: float = 0.01,
) -> np.ndarray:
    """Compatibility wrapper for the public Harris detector API.

    Returns a binary mask with the same height and width as the input image.
    """
    _, mask = harris_corners(
        image,
        block_size=block_size,
        ksize=ksize,
        k=k,
        threshold_ratio=threshold,
    )
    return mask
