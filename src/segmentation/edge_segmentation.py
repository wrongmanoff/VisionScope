"""Edge-based segmentation utilities."""

import cv2
import numpy as np


def edge_mask(image: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """Return a binary edge mask using Canny."""
    if image.ndim != 2:
        raise ValueError("edge_mask expects a grayscale image.")
    if not 0 <= low < high:
        raise ValueError("Thresholds must satisfy 0 <= low < high.")
    return cv2.Canny(image, low, high)
