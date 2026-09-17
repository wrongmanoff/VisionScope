"""Histogram analysis and equalization."""

import cv2
import numpy as np


def histogram(image: np.ndarray) -> np.ndarray:
    """Return a 256-bin grayscale histogram."""
    if image.ndim != 2:
        raise ValueError("histogram expects a grayscale image.")
    return np.bincount(image.ravel(), minlength=256)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """Perform grayscale histogram equalization using a CDF."""
    if image.ndim != 2:
        raise ValueError("equalize_histogram expects a grayscale image.")
    if image.dtype != np.uint8:
        raise ValueError("equalize_histogram expects an 8-bit image.")

    hist = histogram(image)
    cdf = hist.cumsum()
    nonzero = np.flatnonzero(hist)

    if nonzero.size == 0:
        return image.copy()

    cdf_min = cdf[nonzero[0]]
    denominator = image.size - cdf_min

    if denominator == 0:
        return image.copy()

    lut = np.round((cdf - cdf_min) * 255 / denominator)
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    return lut[image]


def opencv_equalize_histogram(image: np.ndarray) -> np.ndarray:
    """Reference implementation using OpenCV for comparison."""
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("OpenCV equalization expects an 8-bit grayscale image.")
    return cv2.equalizeHist(image)
