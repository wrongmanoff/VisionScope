"""Image filtering and edge preprocessing."""

import cv2
import numpy as np

from .convolution import convolve2d, gaussian_kernel


def gaussian_filter(
    image: np.ndarray, size: int = 5, sigma: float = 1.0
) -> np.ndarray:
    """Apply a Gaussian filter using the custom convolution implementation."""
    if image.ndim != 2:
        raise ValueError("gaussian_filter expects a grayscale image.")

    result = convolve2d(image, gaussian_kernel(size, sigma))
    return np.clip(result, 0, 255).astype(np.uint8)


def canny_edges(image: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """Compute Canny edges."""
    if image.ndim != 2:
        raise ValueError("canny_edges expects a grayscale image.")
    if not (0 <= low < high):
        raise ValueError("Canny thresholds must satisfy 0 <= low < high.")
    return cv2.Canny(image, low, high)


def log_edges(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Approximate Laplacian-of-Gaussian edges."""
    if sigma <= 0:
        raise ValueError("Sigma must be greater than zero.")
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    return np.uint8(np.clip(np.abs(laplacian), 0, 255))


def dog_edges(
    image: np.ndarray, sigma1: float = 1.0, sigma2: float = 2.0
) -> np.ndarray:
    """Difference-of-Gaussians response."""
    if sigma1 <= 0 or sigma2 <= 0 or sigma1 == sigma2:
        raise ValueError("sigma1 and sigma2 must be positive and different.")

    first = cv2.GaussianBlur(image, (0, 0), sigma1).astype(np.float64)
    second = cv2.GaussianBlur(image, (0, 0), sigma2).astype(np.float64)
    response = np.abs(first - second)
    return np.uint8(np.clip(response, 0, 255))
