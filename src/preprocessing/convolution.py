"""Convolution implemented with NumPy primitives."""

import numpy as np


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a 2D kernel using zero padding and return a float array."""
    if image.ndim != 2:
        raise ValueError("convolve2d expects a 2D grayscale image.")
    if kernel.ndim != 2:
        raise ValueError("Kernel must be 2D.")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Kernel dimensions must be odd.")

    image = image.astype(np.float64, copy=False)
    kernel = np.flip(kernel.astype(np.float64, copy=False))

    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")

    output = np.empty_like(image, dtype=np.float64)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded[y:y + kernel.shape[0], x:x + kernel.shape[1]]
            output[y, x] = np.sum(region * kernel)

    return output


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Create a normalized 2D Gaussian kernel."""
    if size <= 0 or size % 2 == 0:
        raise ValueError("Gaussian kernel size must be a positive odd integer.")
    if sigma <= 0:
        raise ValueError("Sigma must be greater than zero.")

    radius = size // 2
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel
