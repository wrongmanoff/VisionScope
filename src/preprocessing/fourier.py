"""Fourier-transform analysis."""

import numpy as np


def fft_spectrum(image: np.ndarray) -> np.ndarray:
    """Return a log-scaled, centered magnitude spectrum."""
    if image.ndim != 2:
        raise ValueError("fft_spectrum expects a grayscale image.")

    spectrum = np.fft.fftshift(np.fft.fft2(image.astype(np.float64)))
    magnitude = np.abs(spectrum)
    return np.log1p(magnitude)
