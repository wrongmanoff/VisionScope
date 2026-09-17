import numpy as np
import pytest

from src.preprocessing.convolution import convolve2d, gaussian_kernel
from src.preprocessing.histogram import equalize_histogram
from src.preprocessing.fourier import fft_spectrum


def test_convolution_identity():
    image = np.arange(9, dtype=np.uint8).reshape(3, 3)
    kernel = np.array([[1]], dtype=float)
    np.testing.assert_allclose(convolve2d(image, kernel), image)


def test_gaussian_kernel_normalized():
    kernel = gaussian_kernel(5, 1.0)
    assert kernel.shape == (5, 5)
    np.testing.assert_allclose(kernel.sum(), 1.0)


def test_equalization_preserves_shape():
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    result = equalize_histogram(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_fourier_shape():
    image = np.zeros((32, 24), dtype=np.uint8)
    result = fft_spectrum(image)
    assert result.shape == image.shape


def test_even_kernel_rejected():
    with pytest.raises(ValueError):
        gaussian_kernel(4, 1.0)
