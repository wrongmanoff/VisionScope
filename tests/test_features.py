import numpy as np

from src.features.harris import detect_harris
from src.features.sift import detect_sift
from src.features.hog import hog_descriptor


def test_harris_runs():
    image = np.zeros((64, 64), dtype=np.uint8)
    result = detect_harris(image)
    assert result.shape == image.shape


def test_sift_runs():
    image = np.zeros((128, 128), dtype=np.uint8)
    keypoints, descriptors = detect_sift(image)
    assert isinstance(keypoints, list)
    assert descriptors is None or descriptors.shape[1] == 128


def test_hog_is_nonempty():
    image = np.zeros((128, 64), dtype=np.uint8)
    descriptor = hog_descriptor(image)
    assert isinstance(descriptor, np.ndarray)
    assert descriptor.size > 0


def test_hog_rejects_invalid_bins():
    image = np.zeros((128, 64), dtype=np.uint8)
    try:
        hog_descriptor(image, bins=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for bins <= 0")


def test_hog_resizes_nonstandard_input():
    image = np.zeros((100, 80), dtype=np.uint8)
    descriptor = hog_descriptor(image)
    assert descriptor.size > 0


def test_harris_public_api():
    image = np.zeros((64, 64), dtype=np.uint8)
    result = detect_harris(image)
    assert isinstance(result, np.ndarray)
    assert result.shape == image.shape
