import numpy as np

from src.segmentation.edge_segmentation import edge_mask
from src.segmentation.kmeans import kmeans_segment, reconstruct_segmented_image


def test_kmeans_shape():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    labels, centers = kmeans_segment(image, clusters=2)
    assert labels.shape == image.shape[:2]
    assert centers.shape == (2, 3)


def test_kmeans_reconstruction_shape():
    labels = np.zeros((16, 16), dtype=np.int32)
    centers = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.float32)
    result = reconstruct_segmented_image(labels, centers)
    assert result.shape == (16, 16, 3)
    assert result.dtype == np.uint8


def test_edge_mask_shape():
    image = np.zeros((32, 32), dtype=np.uint8)
    result = edge_mask(image)
    assert result.shape == image.shape
