"""K-Means image segmentation."""

import cv2
import numpy as np


def kmeans_segment(
    image: np.ndarray,
    clusters: int = 3,
    attempts: int = 10,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment an image into K color/intensity clusters.

    Returns labels reshaped to image spatial dimensions and cluster centers.
    """
    if image.ndim not in (2, 3):
        raise ValueError("kmeans_segment expects a grayscale or color image.")
    if clusters < 2:
        raise ValueError("clusters must be at least 2.")
    if attempts <= 0 or max_iter <= 0:
        raise ValueError("attempts and max_iter must be positive.")

    if image.ndim == 2:
        samples = image.reshape(-1, 1).astype(np.float32)
    else:
        samples = image.reshape(-1, image.shape[2]).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max_iter,
        1.0,
    )
    _, labels, centers = cv2.kmeans(
        samples,
        clusters,
        None,
        criteria,
        attempts,
        cv2.KMEANS_PP_CENTERS,
    )

    return labels.reshape(image.shape[:2]), centers


def reconstruct_segmented_image(
    labels: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """Reconstruct an image from K-Means labels and centers."""
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array.")
    if centers.ndim != 2:
        raise ValueError("centers must be a 2D array.")

    result = centers[labels]
    if centers.shape[1] == 1:
        result = result[..., 0]

    return np.clip(result, 0, 255).astype(np.uint8)
