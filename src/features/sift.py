"""SIFT feature extraction and matching."""

import cv2
import numpy as np


def detect_sift(
    image: np.ndarray,
    nfeatures: int = 0,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """Detect SIFT keypoints and descriptors."""
    if image.ndim != 2:
        raise ValueError("detect_sift expects a grayscale image.")

    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # OpenCV can return an empty tuple when no keypoints are found.
    # VisionScope exposes a stable list-based public API.
    keypoints = list(keypoints) if keypoints is not None else []

    return keypoints, descriptors


def match_sift(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    ratio: float = 0.75,
) -> list[cv2.DMatch]:
    """Match SIFT descriptors using Lowe's ratio test."""
    if descriptors1 is None or descriptors2 is None:
        return []
    if not 0 < ratio < 1:
        raise ValueError("ratio must be between 0 and 1.")

    matcher = cv2.BFMatcher()
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good = []
    for pair in knn_matches:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
    return good


def draw_sift_matches(
    image1: np.ndarray,
    keypoints1: list[cv2.KeyPoint],
    image2: np.ndarray,
    keypoints2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
) -> np.ndarray:
    """Create a visualization of accepted feature matches."""
    return cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
