"""Histogram of Oriented Gradients (HOG) feature extraction.

This module implements the core HOG descriptor using NumPy so that the
project does not depend on cv2.HOGDescriptor being available in a particular
OpenCV build.
"""

from __future__ import annotations

import cv2
import numpy as np


def _validate_params(
    image: np.ndarray,
    win_size: tuple[int, int],
    block_size: tuple[int, int],
    block_stride: tuple[int, int],
    cell_size: tuple[int, int],
    bins: int,
) -> None:
    if image.ndim not in (2, 3):
        raise ValueError("hog_descriptor expects a grayscale or color image.")
    if bins <= 0:
        raise ValueError("bins must be positive.")

    width, height = win_size
    bw, bh = block_size
    sw, sh = block_stride
    cw, ch = cell_size

    if min(width, height, bw, bh, sw, sh, cw, ch) <= 0:
        raise ValueError("HOG dimensions and strides must be positive.")

    if bw % cw or bh % ch:
        raise ValueError("block_size must be divisible by cell_size.")

    if sw % cw or sh % ch:
        raise ValueError("block_stride must be divisible by cell_size.")

    if width % cw or height % ch:
        raise ValueError("win_size must be divisible by cell_size.")

    if bw > width or bh > height:
        raise ValueError("block_size cannot exceed win_size.")

    if sw > bw or sh > bh:
        raise ValueError("block_stride cannot exceed block_size.")


def hog_descriptor(
    image: np.ndarray,
    win_size: tuple[int, int] = (64, 128),
    block_size: tuple[int, int] = (16, 16),
    block_stride: tuple[int, int] = (8, 8),
    cell_size: tuple[int, int] = (8, 8),
    bins: int = 9,
) -> np.ndarray:
    """Compute a HOG descriptor using NumPy.

    Parameters follow OpenCV's width/height convention. Gradients use Sobel
    derivatives. Orientations are unsigned (0-180 degrees), and each cell
    contributes a histogram of gradient magnitudes. Histograms are grouped
    into blocks and normalized with L2-Hys normalization.
    """
    _validate_params(
        image, win_size, block_size, block_stride, cell_size, bins
    )

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    width, height = win_size
    if gray.shape[1] != width or gray.shape[0] != height:
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)

    gray = gray.astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle %= 180.0

    cw, ch = cell_size
    cells_x = width // cw
    cells_y = height // ch
    bin_width = 180.0 / bins

    # Each cell stores one orientation histogram.
    hist = np.zeros((cells_y, cells_x, bins), dtype=np.float32)

    for cy in range(cells_y):
        y0, y1 = cy * ch, (cy + 1) * ch
        for cx in range(cells_x):
            x0, x1 = cx * cw, (cx + 1) * cw

            cell_mag = magnitude[y0:y1, x0:x1].ravel()
            cell_ang = angle[y0:y1, x0:x1].ravel()

            # Bilinear interpolation between adjacent orientation bins.
            pos = cell_ang / bin_width
            lower = np.floor(pos).astype(int) % bins
            upper = (lower + 1) % bins
            upper_weight = pos - np.floor(pos)
            lower_weight = 1.0 - upper_weight

            np.add.at(
                hist[cy, cx],
                lower,
                cell_mag * lower_weight,
            )
            np.add.at(
                hist[cy, cx],
                upper,
                cell_mag * upper_weight,
            )

    bw, bh = block_size
    sw, sh = block_stride

    cells_per_block_x = bw // cw
    cells_per_block_y = bh // ch
    stride_cells_x = sw // cw
    stride_cells_y = sh // ch

    blocks_x = (cells_x - cells_per_block_x) // stride_cells_x + 1
    blocks_y = (cells_y - cells_per_block_y) // stride_cells_y + 1

    features: list[np.ndarray] = []

    for by in range(blocks_y):
        cy0 = by * stride_cells_y
        for bx in range(blocks_x):
            cx0 = bx * stride_cells_x

            block = hist[
                cy0 : cy0 + cells_per_block_y,
                cx0 : cx0 + cells_per_block_x,
            ].ravel()

            # L2-Hys normalization: L2 normalize, clip, then normalize again.
            eps = 1e-6
            norm = np.sqrt(np.sum(block * block) + eps * eps)
            normalized = block / norm
            normalized = np.minimum(normalized, 0.2)

            norm2 = np.sqrt(np.sum(normalized * normalized) + eps * eps)
            normalized = normalized / norm2

            features.append(normalized.astype(np.float32))

    if not features:
        return np.empty((0,), dtype=np.float32)

    return np.concatenate(features)


def draw_hog_visualization(
    image: np.ndarray,
    descriptor: np.ndarray,
) -> np.ndarray:
    """Return the source image annotated with basic HOG descriptor metadata."""
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    cv2.putText(
        canvas,
        f"HOG descriptor length: {descriptor.size}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas
