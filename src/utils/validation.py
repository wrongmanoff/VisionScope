"""Input validation helpers for VisionScope."""

from pathlib import Path

import cv2


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def validate_image_path(path: str | Path) -> Path:
    """Validate that a supported image exists and can be decoded."""
    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {image_path}")

    if not image_path.is_file():
        raise ValueError(f"Input path is not a file: {image_path}")

    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {image_path.suffix or '<none>'}"
        )

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to decode image: {image_path}")

    return image_path
