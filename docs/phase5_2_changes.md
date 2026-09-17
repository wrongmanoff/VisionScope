# Phase 5.2 — Final Compatibility Fix

## Fixed

### Harris
The public `detect_harris()` wrapper now correctly maps its `threshold`
argument to the underlying `harris_corners()` function's
`threshold_ratio` parameter.

### SIFT
`detect_sift()` now always returns keypoints as a Python list, including when
OpenCV reports no detected keypoints and returns an empty tuple.

## Validation

The feature test suite covers:

- Harris execution and output shape
- Public Harris API
- SIFT execution and descriptor dimensions
- HOG descriptor generation
- HOG parameter validation
- HOG resizing behavior

The CLI implementations for Harris, SIFT, HOG, K-Means segmentation, and
edge segmentation should continue to work unchanged.
