# Phase 6 — Geometric Vision

## Implemented

VisionScope now includes a geometric-vision module covering:

1. **Normalized DLT Homography**
   - Estimates a 3×3 projective transformation from point correspondences.
   - Uses point normalization and SVD.
   - Includes reprojection-error calculation.

2. **RANSAC Homography**
   - Samples four correspondences per hypothesis.
   - Scores hypotheses using reprojection error.
   - Selects an inlier consensus set and refines the homography.

3. **Pinhole Camera Geometry**
   - Builds the intrinsic matrix K.
   - Builds P = K[R|t].
   - Projects 3D points into image coordinates.

4. **Stereo Rectification API**
   - Provides a validated wrapper around `cv2.stereoRectify`.
   - Requires actual camera calibration parameters rather than inventing them.

## CLI

### Homography + RANSAC

```bash
python -m src.main geometry   --task homography   --input left.jpg   --input2 right.jpg   --method ransac
```

### Homography + DLT

```bash
python -m src.main geometry   --task homography   --input image1.jpg   --input2 image2.jpg   --method dlt
```

### Camera projection demonstration

```bash
python -m src.main geometry --task camera
```

## Academic rationale

The module deliberately exposes the mathematical pipeline rather than hiding
homography estimation behind a single `cv2.findHomography()` call. OpenCV is
still used for image I/O, SIFT, matching, and the stereo-rectification
primitive where appropriate.

## Validation

Tests use synthetic projective correspondences with a known ground-truth
homography, add RANSAC outliers, and verify camera projection numerically.
