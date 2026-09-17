# Phase 5.1 — Bug Fixes

## Fixed

1. **SIFT return type**
   - Normalized OpenCV keypoints to a Python `list`.
   - Keeps the public function API consistent even when OpenCV returns an empty tuple.

2. **HOG compatibility**
   - Removed the dependency on `cv2.HOGDescriptor`.
   - Added a NumPy-based HOG implementation.
   - Includes Sobel gradients, magnitude/orientation calculation, 0–180° orientation bins,
     bilinear orientation binning, block formation, and L2-Hys normalization.

3. **Validation and tests**
   - Added validation for invalid HOG dimensions/parameters.
   - Added tests for normal HOG execution and non-standard input sizes.

## Expected result

The complete test suite should pass, and this command should work:

```bash
python -m src.main features --input data/samples/test.jpg --method hog
```
