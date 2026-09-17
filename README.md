# VisionScope

A Classical Computer Vision Analysis and 3D Reconstruction Toolkit.

> CSE3010 Computer Vision project

## Status

Phase 3 — Project scaffold.

The implementation will be developed incrementally. Experimental results will only be added after the corresponding experiments are actually executed.

## Planned modules

1. Image Processing & Enhancement
2. Feature & Segmentation Analysis
3. Geometric Vision
4. Stereo Depth & 3D Reconstruction
5. Optional Motion Analysis

## Planned stack

- Python 3.11+
- OpenCV
- NumPy
- SciPy
- Scikit-learn
- Matplotlib
- Pandas
- Pytest

## Initial setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main --help
```

## Results

Results are intentionally omitted until experiments are executed.

## Phase 4 — Image Processing

Implemented:

- Custom NumPy convolution
- Gaussian filtering
- Histogram analysis
- Histogram equalization
- Fourier spectrum visualization
- Canny edges
- LoG response
- DoG response

Examples:

```bash
python -m src.main image --input data/samples/image.jpg --task histogram
python -m src.main image --input data/samples/image.jpg --task equalize
python -m src.main image --input data/samples/image.jpg --task filter --method gaussian
python -m src.main image --input data/samples/image.jpg --task fourier
python -m src.main image --input data/samples/image.jpg --task edges --method canny
```

Numerical results and benchmark values will only be documented after real experiments are run.

## Phase 5 — Feature Extraction & Segmentation

Implemented:

- Harris corner detection
- SIFT keypoints and descriptors
- SIFT ratio-test matching primitives
- HOG descriptor extraction
- K-Means image segmentation
- Edge-based segmentation

Examples:

```bash
python -m src.main features --input data/samples/image.jpg --method harris
python -m src.main features --input data/samples/image.jpg --method sift
python -m src.main features --input data/samples/image.jpg --method hog
python -m src.main segment --input data/samples/image.jpg --method kmeans --clusters 3
python -m src.main segment --input data/samples/image.jpg --method edges
```
### HOG implementation note
VisionScope implements the core Histogram of Oriented Gradients (HOG) descriptor with NumPy rather than relying on `cv2.HOGDescriptor`, because that API is not exposed by every OpenCV Python build. The implementation includes gradient computation, unsigned orientation histograms, bilinear binning, block formation, and L2-Hys normalization.


## Stereo Depth & 3D Reconstruction

Phase 7 adds classical stereo SAD block matching, an SGBM comparison baseline,
stereo depth estimation using `Z = fB/d`, and export of a 3D point cloud in PLY
format.


## Phase 8 — Motion Analysis & Evaluation

VisionScope now includes classical motion-analysis and evaluation utilities:
- dense Farneback optical flow analysis
- KLT/Lucas-Kanade feature tracking
- motion magnitude statistics
- classification, segmentation, and regression metrics
- headless CLI outputs for reproducible experiments

Example:
```bash
python -m src.main motion --prev data/samples/frame1.jpg --curr data/samples/frame2.jpg --method flow
```

See `docs/phase8_motion_evaluation.md` for details.


## Phase 9 — Experiments & Results

VisionScope now includes a reproducible stereo-disparity benchmark:
```bash
python -m src.main experiment
```
It compares custom SAD block matching with StereoSGBM on a deterministic
synthetic pair and saves runtime/error measurements to
`outputs/experiments/benchmark_results.csv`.

See `docs/phase9_experiments_results.md`.


## Phase 10 — Final Audit

Before submission, follow `docs/phase10_final_audit.md` and
`docs/submission_checklist.md`. Run the complete test suite and preserve
actual benchmark outputs. Do not fabricate experimental measurements.
