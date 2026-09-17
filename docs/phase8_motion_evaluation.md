# Phase 8 — Motion Analysis & Quantitative Evaluation

## Purpose
Phase 8 extends VisionScope from static-image and stereo reconstruction
into classical motion analysis and reusable quantitative evaluation.

## Functional modules
1. Dense optical flow using Farneback as a classical library baseline.
2. KLT/Lucas-Kanade feature tracking using Shi-Tomasi points.
3. Quantitative metrics for classification, segmentation, and regression.
4. CLI execution with saved headless outputs.

## Optical flow
For consecutive frames, the flow field estimates a displacement vector
`(u, v)` at each pixel. VisionScope reports:
- flow tensor shape
- mean motion magnitude
- maximum motion magnitude
- fraction of pixels with magnitude greater than 1 pixel

The implementation uses OpenCV's Farneback algorithm for the numerical
flow computation and NumPy for analysis. It is explicitly treated as a
library implementation, not a custom reimplementation of Farneback.

## KLT tracking
Shi-Tomasi corners are detected in the previous frame and tracked with
pyramidal Lucas-Kanade optical flow. Valid point pairs are exported as CSV.

## Evaluation
Available metrics:
- Accuracy, precision, recall, F1
- IoU and Dice
- MAE and RMSE

Real experiment results must be measured from actual datasets. Do not
insert invented numbers into the report.

## CLI examples

```bash
python -m src.main motion         --prev data/samples/frame1.jpg         --curr data/samples/frame2.jpg         --method flow
```

```bash
python -m src.main motion         --prev data/samples/frame1.jpg         --curr data/samples/frame2.jpg         --method klt
```

## Outputs
Motion outputs are saved under `outputs/motion/`:
- `optical_flow_magnitude.png`
- `klt_tracks.csv`

## Validation
```bash
pytest
python -m src.main motion --help
```

## Academic distinction
Custom/NumPy work in the project should be described separately from
OpenCV/scikit-learn algorithms used as library baselines.
