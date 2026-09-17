# Phase 7 — Stereo Depth & 3D Reconstruction

VisionScope now provides a classical stereo reconstruction pipeline:

```text
Left Image + Right Image
          ↓
   Stereo Matching
          ↓
       Disparity
          ↓
      Z = fB / d
          ↓
        Depth
          ↓
   X=(u-cx)Z/f
   Y=(v-cy)Z/f
          ↓
     3D Point Cloud
```

## Implemented

### 1. SAD Block Matching

`compute_disparity_sad()` implements classical local stereo matching using
Sum of Absolute Differences (SAD) over square windows.

### 2. SGBM baseline

`compute_disparity_sgbm()` provides an OpenCV StereoSGBM baseline for comparison.
It is explicitly documented as a library implementation.

### 3. Depth estimation

Stereo depth uses:

`Z = fB / d`

where `f` is focal length, `B` is stereo baseline, and `d` is disparity.

### 4. 3D reconstruction

For rectified stereo:

`X = (u-cx)Z/f`
`Y = (v-cy)Z/f`

The result can be exported as an ASCII PLY point cloud.

## CLI

```bash
python -m src.main reconstruct   --left data/samples/left.jpg   --right data/samples/right.jpg   --method sad   --max-disparity 64   --block-size 9   --focal-length 800   --baseline 0.10
```

The command produces:

- disparity visualization (`*_disparity.png`)
- depth array (`*_depth.npy`)
- point cloud (`*_pointcloud.ply`)

## Evaluation

`depth_error_metrics()` calculates MAE and RMSE when ground-truth depth is
available. No ground-truth values are fabricated by the application.

## Input requirement

The stereo images should be a rectified/calibrated stereo pair for physically
meaningful depth. Arbitrary unrelated images should not be used.

The baseline and focal length supplied to the CLI must correspond to the
actual camera setup/calibration.
