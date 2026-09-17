# Phase 9 — Experiments, Benchmarking & Results

## Purpose
Phase 9 turns the implemented algorithms into reproducible experiments.
Results are measured from deterministic synthetic data and written to CSV
and image files. The experiment deliberately avoids fabricated values.

## Benchmark
The included benchmark compares the project's custom SAD stereo disparity
implementation with OpenCV StereoSGBM on a synthetic stereo-like pair with
a known horizontal shift.

Recorded fields:
- method
- runtime_seconds
- valid_ratio
- MAE
- RMSE

The synthetic pair is generated with a fixed random seed and known shift,
so the experiment has a ground-truth disparity for quantitative comparison.

## CLI

```bash
python -m src.main experiment
```

Optional example:

```bash
python -m src.main experiment         --height 160         --width 220         --shift 6         --max-disparity 32         --block-size 9         --repeats 3
```

## Outputs

Results are saved to:

`outputs/experiments/`

including:
- `benchmark_results.csv`
- `sad_disparity.png`
- `sgbm_disparity.png`

## Interpreting results
Do not claim that one algorithm is universally better from this experiment.
Report the measured runtime and error on this defined synthetic setup,
then discuss limitations and dataset dependence.

## Final project experiments
For the final report, supplement this benchmark with real public data where
available:
- stereo pair → disparity/depth error
- feature detection → keypoint counts and runtime
- segmentation → IoU/Dice when ground truth exists
- motion → tracked-point count and flow statistics

Every reported number should come from an executed experiment and record
the dataset, parameters, hardware/software environment, and date.
