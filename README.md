# VisionScope

**A Classical Computer Vision Analysis and 3D Reconstruction Toolkit**

> CSE3010 — Computer Vision Project
> VIT Bhopal University

VisionScope is a modular Python-based computer vision toolkit that demonstrates classical computer vision techniques across image processing, feature extraction, segmentation, geometric vision, stereo reconstruction, motion analysis, and quantitative evaluation.

The project is designed as an educational and experimental toolkit rather than a single-purpose application. Each major component can be executed independently through a command-line interface, while the complete system provides a logical workflow from image preprocessing through visual analysis and 3D reconstruction.

---

## 1. Project Overview

Computer vision systems often combine several fundamental operations:

1. Image formation and preprocessing
2. Feature detection and description
3. Image segmentation
4. Camera and geometric modeling
5. Stereo correspondence and depth estimation
6. 3D reconstruction
7. Motion estimation and tracking
8. Quantitative algorithm evaluation

VisionScope implements these concepts using classical computer vision methods and provides reproducible command-line experiments and automated tests.

The project emphasizes:

* Mathematical understanding of computer vision algorithms
* Modular implementation
* Reproducible experiments
* Quantitative evaluation
* Clear separation between custom implementations and library-based methods
* Headless execution suitable for normal laptops and terminal environments

---

## 2. Objectives

The main objectives of VisionScope are to:

* Implement important classical computer vision algorithms covered in the CSE3010 syllabus.
* Demonstrate image processing and enhancement techniques.
* Extract and analyze visual features.
* Perform image segmentation using classical approaches.
* Demonstrate projective geometry, homography, RANSAC, camera projection, and rectification concepts.
* Estimate stereo disparity and depth.
* Generate 3D point clouds from depth information.
* Demonstrate optical flow and KLT/Lucas-Kanade tracking.
* Compare algorithms using measurable performance metrics.
* Provide automated tests for core functionality and CLI behavior.
* Maintain a clean, modular, and reproducible software architecture.

---

## 3. Implemented Modules

### Module 1 — Image Processing & Enhancement

Implemented techniques include:

* Custom 2D convolution using NumPy
* Gaussian kernel generation
* Gaussian filtering
* Histogram computation
* Histogram equalization
* Fourier transform and spectrum visualization
* Canny edge detection
* Laplacian of Gaussian (LoG)
* Difference of Gaussians (DoG)

Example:

```bash
python -m src.main image --input data/samples/test.jpg --task histogram
```

Other supported tasks include:

```bash
python -m src.main image --input data/samples/test.jpg --task equalize
python -m src.main image --input data/samples/test.jpg --task filter --method gaussian
python -m src.main image --input data/samples/test.jpg --task fourier
python -m src.main image --input data/samples/test.jpg --task edges --method canny
```

---

### Module 2 — Feature Extraction & Segmentation

Implemented feature methods:

* Harris corner detection
* SIFT keypoint and descriptor extraction
* SIFT ratio-test matching primitives
* HOG descriptor extraction

Implemented segmentation methods:

* K-Means image segmentation
* Edge-based segmentation

Examples:

```bash
python -m src.main features --input data/samples/test.jpg --method harris
python -m src.main features --input data/samples/test.jpg --method sift
python -m src.main features --input data/samples/test.jpg --method hog
```

Segmentation:

```bash
python -m src.main segment --input data/samples/test.jpg --method kmeans --clusters 3
python -m src.main segment --input data/samples/test.jpg --method edges
```

### HOG Implementation

VisionScope includes a NumPy implementation of the core HOG descriptor rather than depending on `cv2.HOGDescriptor`.

The implementation includes:

* Image gradient computation
* Gradient magnitude and orientation
* Unsigned orientation histograms
* Bilinear orientation-bin contribution
* Cell and block formation
* L2-Hys normalization

This also improves portability across OpenCV Python builds where the HOG descriptor API may not be available.

---

## 4. Geometric Vision

The geometric vision module demonstrates fundamental projective geometry concepts:

* Homography estimation
* Normalized Direct Linear Transform (DLT)
* RANSAC-based robust estimation
* Camera intrinsic matrix construction
* Projection matrix construction
* 3D-to-2D camera projection
* Stereo image rectification utilities

Example:

```bash
python -m src.main geometry --help
```

The geometric implementations are intended to demonstrate the underlying mathematics rather than simply wrapping a single high-level OpenCV call.

---

## 5. Stereo Depth & 3D Reconstruction

VisionScope provides a classical stereo reconstruction pipeline:

```text
Left Image ─────┐
                ├──> Stereo Matching ──> Disparity ──> Depth ──> Point Cloud
Right Image ────┘
```

Implemented methods include:

### SAD Block Matching

A custom Sum of Absolute Differences (SAD) stereo matcher is provided for disparity estimation.

### StereoSGBM

OpenCV StereoSGBM is included as a comparison baseline.

### Depth Estimation

Depth is estimated using the standard stereo relationship:

```text
Z = fB / d
```

where:

* `Z` = estimated depth
* `f` = focal length
* `B` = stereo baseline
* `d` = disparity

### Point Cloud Generation

Depth maps can be converted into 3D coordinates using the pinhole camera equations:

```text
X = (u - cx)Z / f
Y = (v - cy)Z / f
Z = depth
```

The resulting point cloud can be exported in ASCII PLY format.

Example:

```bash
python -m src.main reconstruct --help
```

---

## 6. Motion Analysis

VisionScope also includes classical motion-analysis techniques.

Implemented methods:

* Dense Farneback optical flow
* KLT / Lucas-Kanade feature tracking
* Motion magnitude statistics

Example:

```bash
python -m src.main motion --help
```

Example optical-flow execution:

```bash
python -m src.main motion \
    --prev data/samples/frame1.jpg \
    --curr data/samples/frame2.jpg \
    --method flow
```

The motion module provides a foundation for analyzing image sequences and estimating apparent motion between frames.

---

## 7. Evaluation & Experiments

VisionScope includes reusable evaluation utilities for:

### Classification

* Accuracy
* Precision
* Recall
* F1-score

### Segmentation

* Intersection over Union (IoU)
* Dice coefficient

### Regression

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

### Stereo Evaluation

The project includes a reproducible benchmark comparing:

* Custom SAD disparity estimation
* StereoSGBM

The benchmark uses a deterministic synthetic stereo-like image pair with a known horizontal shift.

Run:

```bash
python -m src.main experiment
```

The experiment records runtime and disparity-quality measurements.

Example output location:

```text
outputs/experiments/benchmark_results.csv
outputs/experiments/sad_disparity.png
outputs/experiments/sgbm_disparity.png
```

The benchmark results included in the development experiment were generated by actually executing the algorithms; no fabricated measurements are used.

---

## 8. Current Benchmark Results

The implemented stereo benchmark produced the following measurements during development:

| Method     | Runtime (s) | Valid Ratio |    MAE |   RMSE |
| ---------- | ----------: | ----------: | -----: | -----: |
| SAD        |    0.003910 |      0.6852 | 0.0684 | 0.5008 |
| StereoSGBM |    0.000757 |      0.8458 | 0.0019 | 0.0170 |

These values correspond to the deterministic synthetic benchmark configuration used by the project.

They should be interpreted as experiment-specific measurements rather than general claims about the algorithms. Runtime and accuracy can change with image size, scene content, hardware, parameters, and dataset characteristics.

---

## 9. Command-Line Interface

VisionScope provides a unified CLI:

```bash
python -m src.main --help
```

Available commands:

```text
image
features
segment
geometry
reconstruct
motion
experiment
```

### Image processing

```bash
python -m src.main image --help
```

### Features

```bash
python -m src.main features --help
```

### Segmentation

```bash
python -m src.main segment --help
```

### Geometry

```bash
python -m src.main geometry --help
```

### Reconstruction

```bash
python -m src.main reconstruct --help
```

### Motion

```bash
python -m src.main motion --help
```

### Experiments

```bash
python -m src.main experiment --help
```

The CLI validates arguments and reports invalid or missing inputs rather than silently producing invalid results.

---

## 10. Architecture

The project follows a modular architecture:

```text
VisionScope
│
├── preprocessing
│   ├── convolution
│   ├── filtering
│   ├── histogram
│   └── fourier
│
├── features
│   ├── Harris
│   ├── SIFT
│   └── HOG
│
├── segmentation
│   ├── K-Means
│   └── edge segmentation
│
├── geometry
│   ├── camera
│   ├── homography
│   ├── RANSAC
│   └── rectification
│
├── reconstruction
│   ├── disparity
│   ├── depth
│   └── point cloud
│
├── motion
│   ├── optical flow
│   └── KLT tracking
│
├── evaluation
│   └── metrics
│
├── experiments
│   └── benchmark
│
├── visualization
│
└── utils
```

A detailed architecture description is available in:

```text
docs/architecture.md
```

The Mermaid architecture diagram is available at:

```text
docs/diagrams/architecture.mmd
```

---

## 11. Project Structure

```text
VisionScope/
├── README.md
├── statement.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
│
├── docs/
│   ├── algorithms.md
│   ├── architecture.md
│   ├── evaluation.md
│   ├── workflow.md
│   ├── project_audit.md
│   ├── submission_checklist.md
│   ├── phase10_final_audit.md
│   └── diagrams/
│
├── src/
│   ├── main.py
│   ├── preprocessing/
│   ├── features/
│   ├── segmentation/
│   ├── geometry/
│   ├── reconstruction/
│   ├── motion/
│   ├── evaluation/
│   ├── experiments/
│   ├── visualization/
│   └── utils/
│
└── tests/
    ├── test_cli.py
    ├── test_evaluation.py
    ├── test_experiments.py
    ├── test_features.py
    ├── test_geometry.py
    ├── test_motion.py
    ├── test_preprocessing.py
    ├── test_reconstruction.py
    └── test_segmentation.py
```

---

## 12. Requirements

Recommended environment:

* Python 3.11+
* NumPy
* OpenCV
* SciPy
* scikit-learn
* Matplotlib
* Pandas
* Pytest

The project was developed and tested in a Python 3.14.6 virtual environment.

---

## 13. Installation

Clone the repository:

```bash
git clone https://github.com/wrongmanoff/VisionScope.git
cd VisionScope
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify the CLI:

```bash
python -m src.main --help
```

---

## 14. Testing

VisionScope includes an automated pytest suite covering the implemented modules and CLI.

Run:

```bash
pytest -q
```

Current development validation:

```text
38 passed
```

The tests cover:

* Preprocessing
* Feature extraction
* Segmentation
* Geometric vision
* Stereo reconstruction
* Motion analysis
* Evaluation metrics
* Experiment utilities
* CLI behavior

The test suite is intended to catch invalid inputs, shape errors, numerical issues, algorithm failures, and CLI integration problems.

---

## 15. Error Handling & Validation

The toolkit includes validation for common failure conditions such as:

* Missing input files
* Unsupported image formats
* Invalid command-line parameters
* Invalid image dimensions
* Invalid numerical ranges
* Invalid focal length or depth parameters
* Insufficient stereo correspondence
* Invalid point-cloud input
* Empty or incompatible image data

The project also uses structured logging with INFO, WARNING, and ERROR levels where appropriate.

---

## 16. Reproducibility

The experiment pipeline uses deterministic parameters and a fixed random seed for its synthetic stereo benchmark.

For example:

```text
height       = 160
width        = 220
shift        = 6
seed         = 42
```

This allows the benchmark to be repeated under the same software and hardware conditions.

---

## 17. Academic Alignment

VisionScope maps to multiple topics from the CSE3010 Computer Vision syllabus.

| Course Area                 | VisionScope Implementation                               |
| --------------------------- | -------------------------------------------------------- |
| Image formation / filtering | Convolution, Gaussian filtering                          |
| Histogram processing        | Histogram analysis and equalization                      |
| Fourier analysis            | Fourier spectrum                                         |
| Edge detection              | Canny, LoG, DoG                                          |
| Feature detection           | Harris, SIFT                                             |
| Feature description         | SIFT, HOG                                                |
| Segmentation                | K-Means, edge segmentation                               |
| Projective geometry         | Camera projection, homography                            |
| Robust estimation           | RANSAC                                                   |
| Stereo vision               | SAD, StereoSGBM                                          |
| Depth estimation            | `Z = fB/d`                                               |
| 3D reconstruction           | Point-cloud generation                                   |
| Motion estimation           | Optical flow                                             |
| Feature tracking            | KLT / Lucas-Kanade                                       |
| Evaluation                  | Classification, segmentation, regression, stereo metrics |

---

## 18. Limitations

Current limitations include:

* Stereo reconstruction depends on appropriate camera geometry and disparity quality.
* The implemented SAD matcher is intentionally educational and may be less robust than more advanced stereo methods.
* The benchmark uses a deterministic synthetic stereo-like pair and therefore does not represent all real-world scenes.
* HOG is implemented as a classical descriptor rather than as part of a learned detection pipeline.
* Motion estimation is sensitive to illumination changes, texture, occlusion, and frame quality.
* The toolkit is primarily intended for educational experimentation rather than production-scale computer vision.

---

## 19. Future Work

Possible extensions include:

* Real stereo-camera calibration and rectification workflow
* Support for public stereo datasets
* Additional feature descriptors such as SURF/GLOH alternatives
* More segmentation algorithms
* Graph-cut segmentation
* Mean-shift segmentation
* Background subtraction
* Improved optical-flow methods
* Object detection and tracking
* Photometric stereo
* Shape-from-texture and shape-from-motion experiments
* Interactive visualization
* Larger benchmark datasets
* More extensive runtime and memory profiling

---

## 20. Documentation

Additional project documentation:

```text
docs/algorithms.md
docs/architecture.md
docs/evaluation.md
docs/workflow.md
docs/project_audit.md
docs/phase5_1_changes.md
docs/phase5_2_changes.md
docs/phase6_geometric_vision.md
docs/phase7_stereo_reconstruction.md
docs/phase8_motion_evaluation.md
docs/phase9_experiments_results.md
docs/phase10_final_audit.md
docs/submission_checklist.md
```

---

## 21. Academic Integrity

VisionScope is developed as an academic computer vision project.

Experimental values are documented only when the corresponding experiment has actually been executed. Algorithm implementations are distinguished from library-based components where relevant.

No experimental performance values are intentionally fabricated.

---

## 22. Project Status

**Final implementation / submission preparation**

The project currently contains:

* 7 CLI modules
* Image processing and enhancement
* Feature extraction
* Segmentation
* Geometric vision
* Stereo depth estimation
* 3D point-cloud reconstruction
* Motion analysis
* Evaluation utilities
* Reproducible benchmarking
* Automated testing
* Project documentation

Latest automated validation:

```text
38 passed
```

---

## 23. Author

**Bhavesh Katragadda**

B.Tech Computer Science and Engineering
VIT Bhopal University

GitHub:

```text
https://github.com/wrongmanoff/VisionScope
```
