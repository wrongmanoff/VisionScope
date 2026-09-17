# Algorithms

## Phase 4 — Image Processing

### Convolution

VisionScope includes a NumPy-based 2D convolution implementation for educational transparency.

For an image `I` and kernel `K`, convolution computes a weighted neighborhood sum:

`G(x, y) = sum_i sum_j I(x-i, y-j) K(i, j)`

The implementation uses zero padding and flips the kernel before applying it.

### Gaussian Filtering

A Gaussian kernel is generated mathematically and passed through the custom convolution implementation.

### Histogram Equalization

The grayscale histogram is converted into a cumulative distribution function (CDF), which is used to construct an intensity lookup table.

### Fourier Transform

The 2D FFT is computed with NumPy and shifted so low frequencies are centered. A logarithmic magnitude transform is used for visualization.

### Edge Detection

VisionScope currently provides Canny, LoG approximation, and DoG response generation.

Detailed comparisons and numerical evaluation will be added after experiments are executed.

## Phase 5 — Feature Extraction & Segmentation

### Harris Corners

Harris detects local image structures with strong corner response. The response and binary visualization are exposed for inspection.

### SIFT

SIFT detects scale-space keypoints and generates descriptors. VisionScope applies a brute-force descriptor matcher with Lowe's ratio test for robust candidate matches.

### HOG

HOG represents local shape information using histograms of gradient orientations. The current implementation uses OpenCV's HOGDescriptor while exposing the resulting feature vector.

### K-Means Segmentation

Pixels are represented as feature vectors and clustered into K groups. Cluster labels are then mapped back to image space to create a segmented image.

### Edge Segmentation

A Canny edge mask is provided as an edge-based segmentation primitive.

All numerical comparisons and performance claims will be generated from actual experiments.
