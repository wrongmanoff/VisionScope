"""VisionScope command-line interface."""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.motion import compute_dense_optical_flow, flow_statistics, track_features_klt

from src.preprocessing.convolution import gaussian_kernel
from src.preprocessing.fourier import fft_spectrum
from src.preprocessing.histogram import equalize_histogram, histogram
from src.preprocessing.filtering import (
    canny_edges,
    dog_edges,
    gaussian_filter,
    log_edges,
)
from src.features.harris import harris_corners
from src.features.hog import hog_descriptor
from src.features.sift import detect_sift
from src.segmentation.kmeans import kmeans_segment, reconstruct_segmented_image
from src.segmentation.edge_segmentation import edge_mask
from src.geometry.homography import estimate_homography_dlt, reprojection_errors
from src.geometry.ransac import ransac_homography
from src.geometry.camera import create_camera_matrix, projection_matrix, project_3d_points
from src.utils.validation import validate_image_path



from src.reconstruction.disparity import compute_disparity_sad, compute_disparity_sgbm, normalize_disparity
from src.reconstruction.depth import disparity_to_depth
from src.reconstruction.point_cloud import depth_to_point_cloud, save_point_cloud_ply

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cvscope",
        description="VisionScope - Classical Computer Vision Analysis Toolkit",
    )
    parser.add_argument("--version", action="version", version="VisionScope 0.5.0")

    subparsers = parser.add_subparsers(dest="command")

    image = subparsers.add_parser("image", help="Run image-processing operations.")
    image.add_argument("--input", required=True, help="Input image path.")
    image.add_argument("--output", help="Output file or output directory.")
    image.add_argument(
        "--task",
        required=True,
        choices=["histogram", "equalize", "filter", "fourier", "edges"],
        help="Image-processing task.",
    )
    image.add_argument(
        "--method",
        choices=["gaussian", "canny", "log", "dog"],
        help="Algorithm for tasks that require a method.",
    )
    image.add_argument("--size", type=int, default=5, help="Gaussian kernel size.")
    image.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma.")
    image.add_argument("--low", type=int, default=100, help="Canny low threshold.")
    image.add_argument("--high", type=int, default=200, help="Canny high threshold.")

    features = subparsers.add_parser(
        "features", help="Extract and visualize image features."
    )
    features.add_argument("--input", required=True, help="Input image path.")
    features.add_argument(
        "--method", required=True, choices=["harris", "sift", "hog"],
        help="Feature extraction method."
    )
    features.add_argument("--output", help="Output path.")
    features.add_argument("--ratio", type=float, default=0.75)
    features.add_argument("--nfeatures", type=int, default=0)

    segment = subparsers.add_parser(
        "segment", help="Segment an image."
    )
    segment.add_argument("--input", required=True, help="Input image path.")
    segment.add_argument(
        "--method", required=True, choices=["kmeans", "edges"],
        help="Segmentation method."
    )
    segment.add_argument("--output", help="Output path.")
    segment.add_argument("--clusters", type=int, default=3)

    geometry = subparsers.add_parser(
        "geometry", help="Run geometric computer-vision operations."
    )
    geometry.add_argument(
        "--task",
        required=True,
        choices=["homography", "camera"],
        help="Geometric-vision task.",
    )
    geometry.add_argument("--input", help="First input image for homography.")
    geometry.add_argument("--input2", help="Second input image for homography.")
    geometry.add_argument("--output", help="Output visualization path.")
    geometry.add_argument("--method", choices=["dlt", "ransac"], default="ransac")
    geometry.add_argument("--threshold", type=float, default=3.0)
    geometry.add_argument("--iterations", type=int, default=1000)
    geometry.add_argument("--ratio", type=float, default=0.75)
    geometry.add_argument("--fx", type=float, default=800.0)
    geometry.add_argument("--fy", type=float, default=800.0)
    geometry.add_argument("--cx", type=float, default=320.0)
    geometry.add_argument("--cy", type=float, default=240.0)

    reconstruction = subparsers.add_parser(
        "reconstruct", help="Run stereo disparity, depth, and 3D reconstruction."
    )
    reconstruction.add_argument("--left", required=True, help="Left stereo image.")
    reconstruction.add_argument("--right", required=True, help="Right stereo image.")
    reconstruction.add_argument(
        "--method", choices=["sad", "sgbm"], default="sad",
        help="Disparity method."
    )
    reconstruction.add_argument("--max-disparity", type=int, default=64)
    reconstruction.add_argument("--block-size", type=int, default=9)
    reconstruction.add_argument("--focal-length", type=float, required=True)
    reconstruction.add_argument("--baseline", type=float, required=True)
    reconstruction.add_argument("--cx", type=float)
    reconstruction.add_argument("--cy", type=float)
    reconstruction.add_argument("--max-depth", type=float)
    reconstruction.add_argument("--output-prefix", default="outputs/reconstruction/stereo")

    motion = subparsers.add_parser("motion", help="Analyze motion between two frames.")
    motion.add_argument("--prev", required=True, help="Previous frame.")
    motion.add_argument("--curr", required=True, help="Current frame.")
    motion.add_argument("--method", choices=["flow", "klt"], default="flow")
    motion.add_argument("--window-size", type=int, default=15)
    motion.add_argument("--levels", type=int, default=3)
    motion.add_argument("--max-corners", type=int, default=200)
    motion.add_argument("--output-dir", default="outputs/motion")

    experiment = subparsers.add_parser(
        "experiment", help="Run reproducible algorithm benchmarks."
    )
    experiment.add_argument("--height", type=int, default=160)
    experiment.add_argument("--width", type=int, default=220)
    experiment.add_argument("--shift", type=int, default=6)
    experiment.add_argument("--seed", type=int, default=42)
    experiment.add_argument("--max-disparity", type=int, default=32)
    experiment.add_argument("--block-size", type=int, default=9)
    experiment.add_argument("--repeats", type=int, default=3)
    experiment.add_argument("--output-dir", default="outputs/experiments")

    return parser


def _load_grayscale(path: str) -> np.ndarray:
    image_path = validate_image_path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to decode image: {image_path}")
    return image


def _default_output(input_path: str, task: str) -> Path:
    path = Path("outputs/processed") / f"{Path(input_path).stem}_{task}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_histogram(image: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(histogram(image))
    plt.title("Grayscale Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def _save_spectrum(image: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(fft_spectrum(image), cmap="gray")
    plt.title("Fourier Magnitude Spectrum")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def run_image_task(args: argparse.Namespace) -> int:
    image = _load_grayscale(args.input)

    if args.task == "histogram":
        output = Path(args.output) if args.output else _default_output(args.input, "histogram")
        _save_histogram(image, output)
    elif args.task == "equalize":
        output = Path(args.output) if args.output else _default_output(args.input, "equalized")
        cv2.imwrite(str(output), equalize_histogram(image))
    elif args.task == "filter":
        if args.method != "gaussian":
            raise ValueError("The filter task currently supports --method gaussian.")
        output = Path(args.output) if args.output else _default_output(args.input, "gaussian")
        cv2.imwrite(str(output), gaussian_filter(image, args.size, args.sigma))
    elif args.task == "fourier":
        output = Path(args.output) if args.output else _default_output(args.input, "fourier")
        _save_spectrum(image, output)
    elif args.task == "edges":
        if args.method == "canny":
            result = canny_edges(image, args.low, args.high)
        elif args.method == "log":
            result = log_edges(image, args.sigma)
        elif args.method == "dog":
            result = dog_edges(image, args.sigma, args.sigma * 2)
        else:
            raise ValueError("Choose --method canny, log, or dog for edge detection.")

        output = Path(args.output) if args.output else _default_output(
            args.input, f"edges_{args.method}"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), result)

    print(f"Saved result: {output}")
    return 0


def run_feature_task(args: argparse.Namespace) -> int:
    image = _load_grayscale(args.input)
    output = Path(args.output) if args.output else Path(
        "outputs/processed"
    ) / f"{Path(args.input).stem}_{args.method}.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.method == "harris":
        _, mask = harris_corners(image)
        visualization = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        visualization[mask > 0] = (0, 0, 255)
        cv2.imwrite(str(output), visualization)
        print(f"Detected Harris corner pixels: {int(np.count_nonzero(mask))}")

    elif args.method == "sift":
        keypoints, descriptors = detect_sift(image, args.nfeatures)
        visualization = cv2.drawKeypoints(
            image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv2.imwrite(str(output), visualization)
        print(f"SIFT keypoints: {len(keypoints)}")
        print(
            f"SIFT descriptor shape: "
            f"{None if descriptors is None else descriptors.shape}"
        )

    elif args.method == "hog":
        descriptor = hog_descriptor(image)
        np.save(str(output.with_suffix(".npy")), descriptor)
        print(f"HOG descriptor length: {len(descriptor)}")
        print(f"Saved descriptor: {output.with_suffix('.npy')}")

    if args.method != "hog":
        print(f"Saved result: {output}")
    return 0


def run_segment_task(args: argparse.Namespace) -> int:
    image_path = validate_image_path(args.input)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {image_path}")

    output = Path(args.output) if args.output else Path(
        "outputs/processed"
    ) / f"{Path(args.input).stem}_{args.method}_segmentation.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.method == "kmeans":
        labels, centers = kmeans_segment(image, args.clusters)
        result = reconstruct_segmented_image(labels, centers)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = edge_mask(gray)

    cv2.imwrite(str(output), result)
    print(f"Saved result: {output}")
    return 0


def run_geometry_task(args: argparse.Namespace) -> int:
    """Run homography or camera-geometry demonstrations."""
    if args.task == "homography":
        if not args.input or not args.input2:
            raise ValueError("Homography requires both --input and --input2.")

        path1 = validate_image_path(args.input)
        path2 = validate_image_path(args.input2)
        image1 = cv2.imread(str(path1), cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(str(path2), cv2.IMREAD_GRAYSCALE)

        if image1 is None or image2 is None:
            raise ValueError("Unable to decode one or both input images.")

        keypoints1, descriptors1 = detect_sift(image1)
        keypoints2, descriptors2 = detect_sift(image2)

        if descriptors1 is None or descriptors2 is None:
            raise ValueError("Insufficient SIFT features for homography estimation.")

        from src.features.sift import match_sift
        matches = match_sift(descriptors1, descriptors2, args.ratio)

        if len(matches) < 4:
            raise ValueError(
                f"At least 4 good SIFT matches are required; found {len(matches)}."
            )

        points1 = np.float64([keypoints1[m.queryIdx].pt for m in matches])
        points2 = np.float64([keypoints2[m.trainIdx].pt for m in matches])

        if args.method == "dlt":
            H = estimate_homography_dlt(points1, points2)
            errors = reprojection_errors(points1, points2, H)
            inlier_mask = errors <= args.threshold
            inlier_count = int(inlier_mask.sum())
        else:
            H, inlier_mask, _ = ransac_homography(
                points1,
                points2,
                threshold=args.threshold,
                iterations=args.iterations,
            )
            errors = reprojection_errors(points1, points2, H)
            inlier_count = int(inlier_mask.sum())

        output = Path(args.output) if args.output else Path(
            "outputs/processed"
        ) / f"{Path(args.input).stem}_to_{Path(args.input2).stem}_homography.png"
        output.parent.mkdir(parents=True, exist_ok=True)

        visualization = cv2.drawMatches(
            image1,
            keypoints1,
            image2,
            keypoints2,
            [m for i, m in enumerate(matches) if inlier_mask[i]],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(str(output), visualization)

        print(f"Good SIFT matches: {len(matches)}")
        print(f"Homography method: {args.method.upper()}")
        print(f"Inliers: {inlier_count}/{len(matches)}")
        print("Homography matrix:")
        print(H)
        print(f"Median reprojection error: {float(np.median(errors)):.4f} pixels")
        print(f"Saved result: {output}")
        return 0

    # Camera geometry: deterministic pinhole projection demonstration.
    K = create_camera_matrix(args.fx, args.fy, args.cx, args.cy)
    P = projection_matrix(K)
    points_3d = np.array(
        [[-1.0, -1.0, 5.0], [1.0, -1.0, 5.0], [1.0, 1.0, 5.0], [-1.0, 1.0, 5.0]],
        dtype=np.float64,
    )
    points_2d = project_3d_points(points_3d, P)

    print("Camera intrinsic matrix K:")
    print(K)
    print("Projection matrix P = K[R|t]:")
    print(P)
    print("3D points:")
    print(points_3d)
    print("Projected 2D points:")
    print(points_2d)
    return 0



def run_reconstruction_task(args: argparse.Namespace) -> int:
    """Run stereo disparity -> depth -> point-cloud reconstruction."""
    left_path = validate_image_path(args.left)
    right_path = validate_image_path(args.right)

    left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise ValueError("Unable to decode one or both stereo images.")
    if left.shape != right.shape:
        raise ValueError("Left and right stereo images must have identical dimensions.")

    if args.method == "sad":
        disparity = compute_disparity_sad(
            left, right,
            max_disparity=args.max_disparity,
            block_size=args.block_size,
        )
    else:
        num_disp = args.max_disparity
        if num_disp % 16 != 0:
            raise ValueError("--max-disparity must be a multiple of 16 for SGBM.")
        disparity = compute_disparity_sgbm(
            left, right,
            num_disparities=num_disp,
            block_size=args.block_size if args.block_size % 2 else args.block_size + 1,
        )

    h, w = left.shape
    cx = args.cx if args.cx is not None else w / 2.0
    cy = args.cy if args.cy is not None else h / 2.0

    depth = disparity_to_depth(
        disparity,
        focal_length=args.focal_length,
        baseline=args.baseline,
    )
    points = depth_to_point_cloud(
        depth,
        focal_length=args.focal_length,
        cx=cx,
        cy=cy,
        max_depth=args.max_depth,
    )

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    disparity_path = prefix.with_name(prefix.name + "_disparity.png")
    depth_path = prefix.with_name(prefix.name + "_depth.npy")
    cloud_path = prefix.with_name(prefix.name + "_pointcloud.ply")

    cv2.imwrite(str(disparity_path), normalize_disparity(disparity))
    np.save(str(depth_path), depth)
    save_point_cloud_ply(points, str(cloud_path))

    valid = disparity > 0
    print(f"Disparity method: {args.method.upper()}")
    print(f"Valid disparity pixels: {int(valid.sum())}/{valid.size}")
    if np.any(valid):
        print(f"Median disparity: {float(np.median(disparity[valid])):.3f} px")
    valid_depth = depth > 0
    if np.any(valid_depth):
        print(f"Median depth: {float(np.median(depth[valid_depth])):.3f}")
    print(f"Point-cloud points: {len(points)}")
    print(f"Saved disparity: {disparity_path}")
    print(f"Saved depth: {depth_path}")
    print(f"Saved point cloud: {cloud_path}")
    return 0


def run_motion_task(args: argparse.Namespace) -> int:
    """Analyze motion between two frames using optical flow or KLT tracking."""
    prev = cv2.imread(str(validate_image_path(args.prev)), cv2.IMREAD_GRAYSCALE)
    curr = cv2.imread(str(validate_image_path(args.curr)), cv2.IMREAD_GRAYSCALE)
    if prev is None or curr is None:
        raise ValueError("Unable to decode one or both input frames.")
    if prev.shape != curr.shape:
        raise ValueError("Input frames must have identical dimensions.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "flow":
        flow = compute_dense_optical_flow(
            prev, curr, winsize=args.window_size, levels=args.levels
        )
        magnitude = np.linalg.norm(flow, axis=-1)
        magnitude_image = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        output = output_dir / "optical_flow_magnitude.png"
        cv2.imwrite(str(output), magnitude_image)
        stats = flow_statistics(flow)
        print(f"Optical flow shape: {flow.shape}")
        print(f"Mean magnitude: {stats['mean_magnitude']:.4f}")
        print(f"Max magnitude: {stats['max_magnitude']:.4f}")
        print(f"Moving pixel ratio: {stats['moving_pixel_ratio']:.4f}")
        print(f"Saved result: {output}")
    else:
        prev_pts, curr_pts = track_features_klt(
            prev, curr, max_corners=args.max_corners
        )
        output = output_dir / "klt_tracks.csv"
        tracks = np.column_stack((prev_pts, curr_pts)) if len(prev_pts) else np.empty((0, 4))
        np.savetxt(
            str(output), tracks, delimiter=",",
            header="x_prev,y_prev,x_curr,y_curr", comments=""
        )
        print(f"Tracked points: {len(prev_pts)}")
        print(f"Saved tracks: {output}")
    return 0


def run_experiment(args: argparse.Namespace) -> int:
    """Run a reproducible stereo-disparity benchmark."""
    import csv
    from src.experiments.benchmark import (
        make_shift_pair,
        time_callable,
        disparity_quality,
    )
    from src.reconstruction.disparity import (
        compute_disparity_sad,
        compute_disparity_sgbm,
    )

    left, right = make_shift_pair(
        height=args.height,
        width=args.width,
        shift=args.shift,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for method in ("sad", "sgbm"):
        if method == "sad":
            func = compute_disparity_sad
            kwargs = {
                "max_disparity": args.max_disparity,
                "block_size": args.block_size,
            }
        else:
            func = compute_disparity_sgbm
            kwargs = {
                "num_disparities": args.max_disparity,
                "block_size": args.block_size,
            }

        disparity, runtime = time_callable(
            func,
            left,
            right,
            repeats=args.repeats,
            **kwargs,
        )

        quality = disparity_quality(
            disparity,
            float(args.shift),
        )

        rows.append({
            "method": method,
            "runtime_seconds": runtime,
            **quality,
        })

        normalized = cv2.normalize(
            disparity,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

        output_path = output_dir / f"{method}_disparity.png"
        cv2.imwrite(str(output_path), normalized)

    csv_path = output_dir / "benchmark_results.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Benchmark results:")

    for row in rows:
        print(
            f"{row['method']}: "
            f"runtime={row['runtime_seconds']:.6f}s, "
            f"valid_ratio={row['valid_ratio']:.4f}, "
            f"MAE={row['mae']:.4f}, "
            f"RMSE={row['rmse']:.4f}"
        )

    print(f"Saved: {csv_path}")

    return 0

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "image":
            return run_image_task(args)
        if args.command == "features":
            return run_feature_task(args)
        if args.command == "segment":
            return run_segment_task(args)
        if args.command == "geometry":
            return run_geometry_task(args)
        if args.command == "reconstruct":
            return run_reconstruction_task(args)
        if args.command == "motion":
            return run_motion_task(args)
        if args.command == "experiment":
            return run_experiment(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
