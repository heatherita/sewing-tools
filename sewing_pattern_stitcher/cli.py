from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import (
    DEFAULT_ARUCO_DICTIONARY,
    DEFAULT_CALIBRATION_OBJECT,
    DEFAULT_CALIBRATION_UNIT,
    DEFAULT_CALIBRATION_WIDTH,
    CalibrationObject,
    ThresholdMode,
    process_images,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sewstitch",
        description="Stitch ArUco-marked sewing pattern images and export SVG paths.",
    )
    parser.add_argument("images", nargs="+", type=Path, help="Input image paths.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("pattern.svg"),
        help="SVG output path.",
    )
    parser.add_argument(
        "--aruco-dictionary",
        default=DEFAULT_ARUCO_DICTIONARY,
        help="OpenCV ArUco dictionary name, e.g. DICT_4X4_50 or DICT_6X6_250.",
    )
    parser.add_argument(
        "--calibration-object",
        choices=[item.value for item in CalibrationObject],
        default=DEFAULT_CALIBRATION_OBJECT,
        help="Object used to scale SVG output. Defaults to none, which keeps raw pixel units.",
    )
    parser.add_argument(
        "--calibration-width",
        type=float,
        default=DEFAULT_CALIBRATION_WIDTH,
        help="Real width of the calibration object in --calibration-unit.",
    )
    parser.add_argument(
        "--calibration-unit",
        default=DEFAULT_CALIBRATION_UNIT,
        help="SVG unit for calibrated output, e.g. mm, cm, or in.",
    )
    parser.add_argument(
        "--calibration-points",
        type=parse_calibration_points,
        metavar="X1,Y1,X2,Y2",
        help=(
            "Manual calibration endpoints in original image pixels. "
            "Required with --calibration-object manual."
        ),
    )
    parser.add_argument(
        "--calibration-image-index",
        type=int,
        default=0,
        help="Zero-based input image index that contains --calibration-points.",
    )
    parser.add_argument(
        "--threshold",
        choices=[mode.value for mode in ThresholdMode],
        default=ThresholdMode.OTSU.value,
        help="Thresholding method.",
    )
    parser.add_argument(
        "--threshold-value",
        type=int,
        default=127,
        help="Fixed threshold value when --threshold fixed is used.",
    )
    parser.add_argument(
        "--threshold-offset",
        type=int,
        default=0,
        help=(
            "Adjustment added to Otsu's automatically selected threshold. "
            "Negative values are stricter; positive values capture fainter lines."
        ),
    )
    parser.add_argument(
        "--adaptive-block-size",
        type=parse_adaptive_block_size,
        default=35,
        help=(
            "Odd pixel window size for --threshold adaptive. Larger values ignore more "
            "local paper texture and shadow variation."
        ),
    )
    parser.add_argument(
        "--adaptive-c",
        type=float,
        default=5.0,
        help=(
            "Constant subtracted from the local adaptive threshold. Larger values are "
            "stricter for dark-line extraction."
        ),
    )
    threshold_polarity = parser.add_mutually_exclusive_group()
    threshold_polarity.add_argument(
        "--invert",
        dest="invert",
        action="store_true",
        default=True,
        help="Extract dark lines on light paper. This is the default.",
    )
    threshold_polarity.add_argument(
        "--no-invert",
        dest="invert",
        action="store_false",
        help="Extract light lines on dark background.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=100.0,
        help="Drop contours with an area smaller than this many pixels.",
    )
    parser.add_argument(
        "--smooth-epsilon",
        type=float,
        default=0.002,
        help="Contour smoothing as a fraction of contour perimeter.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        help="Optional directory for stitched, grayscale, and threshold debug PNGs.",
    )
    parser.add_argument(
        "--marker-layout",
        type=Path,
        help=(
            "JSON file mapping ArUco IDs to table-coordinate corners, or a measured "
            "layout with width and height."
        ),
    )
    parser.add_argument(
        "--min-layout-markers",
        type=int,
        default=2,
        help="Minimum known layout markers required to warp a photo in marker-layout mode.",
    )
    parser.add_argument(
        "--max-layout-error",
        type=float,
        default=3.0,
        help="Reject marker-layout photo solves whose mean reprojection error exceeds this many layout units.",
    )
    parser.add_argument(
        "--layout-pixels-per-unit",
        type=float,
        default=40.0,
        help="Raster processing resolution for marker-layout mode. SVG output is scaled back to layout units.",
    )
    marker_orientation = parser.add_mutually_exclusive_group()
    marker_orientation.add_argument(
        "--normalize-marker-orientation",
        dest="normalize_marker_orientation",
        action="store_true",
        default=True,
        help="Rotate disconnected marker groups so detected marker top edges point the same way. This is the default.",
    )
    marker_orientation.add_argument(
        "--no-normalize-marker-orientation",
        dest="normalize_marker_orientation",
        action="store_false",
        help="Keep each disconnected marker group's original photo orientation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = process_images(
        image_paths=args.images,
        output_svg_path=args.output,
        aruco_dictionary=args.aruco_dictionary,
        threshold_mode=ThresholdMode(args.threshold),
        threshold_value=args.threshold_value,
        threshold_offset=args.threshold_offset,
        adaptive_block_size=args.adaptive_block_size,
        adaptive_c=args.adaptive_c,
        invert=args.invert,
        min_area=args.min_area,
        smooth_epsilon=args.smooth_epsilon,
        debug_dir=args.debug_dir,
        normalize_marker_orientation=args.normalize_marker_orientation,
        marker_layout_path=args.marker_layout,
        min_layout_markers=args.min_layout_markers,
        max_layout_error=args.max_layout_error,
        layout_pixels_per_unit=args.layout_pixels_per_unit,
        calibration_object=CalibrationObject(args.calibration_object),
        calibration_width=args.calibration_width,
        calibration_unit=args.calibration_unit,
        calibration_points=args.calibration_points,
        calibration_image_index=args.calibration_image_index,
    )
    print(
        f"Wrote {result.svg_path} with {result.contour_count} contour paths "
        f"from {len(result.image_paths)} image(s)."
    )
    for warning in result.warnings:
        print(f"Warning: {warning}")


def parse_calibration_points(value: str) -> tuple[float, float, float, float]:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Expected four comma-separated numbers: x1,y1,x2,y2.")
    try:
        x1, y1, x2, y2 = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Calibration points must be numbers.") from exc
    return x1, y1, x2, y2


def parse_adaptive_block_size(value: str) -> int:
    try:
        block_size = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--adaptive-block-size must be an integer.") from exc
    if block_size < 3 or block_size % 2 == 0:
        raise argparse.ArgumentTypeError(
            "--adaptive-block-size must be an odd integer greater than or equal to 3."
        )
    return block_size


if __name__ == "__main__":
    main()
