from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import cv2
import numpy as np


DEFAULT_ARUCO_DICTIONARY = "DICT_4X4_50"


class ThresholdMode(str, Enum):
    FIXED = "fixed"
    OTSU = "otsu"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class ProcessResult:
    svg_path: Path
    image_paths: tuple[Path, ...]
    contour_count: int
    canvas_size: tuple[int, int]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MarkerSet:
    ids: np.ndarray
    centers: np.ndarray
    corners: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class MarkerLink:
    source_index: int
    reference_index: int
    source_to_reference: np.ndarray
    shared_marker_count: int
    inlier_count: int
    reprojection_error: float


@dataclass(frozen=True)
class MarkerLayout:
    markers: dict[int, np.ndarray]
    unit: str | None = None


@dataclass(frozen=True)
class LayoutSolve:
    image_index: int
    matched_marker_ids: tuple[int, ...]
    point_count: int
    inlier_count: int
    mean_error: float | None
    max_error: float | None
    accepted: bool
    reason: str
    transform: np.ndarray | None = None


@dataclass(frozen=True)
class TransformPlan:
    transforms: list[np.ndarray]
    component_ids: list[int]
    links: list[MarkerLink]
    selected_links: list[MarkerLink]
    layout_based: bool = False
    layout_solves: list[LayoutSolve] = None
    svg_unit: str | None = None
    svg_units_per_pixel: float = 1.0


@dataclass(frozen=True)
class StitchResult:
    image: np.ndarray
    marker_sets: list[MarkerSet]
    plan: TransformPlan


def process_images(
    image_paths: Iterable[Path],
    output_svg_path: Path,
    aruco_dictionary: str = DEFAULT_ARUCO_DICTIONARY,
    threshold_mode: ThresholdMode = ThresholdMode.OTSU,
    threshold_value: int = 127,
    invert: bool = True,
    min_area: float = 100.0,
    smooth_epsilon: float = 0.002,
    debug_dir: Path | None = None,
    normalize_marker_orientation: bool = True,
    marker_layout_path: Path | None = None,
    min_layout_markers: int = 2,
    max_layout_error: float = 3.0,
    layout_pixels_per_unit: float = 4.0,
) -> ProcessResult:
    paths = tuple(Path(path) for path in image_paths)
    if not paths:
        raise ValueError("At least one input image is required.")

    images = [_read_image(path) for path in paths]
    marker_layout = read_marker_layout(marker_layout_path) if marker_layout_path is not None else None
    stitch_result = stitch_images(
        images,
        aruco_dictionary,
        normalize_marker_orientation,
        marker_layout,
        min_layout_markers,
        max_layout_error,
        layout_pixels_per_unit,
    )
    stitched = stitch_result.image
    grayscale = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    thresholded = threshold_image(grayscale, threshold_mode, threshold_value, invert)
    contours = find_smoothed_contours(thresholded, min_area, smooth_epsilon)

    output_svg_path.parent.mkdir(parents=True, exist_ok=True)
    write_svg(
        output_svg_path,
        contours,
        stitched.shape[1],
        stitched.shape[0],
        unit=stitch_result.plan.svg_unit,
        coordinate_scale=stitch_result.plan.svg_units_per_pixel,
    )

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "01_stitched.png"), stitched)
        cv2.imwrite(str(debug_dir / "02_grayscale.png"), grayscale)
        cv2.imwrite(str(debug_dir / "03_threshold.png"), thresholded)
        write_debug_report(debug_dir / "stitch-report.txt", paths, images, stitch_result)
        write_marker_debug_images(debug_dir, paths, images, stitch_result.marker_sets)

    return ProcessResult(
        svg_path=output_svg_path,
        image_paths=paths,
        contour_count=len(contours),
        canvas_size=(stitched.shape[1], stitched.shape[0]),
        warnings=build_warnings(stitch_result),
    )


def build_warnings(stitch_result: StitchResult) -> tuple[str, ...]:
    warnings: list[str] = []
    if stitch_result.plan.layout_based:
        rejected = [solve for solve in stitch_result.plan.layout_solves or [] if not solve.accepted]
        if rejected:
            rejected_indexes = ", ".join(str(solve.image_index) for solve in rejected)
            warnings.append(
                f"Marker layout was not usable for image(s): {rejected_indexes}. "
                "Those images were placed as disconnected fallback groups."
            )

    marker_id_sets = [tuple(sorted(markers.ids.tolist())) for markers in stitch_result.marker_sets]
    nonempty_sets = [marker_ids for marker_ids in marker_id_sets if marker_ids]
    if (
        not stitch_result.plan.layout_based
        and len(marker_id_sets) > 1
        and len(nonempty_sets) == len(marker_id_sets)
        and len(set(nonempty_sets)) == 1
    ):
        warnings.append(
            "Every input image has the same detected ArUco marker IDs. "
            "The stitcher will treat those as the same physical anchors and overlay the images."
        )
    if (
        not stitch_result.plan.layout_based
        and len(stitch_result.plan.selected_links) == 0
        and len(stitch_result.marker_sets) > 1
    ):
        warnings.append(
            "No shared ArUco marker links were found between images. "
            "Images were placed as disconnected groups."
        )
    component_count = len(set(stitch_result.plan.component_ids))
    if component_count > 1:
        warnings.append(
            f"Images formed {component_count} disconnected marker groups. "
            "Add shared markers between groups for exact relative placement."
        )
    return tuple(warnings)


def stitch_images(
    images: list[np.ndarray],
    aruco_dictionary: str,
    normalize_marker_orientation: bool = True,
    marker_layout: MarkerLayout | None = None,
    min_layout_markers: int = 2,
    max_layout_error: float = 3.0,
    layout_pixels_per_unit: float = 4.0,
) -> StitchResult:
    marker_sets = [detect_markers(image, aruco_dictionary) for image in images]
    cleaned_images = [erase_markers(image, markers) for image, markers in zip(images, marker_sets)]
    if len(cleaned_images) == 1 and marker_layout is None:
        plan = TransformPlan(
            transforms=[np.eye(3, dtype=np.float64)],
            component_ids=[0],
            links=[],
            selected_links=[],
            layout_solves=[],
        )
        return StitchResult(image=cleaned_images[0], marker_sets=marker_sets, plan=plan)

    if marker_layout is not None:
        plan = build_marker_layout_transforms(
            cleaned_images,
            marker_sets,
            marker_layout,
            min_layout_markers,
            max_layout_error,
            layout_pixels_per_unit,
        )
    else:
        plan = build_global_transforms(cleaned_images, marker_sets, normalize_marker_orientation)

    return StitchResult(
        image=warp_to_canvas(cleaned_images, plan.transforms),
        marker_sets=marker_sets,
        plan=plan,
    )


def detect_markers(image: np.ndarray, dictionary_name: str) -> MarkerSet:
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")

    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is None or len(ids) == 0:
        return MarkerSet(
            ids=np.empty((0,), dtype=np.int32),
            centers=np.empty((0, 2), dtype=np.float32),
            corners=(),
        )

    flat_ids = ids.flatten().astype(np.int32)
    reshaped_corners = tuple(corner.reshape(4, 2).astype(np.float32) for corner in corners)
    centers = np.array([corner.mean(axis=0) for corner in reshaped_corners], dtype=np.float32)
    return MarkerSet(ids=flat_ids, centers=centers, corners=reshaped_corners)


def erase_markers(image: np.ndarray, markers: MarkerSet, padding: float = 6.0) -> np.ndarray:
    cleaned = image.copy()
    for corners in markers.corners:
        center = corners.mean(axis=0)
        expanded = center + (corners - center) * (1.0 + padding / max(1.0, np.linalg.norm(corners[0] - corners[2])))
        polygon = np.round(expanded).astype(np.int32)
        cv2.fillConvexPoly(cleaned, polygon, (255, 255, 255))
    return cleaned


def read_marker_layout(path: Path) -> MarkerLayout:
    data = json.loads(path.read_text(encoding="utf-8"))
    marker_data = data.get("markers", data)
    if not isinstance(marker_data, dict):
        raise ValueError("Marker layout must be a JSON object or contain a 'markers' object.")

    layout: dict[int, np.ndarray] = {}
    for raw_marker_id, raw_corners in marker_data.items():
        corners = np.array(raw_corners, dtype=np.float32)
        if corners.shape != (4, 2):
            raise ValueError(
                f"Marker {raw_marker_id} must have four [x, y] corners in printed marker order."
            )
        layout[int(raw_marker_id)] = corners

    if not layout:
        raise ValueError("Marker layout does not contain any markers.")
    unit = data.get("unit")
    if unit is not None and not isinstance(unit, str):
        raise ValueError("Marker layout 'unit' must be a string, e.g. 'mm'.")
    return MarkerLayout(markers=layout, unit=unit)


def build_marker_layout_transforms(
    images: list[np.ndarray],
    marker_sets: list[MarkerSet],
    marker_layout: MarkerLayout,
    min_layout_markers: int,
    max_layout_error: float,
    layout_pixels_per_unit: float,
) -> TransformPlan:
    transforms: list[np.ndarray | None] = []
    component_ids: list[int | None] = []
    layout_solves: list[LayoutSolve] = []
    fallback_component_id = 1
    scale_to_raster = scale_matrix(layout_pixels_per_unit)

    for image_index, markers in enumerate(marker_sets):
        solve = estimate_layout_transform(
            image_index,
            markers,
            marker_layout,
            min_layout_markers,
            max_layout_error,
        )
        layout_solves.append(solve)
        if not solve.accepted or solve.transform is None:
            transforms.append(np.eye(3, dtype=np.float64))
            component_ids.append(fallback_component_id)
            fallback_component_id += 1
        else:
            transforms.append(scale_to_raster @ solve.transform)
            component_ids.append(0)

    placed_transforms = place_components(images, transforms, component_ids)
    return TransformPlan(
        transforms=placed_transforms,
        component_ids=[_require_component_id(component_id) for component_id in component_ids],
        links=[],
        selected_links=[],
        layout_based=True,
        layout_solves=layout_solves,
        svg_unit=marker_layout.unit,
        svg_units_per_pixel=1.0 / layout_pixels_per_unit,
    )


def estimate_layout_transform(
    image_index: int,
    markers: MarkerSet,
    marker_layout: MarkerLayout,
    min_layout_markers: int,
    max_layout_error: float,
) -> LayoutSolve:
    source_points: list[np.ndarray] = []
    destination_points: list[np.ndarray] = []
    matched_marker_ids: list[int] = []
    for marker_id, corners in zip(markers.ids, markers.corners):
        marker_id_int = int(marker_id)
        layout_corners = marker_layout.markers.get(marker_id_int)
        if layout_corners is None:
            continue
        matched_marker_ids.append(marker_id_int)
        source_points.append(corners)
        destination_points.append(layout_corners)

    if len(matched_marker_ids) < min_layout_markers:
        return LayoutSolve(
            image_index=image_index,
            matched_marker_ids=tuple(sorted(matched_marker_ids)),
            point_count=len(matched_marker_ids) * 4,
            inlier_count=0,
            mean_error=None,
            max_error=None,
            accepted=False,
            reason=f"need at least {min_layout_markers} layout markers",
        )

    src = np.concatenate(source_points).astype(np.float32)
    dst = np.concatenate(destination_points).astype(np.float32)
    if len(src) < 4:
        return LayoutSolve(
            image_index=image_index,
            matched_marker_ids=tuple(sorted(matched_marker_ids)),
            point_count=len(src),
            inlier_count=0,
            mean_error=None,
            max_error=None,
            accepted=False,
            reason="need at least four corner points",
        )

    transform, inliers = cv2.findHomography(src, dst, cv2.RANSAC, max_layout_error)
    if transform is None:
        return LayoutSolve(
            image_index=image_index,
            matched_marker_ids=tuple(sorted(matched_marker_ids)),
            point_count=len(src),
            inlier_count=0,
            mean_error=None,
            max_error=None,
            accepted=False,
            reason="homography solve failed",
        )

    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), transform).reshape(-1, 2)
    distances = np.linalg.norm(projected - dst, axis=1)
    mean_error = float(distances.mean())
    max_error_observed = float(distances.max())
    inlier_count = int(inliers.sum()) if inliers is not None else int(np.count_nonzero(distances <= max_layout_error))

    if mean_error > max_layout_error:
        return LayoutSolve(
            image_index=image_index,
            matched_marker_ids=tuple(sorted(matched_marker_ids)),
            point_count=len(src),
            inlier_count=inlier_count,
            mean_error=mean_error,
            max_error=max_error_observed,
            accepted=False,
            reason=f"mean layout error exceeds {max_layout_error}",
            transform=transform,
        )

    if not transform_preserves_orientation(transform):
        return LayoutSolve(
            image_index=image_index,
            matched_marker_ids=tuple(sorted(matched_marker_ids)),
            point_count=len(src),
            inlier_count=inlier_count,
            mean_error=mean_error,
            max_error=max_error_observed,
            accepted=False,
            reason="homography appears mirrored",
            transform=transform,
        )

    return LayoutSolve(
        image_index=image_index,
        matched_marker_ids=tuple(sorted(matched_marker_ids)),
        point_count=len(src),
        inlier_count=inlier_count,
        mean_error=mean_error,
        max_error=max_error_observed,
        accepted=True,
        reason="accepted",
        transform=transform,
    )


def build_global_transforms(
    images: list[np.ndarray],
    marker_sets: list[MarkerSet],
    normalize_marker_orientation: bool = True,
) -> TransformPlan:
    links = find_marker_links(marker_sets)
    forest, selected_links = build_alignment_forest(len(images), links)
    transforms: list[np.ndarray | None] = [None] * len(images)
    component_ids: list[int | None] = [None] * len(images)
    component_id = 0

    for root_index in range(len(images)):
        if transforms[root_index] is not None:
            continue

        transforms[root_index] = np.eye(3, dtype=np.float64)
        component_ids[root_index] = component_id
        pending = {root_index}

        while pending:
            reference_index = pending.pop()
            reference_transform = transforms[reference_index]
            if reference_transform is None:
                continue

            for source_index, edge_transform in forest[reference_index]:
                source_transform = transforms[source_index]
                if source_transform is not None:
                    continue

                transforms[source_index] = reference_transform @ edge_transform
                component_ids[source_index] = component_id
                pending.add(source_index)

        component_id += 1

    if normalize_marker_orientation:
        transforms = normalize_components_by_marker_orientation(marker_sets, transforms, component_ids)

    placed_transforms = place_components(images, transforms, component_ids)
    return TransformPlan(
        transforms=placed_transforms,
        component_ids=[_require_component_id(component_id) for component_id in component_ids],
        links=links,
        selected_links=selected_links,
    )


def find_marker_links(marker_sets: list[MarkerSet]) -> list[MarkerLink]:
    links: list[MarkerLink] = []
    for reference_index in range(len(marker_sets)):
        for source_index in range(reference_index + 1, len(marker_sets)):
            link = estimate_marker_link(
                source_index,
                marker_sets[source_index],
                reference_index,
                marker_sets[reference_index],
            )
            if link is None:
                continue
            links.append(link)
    return links


def estimate_marker_link(
    source_index: int,
    source: MarkerSet,
    reference_index: int,
    reference: MarkerSet,
) -> MarkerLink | None:
    if len(source.ids) == 0 or len(reference.ids) == 0:
        return None

    source_by_id = {int(marker_id): corners for marker_id, corners in zip(source.ids, source.corners)}
    reference_by_id = {
        int(marker_id): corners for marker_id, corners in zip(reference.ids, reference.corners)
    }
    common_ids = sorted(source_by_id.keys() & reference_by_id.keys())
    if not common_ids:
        return None

    src_points = np.concatenate([source_by_id[marker_id] for marker_id in common_ids]).astype(
        np.float32
    )
    dst_points = np.concatenate([reference_by_id[marker_id] for marker_id in common_ids]).astype(
        np.float32
    )
    if len(src_points) < 4:
        return None

    transform, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    if transform is None:
        return None

    projected = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), transform).reshape(-1, 2)
    distances = np.linalg.norm(projected - dst_points, axis=1)
    inlier_count = int(np.count_nonzero(distances <= 5.0))
    reprojection_error = float(distances.mean())

    return MarkerLink(
        source_index=source_index,
        reference_index=reference_index,
        source_to_reference=transform,
        shared_marker_count=len(common_ids),
        inlier_count=inlier_count,
        reprojection_error=reprojection_error,
    )


def build_alignment_forest(
    image_count: int,
    links: list[MarkerLink],
) -> tuple[list[list[tuple[int, np.ndarray]]], list[MarkerLink]]:
    parents = list(range(image_count))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    forest: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(image_count)]
    sorted_links = sorted(
        links,
        key=lambda link: (
            -link.shared_marker_count,
            -link.inlier_count,
            link.reprojection_error,
            link.reference_index,
            link.source_index,
        ),
    )
    selected_links: list[MarkerLink] = []

    for link in sorted_links:
        source_root = find(link.source_index)
        reference_root = find(link.reference_index)
        if source_root == reference_root:
            continue

        parents[source_root] = reference_root
        selected_links.append(link)
        forest[link.reference_index].append((link.source_index, link.source_to_reference))
        reference_to_source = np.linalg.inv(link.source_to_reference)
        forest[link.source_index].append((link.reference_index, reference_to_source))

    return forest, selected_links


def place_components(
    images: list[np.ndarray],
    transforms: list[np.ndarray | None],
    component_ids: list[int | None],
    gap: int = 40,
) -> list[np.ndarray]:
    placed: list[np.ndarray] = [np.eye(3, dtype=np.float64)] * len(images)
    cursor_x = 0

    for component_id in sorted({component_id for component_id in component_ids if component_id is not None}):
        indices = [index for index, value in enumerate(component_ids) if value == component_id]
        corners = np.concatenate(
            [
                transformed_image_corners(images[index], _require_transform(transforms[index]))
                for index in indices
            ],
            axis=0,
        )
        min_x, min_y = np.floor(corners.min(axis=0)).astype(int)
        max_x = int(np.ceil(corners[:, 0].max()))
        offset = np.array(
            [[1.0, 0.0, float(cursor_x - min_x)], [0.0, 1.0, float(-min_y)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        for index in indices:
            placed[index] = offset @ _require_transform(transforms[index])
        cursor_x += (max_x - min_x) + gap

    return placed


def normalize_components_by_marker_orientation(
    marker_sets: list[MarkerSet],
    transforms: list[np.ndarray | None],
    component_ids: list[int | None],
) -> list[np.ndarray | None]:
    normalized = list(transforms)
    for component_id in sorted({component_id for component_id in component_ids if component_id is not None}):
        indices = [index for index, value in enumerate(component_ids) if value == component_id]
        angles: list[float] = []
        for index in indices:
            transform = _require_transform(transforms[index])
            for corners in marker_sets[index].corners:
                transformed = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), transform).reshape(
                    -1, 2
                )
                top_edge = transformed[1] - transformed[0]
                if np.linalg.norm(top_edge) > 0:
                    angles.append(float(np.arctan2(top_edge[1], top_edge[0])))

        if not angles:
            continue

        angle = circular_mean(angles)
        rotation = rotation_matrix(-angle)
        for index in indices:
            normalized[index] = rotation @ _require_transform(normalized[index])

    return normalized


def circular_mean(angles: list[float]) -> float:
    return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


def rotation_matrix(angle: float) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def scale_matrix(scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("--layout-pixels-per-unit must be greater than zero.")
    return np.array(
        [[float(scale), 0.0, 0.0], [0.0, float(scale), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def transform_preserves_orientation(transform: np.ndarray) -> bool:
    affine_part = transform[:2, :2]
    return float(np.linalg.det(affine_part)) > 0.0


def write_debug_report(
    path: Path,
    image_paths: tuple[Path, ...],
    images: list[np.ndarray],
    stitch_result: StitchResult,
) -> None:
    lines = [
        "Sewing Pattern Stitcher Debug Report",
        "",
        f"stitched_canvas: {stitch_result.image.shape[1]}x{stitch_result.image.shape[0]}",
        f"image_count: {len(image_paths)}",
        f"alignment_mode: {'marker_layout' if stitch_result.plan.layout_based else 'pairwise_marker_links'}",
        "",
        "Warnings",
    ]
    warnings = build_warnings(stitch_result)
    lines.extend(warnings if warnings else ["none"])
    lines.extend(
        [
            "",
            "Images",
        ]
    )

    for index, (image_path, image, markers, component_id) in enumerate(
        zip(image_paths, images, stitch_result.marker_sets, stitch_result.plan.component_ids)
    ):
        marker_ids = ", ".join(str(marker_id) for marker_id in sorted(markers.ids.tolist()))
        if not marker_ids:
            marker_ids = "none"
        lines.extend(
            [
                f"{index}: {image_path}",
                f"  size: {image.shape[1]}x{image.shape[0]}",
                f"  component: {component_id}",
                f"  markers: {marker_ids}",
            ]
        )

    if stitch_result.plan.layout_based:
        lines.extend(["", "Layout Solves"])
        for solve in stitch_result.plan.layout_solves or []:
            matched_ids = ", ".join(str(marker_id) for marker_id in solve.matched_marker_ids)
            if not matched_ids:
                matched_ids = "none"
            mean_error = "n/a" if solve.mean_error is None else f"{solve.mean_error:.3f}"
            max_error = "n/a" if solve.max_error is None else f"{solve.max_error:.3f}"
            lines.extend(
                [
                    f"{solve.image_index}: {'accepted' if solve.accepted else 'rejected'}",
                    f"  reason: {solve.reason}",
                    f"  matched_layout_markers: {matched_ids}",
                    f"  points: {solve.point_count}",
                    f"  inliers: {solve.inlier_count}",
                    f"  mean_error: {mean_error}",
                    f"  max_error: {max_error}",
                ]
            )

    lines.extend(["", "Candidate Links"])
    if stitch_result.plan.links:
        for link in sorted(
            stitch_result.plan.links,
            key=lambda item: (item.reference_index, item.source_index),
        ):
            lines.append(_format_link(link))
    else:
        lines.append("none")

    lines.extend(["", "Selected Links"])
    if stitch_result.plan.selected_links:
        for link in stitch_result.plan.selected_links:
            lines.append(_format_link(link))
    else:
        lines.append("none")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_marker_debug_images(
    debug_dir: Path,
    image_paths: tuple[Path, ...],
    images: list[np.ndarray],
    marker_sets: list[MarkerSet],
) -> None:
    marker_dir = debug_dir / "markers"
    marker_dir.mkdir(parents=True, exist_ok=True)
    for index, (image_path, image, markers) in enumerate(zip(image_paths, images, marker_sets)):
        overlay = image.copy()
        for marker_id, corners, center in zip(markers.ids, markers.corners, markers.centers):
            polygon = np.round(corners).astype(np.int32)
            cv2.polylines(overlay, [polygon], isClosed=True, color=(0, 0, 255), thickness=4)
            cv2.putText(
                overlay,
                str(int(marker_id)),
                tuple(np.round(center).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        output_name = f"{index:02d}_{image_path.stem}_markers.png"
        cv2.imwrite(str(marker_dir / output_name), overlay)


def _format_link(link: MarkerLink) -> str:
    return (
        f"{link.source_index} -> {link.reference_index}: "
        f"shared_markers={link.shared_marker_count}, "
        f"inliers={link.inlier_count}, "
        f"mean_error={link.reprojection_error:.2f}px"
    )


def _require_transform(transform: np.ndarray | None) -> np.ndarray:
    if transform is None:
        raise ValueError("Internal error: image transform was not assigned.")
    return transform


def _require_component_id(component_id: int | None) -> int:
    if component_id is None:
        raise ValueError("Internal error: image component was not assigned.")
    return component_id


def transformed_image_corners(image: np.ndarray, transform: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    corners = np.array(
        [[[0, 0]], [[width, 0]], [[width, height]], [[0, height]]],
        dtype=np.float32,
    )
    return cv2.perspectiveTransform(corners, transform).reshape(-1, 2)


def warp_to_canvas(images: list[np.ndarray], transforms: list[np.ndarray]) -> np.ndarray:
    image_corners = []
    for image, transform in zip(images, transforms):
        image_corners.append(transformed_image_corners(image, transform))

    all_corners = np.concatenate(image_corners, axis=0)
    min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)
    canvas_width = max(1, max_x - min_x)
    canvas_height = max(1, max_y - min_y)

    offset = np.array(
        [[1.0, 0.0, -float(min_x)], [0.0, 1.0, -float(min_y)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

    for image, transform in zip(images, transforms):
        warped = cv2.warpPerspective(
            image,
            offset @ transform,
            (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) < 250
        canvas[mask] = warped[mask]

    return canvas


def threshold_image(
    grayscale: np.ndarray,
    mode: ThresholdMode,
    threshold_value: int,
    invert: bool,
) -> np.ndarray:
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    if mode == ThresholdMode.FIXED:
        _, thresholded = cv2.threshold(grayscale, threshold_value, 255, threshold_type)
    elif mode == ThresholdMode.OTSU:
        _, thresholded = cv2.threshold(grayscale, 0, 255, threshold_type | cv2.THRESH_OTSU)
    elif mode == ThresholdMode.ADAPTIVE:
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        thresholded = cv2.adaptiveThreshold(
            grayscale,
            255,
            adaptive_method,
            threshold_type,
            35,
            5,
        )
    else:
        raise ValueError(f"Unsupported threshold mode: {mode}")

    return thresholded


def find_smoothed_contours(
    thresholded: np.ndarray,
    min_area: float,
    smooth_epsilon: float,
) -> list[np.ndarray]:
    contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    smoothed: list[np.ndarray] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = max(0.5, smooth_epsilon * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(approx) >= 3:
            smoothed.append(approx)

    smoothed.sort(key=cv2.contourArea, reverse=True)
    return smoothed


def write_svg(
    path: Path,
    contours: list[np.ndarray],
    width: int,
    height: int,
    unit: str | None = None,
    coordinate_scale: float = 1.0,
) -> None:
    svg_width = width * coordinate_scale
    svg_height = height * coordinate_scale
    path_data = [_contour_to_path(contour, coordinate_scale) for contour in contours]
    elements = "\n".join(
        f'  <path d="{escape(data)}" fill="none" stroke="black" stroke-width="1"/>'
        for data in path_data
    )
    width_attr = f'{svg_width:.3f}{unit}' if unit else f"{svg_width:.3f}"
    height_attr = f'{svg_height:.3f}{unit}' if unit else f"{svg_height:.3f}"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_attr}" height="{height_attr}" '
        f'viewBox="0 0 {svg_width:.3f} {svg_height:.3f}">\n'
        f"{elements}\n"
        "</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def _contour_to_path(contour: np.ndarray, coordinate_scale: float = 1.0) -> str:
    points = contour.reshape(-1, 2).astype(np.float64) * coordinate_scale
    first_x, first_y = points[0]
    commands = [f"M {first_x:.2f} {first_y:.2f}"]
    commands.extend(f"L {x:.2f} {y:.2f}" for x, y in points[1:])
    commands.append("Z")
    return " ".join(commands)


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image
