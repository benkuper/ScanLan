from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .calibration import rgb_depth_zbuffer, robust_depth_mask, world_from_depth_opencv
from .io import (
    PhaseData,
    frame_rgb_camera,
    frame_rgb_from_depth,
    load_depth,
    load_source_rgb,
    phase_roots,
    read_phase,
    read_project,
    write_json,
)


MAX_LOCALIZATION_REFERENCES = 128
MAX_REFERENCE_IMAGE_DIMENSION = 1280
MAX_RETRIEVAL_CANDIDATES = 20


@dataclass(frozen=True)
class ReferenceFeatures:
    descriptors: np.ndarray
    image_points: np.ndarray
    world_points: np.ndarray
    camera_matrix: np.ndarray
    distortion: np.ndarray | None
    frame_index: int


def write_localization_progress(
    project_root: Path,
    *,
    status: str,
    stage: str,
    detail: str,
    progress: float,
    processed_photos: int,
    total_photos: int,
    localized_photos: int = 0,
    failed_photos: int = 0,
) -> None:
    write_json(
        project_root / "outputs" / "photo-localization-progress.json",
        {
            "schemaVersion": 1,
            "status": status,
            "stage": stage,
            "detail": detail,
            "progress": round(float(np.clip(progress, 0.0, 1.0)), 5),
            "processedPhotos": processed_photos,
            "totalPhotos": total_photos,
            "localizedPhotos": localized_photos,
            "failedPhotos": failed_photos,
        },
    )


def _photo_id(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return hashlib.sha256(str(path).encode("utf-8", errors="replace")).hexdigest()[:16]


def _photo_quality(
    inliers: int,
    two_view_inliers: int,
    rmse: float,
    reference_distance_m: float,
) -> tuple[int, str]:
    score = round(
        100.0
        * (
            0.4 * min(1.0, inliers / 24.0)
            + 0.2 * min(1.0, two_view_inliers / 40.0)
            + 0.3 * max(0.0, 1.0 - rmse / 6.0)
            + 0.1 * max(0.0, 1.0 - reference_distance_m / 2.0)
        )
    )
    label = "Excellent" if score >= 75 else "Good" if score >= 55 else "Usable" if score >= 35 else "Weak"
    return score, label


def _world_from_depth_pose(pose: dict[str, Any], flip_x: bool) -> np.ndarray:
    canonical = pose.get("worldFromDepthCameraOpenCv")
    if canonical is not None:
        return np.asarray(canonical, dtype=np.float64).reshape(4, 4)

    # Pose files produced before supplemental-photo support only stored the
    # viewer transform.  Its display-axis transform is deterministic for each
    # reconstruction backend, so recover the original metric camera pose.
    display = pose.get("matrix")
    if display is None:
        raise ValueError("camera pose has neither a canonical nor display transform")
    image_y_up = bool(pose.get("imageYUp", False))
    display_axes = np.diag(
        [
            -1.0 if flip_x else 1.0,
            1.0 if image_y_up else -1.0,
            -1.0,
            1.0,
        ]
    )
    camera_to_global = display_axes @ np.asarray(display, dtype=np.float64).reshape(4, 4)
    return world_from_depth_opencv(camera_to_global, image_y_up)


def _opencv() -> Any:
    try:
        import cv2
    except Exception as error:
        raise RuntimeError(
            "Supplemental-photo localization requires the packaged OpenCV feature runtime"
        ) from error
    return cv2


def _normalized_photo(path: Path) -> tuple[np.ndarray, int, int, float | None]:
    with Image.open(path) as source:
        exif = source.getexif()
        focal_35mm = exif.get(41989)
        image = ImageOps.exif_transpose(source).convert("RGB")
        rgb = np.asarray(image, dtype=np.uint8)
    focal_pixels = None
    if focal_35mm is not None:
        try:
            focal_pixels = (
                math.hypot(rgb.shape[1], rgb.shape[0])
                * float(focal_35mm)
                / math.hypot(36.0, 24.0)
            )
        except (TypeError, ValueError, ZeroDivisionError):
            focal_pixels = None
    return rgb, rgb.shape[1], rgb.shape[0], focal_pixels


def _scaled_feature_image(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    cv2 = _opencv()
    scale = min(1.0, MAX_REFERENCE_IMAGE_DIMENSION / max(rgb.shape[:2]))
    if scale >= 1.0:
        return rgb, 1.0
    width = max(1, round(rgb.shape[1] * scale))
    height = max(1, round(rgb.shape[0] * scale))
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA), scale


def _retrieval_descriptors(
    phase: PhaseData,
    frame_index: int,
    detector: Any,
) -> np.ndarray | None:
    cv2 = _opencv()
    image, _ = _scaled_feature_image(
        load_source_rgb(phase.frames[frame_index], phase)
    )
    _, descriptors = detector.detectAndCompute(
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None
    )
    return descriptors


def _reference_features(
    phase: PhaseData,
    frame_index: int,
    world_from_depth: np.ndarray,
    sift: Any,
) -> ReferenceFeatures | None:
    cv2 = _opencv()
    frame = phase.frames[frame_index]
    color = load_source_rgb(frame, phase)
    rgb_camera = frame_rgb_camera(frame, phase)
    scale = min(1.0, MAX_REFERENCE_IMAGE_DIMENSION / max(color.shape[:2]))
    if scale < 1.0:
        width = max(1, round(color.shape[1] * scale))
        height = max(1, round(color.shape[0] * scale))
        color = cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA)
    scale_x = color.shape[1] / rgb_camera.width
    scale_y = color.shape[0] / rgb_camera.height
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or not keypoints:
        return None
    image_points = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32)

    raw_depth = load_depth(frame, phase.camera)
    depth = raw_depth.astype(np.float64) / phase.camera.depth_scale
    _, uv_map, visibility = rgb_depth_zbuffer(raw_depth, phase, frame)
    valid = (
        visibility
        & robust_depth_mask(depth)
        & (depth > 0.25)
        & (depth <= phase.camera.max_depth_m)
    )
    depth_y, depth_x = np.nonzero(valid)
    if len(depth_x) < 32:
        return None
    z = depth[depth_y, depth_x]
    camera_points = np.column_stack(
        (
            (depth_x - phase.camera.cx) * z / phase.camera.fx,
            (depth_y - phase.camera.cy) * z / phase.camera.fy,
            z,
        )
    )
    projected_world = (
        camera_points @ world_from_depth[:3, :3].T + world_from_depth[:3, 3]
    )
    projected_uv = uv_map[depth_y, depth_x].astype(np.float64)
    projected_x = np.rint(projected_uv[:, 0] * scale_x).astype(np.int64)
    projected_y = np.rint(projected_uv[:, 1] * scale_y).astype(np.int64)
    inside = (
        (projected_x >= 0)
        & (projected_x < color.shape[1])
        & (projected_y >= 0)
        & (projected_y < color.shape[0])
    )
    projected_x = projected_x[inside]
    projected_y = projected_y[inside]
    projected_world = projected_world[inside]
    point_indices = np.full(color.shape[:2], -1, dtype=np.int32)
    point_indices[projected_y, projected_x] = np.arange(
        len(projected_world), dtype=np.int32
    )
    world_points = np.full((len(keypoints), 3), np.nan, dtype=np.float32)
    radius = 5
    for keypoint_index, (x, y) in enumerate(image_points):
        left = max(0, int(round(float(x))) - radius)
        right = min(color.shape[1], int(round(float(x))) + radius + 1)
        top = max(0, int(round(float(y))) - radius)
        bottom = min(color.shape[0], int(round(float(y))) + radius + 1)
        patch = point_indices[top:bottom, left:right]
        nearby = np.argwhere(patch >= 0)
        if not len(nearby):
            continue
        offsets = nearby.astype(np.float64) + [top - y, left - x]
        nearest = nearby[int(np.argmin(np.sum(offsets * offsets, axis=1)))]
        world_points[keypoint_index] = projected_world[patch[tuple(nearest)]]
    if np.count_nonzero(np.isfinite(world_points).all(axis=1)) < 24:
        return None
    camera_matrix = np.asarray(
        [
            [rgb_camera.fx * scale_x, 0.0, (rgb_camera.cx + 0.5) * scale_x - 0.5],
            [0.0, rgb_camera.fy * scale_y, (rgb_camera.cy + 0.5) * scale_y - 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return ReferenceFeatures(
        np.asarray(descriptors, dtype=np.float32),
        image_points,
        world_points,
        camera_matrix,
        (
            np.asarray(rgb_camera.distortion, dtype=np.float64)
            if rgb_camera.distortion
            else None
        ),
        frame_index,
    )


def _focal_candidates(width: int, height: int, exif_focal: float | None) -> list[float]:
    longest = float(max(width, height))
    if exif_focal is not None and math.isfinite(exif_focal):
        return [exif_focal * scale for scale in np.linspace(0.88, 1.12, 9)]
    # Phone JPEGs commonly omit 35 mm-equivalent focal length. Keep the
    # uncalibrated search within realistic corrected-phone fields of view;
    # very narrow hypotheses can overfit repeated indoor lines and explode
    # the recovered translation scale.
    return [longest * scale for scale in (0.45, 0.55, 0.65, 0.72, 0.78, 0.86, 0.95)]


def _solve_photo_pose(
    world_points: np.ndarray,
    image_points: np.ndarray,
    width: int,
    height: int,
    focal_candidates: list[float],
    *,
    minimum_inliers: int = 20,
    reprojection_error: float = 4.0,
) -> tuple[np.ndarray, float, int, float]:
    cv2 = _opencv()
    best: tuple[int, float, float, np.ndarray, np.ndarray, np.ndarray] | None = None
    center_x = (width - 1) * 0.5
    center_y = (height - 1) * 0.5
    for focal in focal_candidates:
        camera = np.asarray(
            [[focal, 0.0, center_x], [0.0, focal, center_y], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        success, rotation, translation, inliers = cv2.solvePnPRansac(
            world_points,
            image_points,
            camera,
            None,
            iterationsCount=5000,
            reprojectionError=reprojection_error,
            confidence=0.999,
            flags=cv2.SOLVEPNP_AP3P,
        )
        if not success or inliers is None or len(inliers) < minimum_inliers:
            continue
        inlier_indices = inliers.reshape(-1)
        try:
            rotation, translation = cv2.solvePnPRefineLM(
                world_points[inlier_indices],
                image_points[inlier_indices],
                camera,
                None,
                rotation,
                translation,
            )
        except Exception:
            pass
        projected, _ = cv2.projectPoints(
            world_points[inlier_indices], rotation, translation, camera, None
        )
        residual = projected.reshape(-1, 2) - image_points[inlier_indices]
        rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        candidate = (len(inlier_indices), -rmse, focal, rotation, translation, camera)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("could not find a geometrically consistent camera pose")
    inlier_count, negative_rmse, focal, rotation, translation, _ = best
    rotation_matrix, _ = cv2.Rodrigues(rotation)
    camera_from_world = np.eye(4, dtype=np.float64)
    camera_from_world[:3, :3] = rotation_matrix
    camera_from_world[:3, 3] = translation.reshape(3)
    return np.linalg.inv(camera_from_world), float(focal), int(inlier_count), -negative_rmse


def _localization_pose_candidates(camera_poses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [pose for pose in camera_poses if not pose.get("supplementalPhoto")]
    if len(candidates) <= MAX_LOCALIZATION_REFERENCES:
        return candidates
    selected_indices = set(
        np.linspace(0, len(candidates) - 1, MAX_LOCALIZATION_REFERENCES, dtype=np.int64).tolist()
    )
    selected_indices.update(
        index for index, pose in enumerate(candidates) if pose.get("textureFrame")
    )
    return [pose for index, pose in enumerate(candidates) if index in selected_indices]


def _unique_ratio_matches(matcher: Any, photo: np.ndarray, reference: np.ndarray) -> list[Any]:
    pairs = matcher.knnMatch(photo, reference, k=2)
    by_reference: dict[int, Any] = {}
    for pair in pairs:
        if len(pair) != 2 or pair[0].distance >= 0.78 * pair[1].distance:
            continue
        match = pair[0]
        previous = by_reference.get(match.trainIdx)
        if previous is None or match.distance < previous.distance:
            by_reference[match.trainIdx] = match
    return list(by_reference.values())


def _two_view_inliers(
    reference: ReferenceFeatures,
    photo_keypoints: list[Any],
    matches: list[Any],
    width: int,
    height: int,
    focal_candidates: list[float],
) -> tuple[np.ndarray, int]:
    cv2 = _opencv()
    if len(matches) < 10:
        return np.zeros(len(matches), dtype=bool), 0
    reference_points = np.asarray(
        [reference.image_points[match.trainIdx] for match in matches], dtype=np.float32
    ).reshape(-1, 1, 2)
    photo_points = np.asarray(
        [photo_keypoints[match.queryIdx].pt for match in matches], dtype=np.float32
    ).reshape(-1, 1, 2)
    normalized_reference = cv2.undistortPoints(
        reference_points,
        reference.camera_matrix,
        reference.distortion,
    )
    best = np.zeros(len(matches), dtype=bool)
    best_count = 0
    center_x = (width - 1) * 0.5
    center_y = (height - 1) * 0.5
    # A few representative focal hypotheses are sufficient for the epipolar
    # filter; the later metric PnP stage searches the full candidate set.
    focal_indices = sorted({0, len(focal_candidates) // 2, len(focal_candidates) - 1})
    for focal_index in focal_indices:
        focal = focal_candidates[focal_index]
        photo_camera = np.asarray(
            [[focal, 0.0, center_x], [0.0, focal, center_y], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        normalized_photo = cv2.undistortPoints(photo_points, photo_camera, None)
        essential, mask = cv2.findEssentialMat(
            normalized_reference,
            normalized_photo,
            np.eye(3),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=0.004,
            maxIters=3000,
        )
        if essential is None or mask is None:
            continue
        try:
            _, _, _, pose_mask = cv2.recoverPose(
                essential,
                normalized_reference,
                normalized_photo,
                np.eye(3),
                mask=mask,
            )
        except Exception:
            continue
        inliers = pose_mask.reshape(-1) > 0
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best = inliers
            best_count = count
    return best, best_count


def _copy_normalized_photo(project_root: Path, source_path: Path, rgb: np.ndarray) -> Path:
    digest = _photo_id(source_path)
    destination = project_root / "supplemental" / f"{digest}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        Image.fromarray(rgb).save(destination, format="PNG", compress_level=2)
    return destination


def localize_supplemental_photos(project_root: Path, photo_paths: list[Path]) -> dict[str, Any]:
    cv2 = _opencv()
    total_photo_count = len(photo_paths)
    write_localization_progress(
        project_root,
        status="running",
        stage="preparing",
        detail="Reading the RGB-D reconstruction",
        progress=0.0,
        processed_photos=0,
        total_photos=total_photo_count,
    )
    manifest_path = project_root / "supplemental-photos.json"
    existing = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {"schemaVersion": 1, "photos": []}
    )
    photos_by_id = {
        str(photo["id"]): photo
        for photo in existing.get("photos", [])
        if photo.get("id")
    }
    attempts_by_id = {
        str(attempt["id"]): attempt
        for attempt in existing.get("attempts", [])
        if attempt.get("id")
    }
    for photo in photos_by_id.values():
        attempts_by_id.setdefault(
            str(photo["id"]),
            {**photo, "status": "localized"},
        )
    for photo_path in photo_paths:
        queued_path = Path(photo_path)
        queued_id = _photo_id(queued_path)
        if queued_id not in photos_by_id:
            attempts_by_id[queued_id] = {
                "id": queued_id,
                "name": queued_path.stem,
                "path": str(queued_path),
                "sourcePath": str(queued_path),
                "status": "queued",
                "qualityLabel": "Waiting",
            }
    write_json(
        manifest_path,
        {
            "schemaVersion": 1,
            "coordinateConvention": "scanlan_world_opencv_camera_axes",
            "photos": list(photos_by_id.values()),
            "attempts": list(attempts_by_id.values()),
        },
    )
    pose_path = project_root / "outputs" / "camera-poses.json"
    if not pose_path.is_file():
        raise RuntimeError("Build the RGB-D mesh once before localizing supplemental photos")
    camera_poses = json.loads(pose_path.read_text(encoding="utf-8"))
    selected_candidates = _localization_pose_candidates(camera_poses)
    if not selected_candidates:
        raise RuntimeError("The existing reconstruction has no RGB-D reference cameras")
    project = read_project(project_root)
    phases = {str(phase.manifest.get("id", phase.root.name)): phase for phase in (
        # Reconstruction indexes pose records against the complete captured
        # frame list, including frames rejected by the live tracking journal.
        read_phase(root, include_tracking_rejected=True)
        for root in phase_roots(project_root, project)
    )}
    flip_x = bool(phases) and all(
        phase.manifest.get("sensor", {}).get("kind", "kinect_v2") == "kinect_v2"
        for phase in phases.values()
    )
    retrieval_detector = cv2.AKAZE_create(threshold=0.0003)
    retrieval_descriptors: list[np.ndarray] = []
    reference_sources: list[tuple[PhaseData, dict[str, Any], np.ndarray]] = []
    for pose_index, pose in enumerate(selected_candidates):
        phase = phases.get(str(pose["phaseId"]))
        if phase is None:
            continue
        world_from_depth = _world_from_depth_pose(pose, flip_x)
        descriptors = _retrieval_descriptors(
            phase, int(pose["frameIndex"]), retrieval_detector
        )
        if descriptors is not None and len(descriptors) >= 24:
            retrieval_descriptors.append(descriptors)
            reference_sources.append((phase, pose, world_from_depth))
        if pose_index % 3 == 0 or pose_index + 1 == len(selected_candidates):
            write_localization_progress(
                project_root,
                status="running",
                stage="preparing_references",
                detail=(
                    f"Preparing RGB-D reference view {pose_index + 1} "
                    f"of {len(selected_candidates)}"
                ),
                progress=0.3 * (pose_index + 1) / max(len(selected_candidates), 1),
                processed_photos=0,
                total_photos=total_photo_count,
            )
    if not retrieval_descriptors:
        raise RuntimeError("RGB-D reference views did not yield stable visual landmarks")
    kaze = cv2.KAZE_create(threshold=0.0003)
    kaze_references: dict[int, ReferenceFeatures | None] = {}

    localized: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    retrieval_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    for photo_index, source_path in enumerate(photo_paths):
        source_path = Path(source_path)
        source_id = _photo_id(source_path)
        source_name = source_path.stem
        if source_id not in photos_by_id:
            attempts_by_id[source_id] = {
                **attempts_by_id.get(source_id, {}),
                "id": source_id,
                "name": source_name,
                "path": str(source_path),
                "sourcePath": str(source_path),
                "status": "localizing",
                "qualityLabel": "Matching",
            }
            write_json(
                manifest_path,
                {
                    "schemaVersion": 1,
                    "coordinateConvention": "scanlan_world_opencv_camera_axes",
                    "photos": list(photos_by_id.values()),
                    "attempts": list(attempts_by_id.values()),
                },
            )
        write_localization_progress(
            project_root,
            status="running",
            stage="matching_photo",
            detail=f"Matching {source_name} ({photo_index + 1} of {total_photo_count})",
            progress=0.3 + 0.7 * photo_index / max(total_photo_count, 1),
            processed_photos=photo_index,
            total_photos=total_photo_count,
            localized_photos=len(localized),
            failed_photos=len(failures),
        )
        try:
            source_path = source_path.resolve(strict=True)
            source_id = _photo_id(source_path)
            source_name = source_path.stem
            rgb, width, height, exif_focal = _normalized_photo(source_path)
            feature_rgb, feature_scale = _scaled_feature_image(rgb)
            feature_height, feature_width = feature_rgb.shape[:2]
            gray = cv2.cvtColor(feature_rgb, cv2.COLOR_RGB2GRAY)
            _, photo_retrieval_descriptors = retrieval_detector.detectAndCompute(gray, None)
            if photo_retrieval_descriptors is None or len(photo_retrieval_descriptors) < 40:
                raise RuntimeError("not enough stable visual features")
            retrieval_scores = [
                len(
                    _unique_ratio_matches(
                        retrieval_matcher,
                        photo_retrieval_descriptors,
                        reference,
                    )
                )
                for reference in retrieval_descriptors
            ]
            candidate_indices = sorted(
                range(len(retrieval_scores)),
                key=lambda index: retrieval_scores[index],
                reverse=True,
            )[:MAX_RETRIEVAL_CANDIDATES]
            if not candidate_indices:
                raise RuntimeError("no RGB-D views shared stable visual features")

            keypoints, descriptors = kaze.detectAndCompute(gray, None)
            if descriptors is None or len(keypoints) < 40:
                raise RuntimeError("not enough stable multiscale visual features")
            descriptors = np.asarray(descriptors, dtype=np.float32)
            focal_candidates = _focal_candidates(
                feature_width,
                feature_height,
                exif_focal * feature_scale if exif_focal is not None else None,
            )
            hypotheses: list[
                tuple[int, int, float, float, np.ndarray, float, int, int]
            ] = []
            strongest_match_count = 0
            for candidate_position, reference_index in enumerate(candidate_indices):
                if candidate_position % 4 == 0:
                    write_localization_progress(
                        project_root,
                        status="running",
                        stage="solving_photo_pose",
                        detail=(
                            f"Validating camera pose for {source_name} "
                            f"({photo_index + 1} of {total_photo_count})"
                        ),
                        progress=(
                            0.3
                            + 0.7
                            * (
                                photo_index
                                + candidate_position / max(len(candidate_indices), 1)
                            )
                            / max(total_photo_count, 1)
                        ),
                        processed_photos=photo_index,
                        total_photos=total_photo_count,
                        localized_photos=len(localized),
                        failed_photos=len(failures),
                    )
                if reference_index not in kaze_references:
                    phase, pose, world_from_depth = reference_sources[reference_index]
                    kaze_references[reference_index] = _reference_features(
                        phase,
                        int(pose["frameIndex"]),
                        world_from_depth,
                        kaze,
                    )
                reference = kaze_references[reference_index]
                if reference is None:
                    continue
                matches = _unique_ratio_matches(
                    matcher, descriptors, reference.descriptors
                )
                strongest_match_count = max(strongest_match_count, len(matches))
                epipolar_inliers, two_view_count = _two_view_inliers(
                    reference,
                    keypoints,
                    matches,
                    feature_width,
                    feature_height,
                    focal_candidates,
                )
                if two_view_count < 12:
                    continue
                depth_backed = np.asarray(
                    [
                        np.isfinite(reference.world_points[match.trainIdx]).all()
                        for match in matches
                    ],
                    dtype=bool,
                )
                metric_indices = np.flatnonzero(epipolar_inliers & depth_backed)
                if len(metric_indices) < 8:
                    continue
                image_points = np.asarray(
                    [keypoints[matches[index].queryIdx].pt for index in metric_indices],
                    dtype=np.float32,
                )
                world_points = np.asarray(
                    [
                        reference.world_points[matches[index].trainIdx]
                        for index in metric_indices
                    ],
                    dtype=np.float32,
                )
                try:
                    world_from_camera, focal, inliers, rmse = _solve_photo_pose(
                        world_points,
                        image_points,
                        feature_width,
                        feature_height,
                        focal_candidates,
                        minimum_inliers=8,
                        reprojection_error=8.0,
                    )
                except RuntimeError:
                    continue
                if (
                    inliers < 8
                    or inliers / len(world_points) < 0.55
                    or rmse > 6.0
                ):
                    continue
                source_phase, _, source_world_from_depth = reference_sources[
                    reference_index
                ]
                source_frame = source_phase.frames[reference.frame_index]
                world_from_reference_rgb = source_world_from_depth @ np.linalg.inv(
                    frame_rgb_from_depth(source_frame, source_phase)
                )
                reference_distance_m = float(
                    np.linalg.norm(
                        world_from_camera[:3, 3]
                        - world_from_reference_rgb[:3, 3]
                    )
                )
                if reference_distance_m > 2.0:
                    continue
                hypotheses.append(
                    (
                        inliers,
                        two_view_count,
                        -rmse,
                        -reference_distance_m,
                        world_from_camera,
                        focal,
                        len(matches),
                        reference.frame_index,
                    )
                )
            if not hypotheses:
                raise RuntimeError(
                    "no camera pose passed geometric validation "
                    f"(best candidate had {strongest_match_count} unique matches)"
                )
            (
                inliers,
                two_view_inliers,
                negative_rmse,
                negative_reference_distance,
                world_from_camera,
                focal,
                match_count,
                reference_frame_index,
            ) = max(hypotheses, key=lambda value: value[:4])
            rmse = -negative_rmse
            reference_distance_m = -negative_reference_distance
            focal /= feature_scale
            quality_score, quality_label = _photo_quality(
                inliers,
                two_view_inliers,
                rmse,
                reference_distance_m,
            )
            destination = _copy_normalized_photo(project_root, source_path, rgb)
            photo_id = destination.stem
            payload = {
                "id": photo_id,
                "name": source_path.stem,
                "path": destination.relative_to(project_root).as_posix(),
                "sourcePath": str(source_path),
                "camera": {
                    "width": width,
                    "height": height,
                    "fx": round(focal, 8),
                    "fy": round(focal, 8),
                    "cx": round((width - 1) * 0.5, 8),
                    "cy": round((height - 1) * 0.5, 8),
                    "model": "pinhole",
                    "distortion": [],
                },
                "worldFromCamera": [
                    round(float(value), 10) for value in world_from_camera.reshape(-1)
                ],
                "matchCount": match_count,
                "twoViewInlierCount": two_view_inliers,
                "inlierCount": inliers,
                "reprojectionRmsePixels": round(rmse, 4),
                "referenceFrameIndex": reference_frame_index,
                "referenceDistanceMeters": round(reference_distance_m, 4),
                "qualityScore": quality_score,
                "qualityLabel": quality_label,
            }
            photos_by_id[photo_id] = payload
            attempts_by_id[photo_id] = {**payload, "status": "localized"}
            localized.append(payload)
        except Exception as error:
            failure = {
                "id": source_id,
                "name": source_name,
                "path": str(source_path),
                "sourcePath": str(source_path),
                "status": "rejected",
                "qualityScore": 0,
                "qualityLabel": "Rejected",
                "error": str(error),
            }
            failures.append(failure)
            if source_id not in photos_by_id:
                attempts_by_id[source_id] = failure
        write_localization_progress(
            project_root,
            status="running",
            stage="localizing_photos",
            detail=(
                f"Processed {photo_index + 1} of {total_photo_count} photos: "
                f"{len(localized)} localized, {len(failures)} rejected"
            ),
            progress=0.3 + 0.7 * (photo_index + 1) / max(total_photo_count, 1),
            processed_photos=photo_index + 1,
            total_photos=total_photo_count,
            localized_photos=len(localized),
            failed_photos=len(failures),
        )
        # Keep the library usable while a long batch is still running. This
        # also lets a reloaded desktop webview recover every completed result.
        write_json(
            manifest_path,
            {
                "schemaVersion": 1,
                "coordinateConvention": "scanlan_world_opencv_camera_axes",
                "photos": list(photos_by_id.values()),
                "attempts": list(attempts_by_id.values()),
            },
        )

    manifest = {
        "schemaVersion": 1,
        "coordinateConvention": "scanlan_world_opencv_camera_axes",
        "photos": list(photos_by_id.values()),
        "attempts": list(attempts_by_id.values()),
    }
    write_json(manifest_path, manifest)
    write_localization_progress(
        project_root,
        status="complete",
        stage="complete",
        detail=(
            f"Localized {len(localized)} of {total_photo_count} photos; "
            f"rejected {len(failures)}"
        ),
        progress=1.0,
        processed_photos=total_photo_count,
        total_photos=total_photo_count,
        localized_photos=len(localized),
        failed_photos=len(failures),
    )
    return {
        "localizedPhotoCount": len(localized),
        "failedPhotoCount": len(failures),
        "localized": localized,
        "failures": failures,
        "manifestPath": str(manifest_path),
    }
