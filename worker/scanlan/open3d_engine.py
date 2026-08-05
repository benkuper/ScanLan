from __future__ import annotations

import hashlib
import math
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

from .compute import (
    ComputeBackend,
    integrate_tsdf,
    merge_surfel_cloud,
    select_compute_backend,
    tensor_odometry,
    tensor_refine_registration,
    tensor_rgbd,
)
from .imu import odometry_rotation_prior
from .io import PhaseData, load_color, load_depth, save_preview
from .realtime import evaluate_depth_alignment
from .stream import StreamCamera


LOCAL_CACHE_VERSION = 7

# Frame-to-frame RGB-D odometry is locally accurate but inevitably drifts over a
# long handheld capture.  Short internal fragments let the already-seen room
# pull the trajectory back into place without exposing sensor-specific tuning in
# the UI.  Keyframe selection is capped at roughly five frames per second. Four
# keyframes therefore keep each rigid local map below one second: short enough
# that a rotating view of a broad wall cannot accumulate into a second surface
# before the accumulated colored room constrains it again.
TRACKING_FRAGMENT_KEYFRAMES = 4
TRACKING_FRAGMENT_MIN_REMAINDER = 2
TRACKING_FRAGMENT_MATCH_FRAMES = 4
TRACKING_FRAGMENT_VOXEL_M = 0.03
LOOP_CLOSURE_MIN_FRAGMENT_GAP = 6
LOOP_CLOSURE_SEARCH_RADIUS_M = 3.0
LOOP_CLOSURE_MAX_CANDIDATES = 2


@dataclass
class LocalPhase:
    source: PhaseData
    frame_indices: list[int]
    camera_to_phase: list[np.ndarray]
    cloud: Any
    tracking_confidence: int
    tracking_detail: str


@dataclass
class PhaseAlignment:
    phase: int
    method: str
    fitness: float
    inlier_rmse_m: float
    source_overlap: float
    target_overlap: float
    color_consistency: float
    up_tilt_degrees: float
    score: int


@dataclass
class TrajectoryStabilization:
    poses: list[np.ndarray]
    fragment_count: int
    weakest_score: int
    maximum_correction_m: float
    relocalization_count: int
    loop_closure_count: int


def _phase_cache_signature(
    phase: PhaseData,
    voxel_size_m: float,
    backend: ComputeBackend,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"local-phase-v{LOCAL_CACHE_VERSION}\0".encode())
    digest.update(f"{voxel_size_m:.8f}\0{backend.key}\0".encode())

    def add_file(path: Path, hash_contents: bool = False) -> None:
        relative = path.relative_to(phase.root).as_posix().encode("utf-8")
        stat = path.stat()
        digest.update(relative)
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
        if hash_contents:
            digest.update(path.read_bytes())

    add_file(phase.root / "phase.json", True)
    add_file(phase.root / "frames.csv", True)
    tracking_path = phase.root / "tracking.jsonl"
    if tracking_path.is_file():
        add_file(tracking_path, True)
    imu = phase.manifest.get("imu")
    if isinstance(imu, dict):
        imu_path = phase.root / str(imu.get("path", "imu.csv"))
        if imu_path.is_file():
            add_file(imu_path, True)
    for frame in phase.frames:
        add_file(frame.depth_path)
        add_file(frame.color_path)
    return digest.hexdigest()[:24]


def _local_cache_path(
    cache_root: Path,
    phase: PhaseData,
    voxel_size_m: float,
    backend: ComputeBackend,
) -> Path:
    signature = _phase_cache_signature(phase, voxel_size_m, backend)
    phase_id = str(phase.manifest.get("id", phase.root.name))
    return cache_root / "local-phases" / f"{phase_id}-{signature}.npz"


def _load_local_phase_cache(
    o3d: Any,
    cache_path: Path,
    phase: PhaseData,
) -> LocalPhase | None:
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            if int(cached["version"]) != LOCAL_CACHE_VERSION:
                return None
            frame_indices = cached["frame_indices"].astype(np.int64).tolist()
            poses = [matrix for matrix in cached["camera_to_phase"].astype(np.float64)]
            points = cached["points"].astype(np.float64)
            colors = cached["colors"].astype(np.float64)
            normals = cached["normals"].astype(np.float64)
            if (
                not frame_indices
                or len(frame_indices) != len(poses)
                or points.ndim != 2
                or points.shape[1:] != (3,)
                or colors.shape != points.shape
                or normals.shape != points.shape
            ):
                return None
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(colors)
            cloud.normals = o3d.utility.Vector3dVector(normals)
            return LocalPhase(
                source=phase,
                frame_indices=[int(value) for value in frame_indices],
                camera_to_phase=poses,
                cloud=cloud,
                tracking_confidence=int(cached["tracking_confidence"]),
                tracking_detail=str(cached["tracking_detail"].item()),
            )
    except (KeyError, OSError, ValueError):
        cache_path.unlink(missing_ok=True)
        return None


def _save_local_phase_cache(cache_path: Path, local: LocalPhase) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    points = np.asarray(local.cloud.points, dtype=np.float32)
    colors = np.asarray(local.cloud.colors, dtype=np.float32)
    normals = np.asarray(local.cloud.normals, dtype=np.float32)
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            version=np.asarray(LOCAL_CACHE_VERSION, dtype=np.int32),
            frame_indices=np.asarray(local.frame_indices, dtype=np.int32),
            camera_to_phase=np.asarray(local.camera_to_phase, dtype=np.float64),
            points=points,
            colors=colors,
            normals=normals,
            tracking_confidence=np.asarray(local.tracking_confidence, dtype=np.int16),
            tracking_detail=np.asarray(local.tracking_detail),
        )
    os.replace(temporary, cache_path)
    phase_id = str(local.source.manifest.get("id", local.source.root.name))
    for stale in cache_path.parent.glob(f"{phase_id}-*.npz"):
        if stale != cache_path:
            stale.unlink(missing_ok=True)


def _import_open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "Open3D is required for real Kinect captures. Install worker/requirements.txt "
            "or run the packaged reconstruction worker."
        ) from error
    return o3d


def _intrinsic(o3d: Any, phase: PhaseData) -> Any:
    camera = phase.camera
    return o3d.camera.PinholeCameraIntrinsic(
        camera.width,
        camera.height,
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
    )


def _rgbd(o3d: Any, phase: PhaseData, frame_index: int) -> Any:
    frame = phase.frames[frame_index]
    color = np.ascontiguousarray(load_color(frame, phase.camera))
    depth = np.ascontiguousarray(load_depth(frame, phase.camera))
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color),
        o3d.geometry.Image(depth),
        depth_scale=phase.camera.depth_scale,
        depth_trunc=phase.camera.max_depth_m,
        convert_rgb_to_intensity=False,
    )


ProgressCallback = Callable[..., None]


def _display_points(cloud: Any, flip_x: bool) -> np.ndarray:
    points = np.asarray(cloud.points, dtype=np.float32).copy()
    if flip_x:
        points[:, 0] *= -1.0
    points[:, 1:] *= -1.0
    return points


def _publish_preview(cloud: Any, path: Path | None, flip_x: bool) -> int:
    if path is None:
        return len(cloud.points)
    # Y-up/-Z-forward is common to every backend. Kinect v2 depth images need
    # the additional X mirror; Azure Kinect and Femto Mega do not.
    points = _display_points(cloud, flip_x)
    colors = np.rint(np.asarray(cloud.colors) * 255.0).clip(0, 255).astype(np.uint8)
    if colors.shape != points.shape:
        colors = np.full(points.shape, 180, dtype=np.uint8)
    save_preview(path, points, colors, limit=30_000)
    return len(points)


def _rotation_degrees(matrix: np.ndarray) -> float:
    cosine = np.clip((np.trace(matrix[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _interpolate_rigid_transform(
    left: np.ndarray,
    right: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Interpolate two rigid corrections while keeping the rotation orthonormal."""
    amount = float(np.clip(fraction, 0.0, 1.0))
    if amount <= 0.0:
        return left.copy()
    if amount >= 1.0:
        return right.copy()
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = (1.0 - amount) * left[:3, 3] + amount * right[:3, 3]

    left_rotation = left[:3, :3]
    relative = left_rotation.T @ right[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-8:
        result[:3, :3] = left_rotation
        return result

    if abs(math.sin(angle)) < 1e-7:
        # The near-180-degree case is not expected for tracking corrections, but
        # SVD projection gives a stable rigid interpolation if corrupt input ever
        # reaches this guard.
        blended = (1.0 - amount) * left_rotation + amount * right[:3, :3]
        u, _, vh = np.linalg.svd(blended)
        rotation = u @ vh
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vh
        result[:3, :3] = rotation
        return result

    axis = np.asarray(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * math.sin(angle))
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    partial_angle = angle * amount
    partial = (
        np.eye(3)
        + math.sin(partial_angle) * skew
        + (1.0 - math.cos(partial_angle)) * (skew @ skew)
    )
    result[:3, :3] = left_rotation @ partial
    return result


def _apply_fragment_corrections(
    poses: list[np.ndarray],
    anchor_frames: list[int],
    corrections: list[np.ndarray],
) -> list[np.ndarray]:
    if len(anchor_frames) != len(corrections) or not anchor_frames:
        raise ValueError("Trajectory correction anchors are incomplete")
    if anchor_frames[0] != 0 or any(
        right <= left for left, right in zip(anchor_frames, anchor_frames[1:])
    ):
        raise ValueError("Trajectory correction anchors must start at zero and increase")

    corrected: list[np.ndarray] = []
    interval = 0
    for frame_index, pose in enumerate(poses):
        while interval + 1 < len(anchor_frames) and frame_index > anchor_frames[interval + 1]:
            interval += 1
        if interval + 1 < len(anchor_frames):
            left_frame = anchor_frames[interval]
            right_frame = anchor_frames[interval + 1]
            fraction = (frame_index - left_frame) / max(right_frame - left_frame, 1)
            correction = _interpolate_rigid_transform(
                corrections[interval], corrections[interval + 1], fraction
            )
        else:
            correction = corrections[-1]
        corrected.append(correction @ pose)
    return corrected


def _captured_poses(phase: PhaseData) -> tuple[list[np.ndarray], int, str] | None:
    kinect_fusion = phase.manifest.get("poseSource") == "kinect_fusion" and all(
        frame.pose is not None for frame in phase.frames
    )
    realtime_tracking = bool(phase.frames) and all(
        frame.source_sequence in phase.tracking_camera_to_world for frame in phase.frames
    )
    if kinect_fusion:
        raw = [np.asarray(frame.pose, dtype=np.float64) for frame in phase.frames]
        source_label = "Kinect Fusion"
    elif realtime_tracking:
        raw = [
            np.asarray(phase.tracking_camera_to_world[frame.source_sequence], dtype=np.float64)
            for frame in phase.frames
        ]
        source_label = "realtime RGB-D"
    else:
        return None
    if any(pose.shape != (4, 4) or not np.isfinite(pose).all() for pose in raw):
        return None
    try:
        origin_inverse = np.linalg.inv(raw[0])
    except np.linalg.LinAlgError:
        return None
    poses = [origin_inverse @ pose for pose in raw]
    camera = StreamCamera(
        phase.camera.width,
        phase.camera.height,
        phase.camera.fx,
        phase.camera.fy,
        phase.camera.cx,
        phase.camera.cy,
        phase.camera.depth_scale,
        0.1,
        phase.camera.max_depth_m,
    )
    previous_depth = load_depth(phase.frames[0], phase.camera) if kinect_fusion else None
    depth_overlaps: list[float] = []
    depth_errors_mm: list[float] = []
    path_length = 0.0
    max_speed = 0.0
    max_angular_speed = 0.0
    for index, pose in enumerate(poses):
        rotation = pose[:3, :3]
        if (
            not np.isfinite(pose).all()
            or abs(np.linalg.det(rotation) - 1.0) > 0.035
            or np.linalg.norm(rotation.T @ rotation - np.eye(3)) > 0.06
        ):
            return None
        if index == 0:
            continue
        elapsed = max(
            (phase.frames[index].timestamp_us - phase.frames[index - 1].timestamp_us) / 1_000_000.0,
            1.0 / 30.0,
        )
        relative = np.linalg.inv(poses[index - 1]) @ pose
        distance = float(np.linalg.norm(relative[:3, 3]))
        angle = _rotation_degrees(relative)
        path_length += distance
        max_speed = max(max_speed, distance / elapsed)
        max_angular_speed = max(max_angular_speed, angle / elapsed)
        if previous_depth is not None:
            current_depth = load_depth(phase.frames[index], phase.camera)
            previous_to_current = np.linalg.inv(pose) @ poses[index - 1]
            quality = evaluate_depth_alignment(
                previous_depth,
                current_depth,
                camera,
                previous_to_current,
                depth_threshold_m=0.06,
                minimum_samples=min(
                    350,
                    max(40, phase.camera.width * phase.camera.height // 80),
                ),
            )
            if not quality.accepted:
                return None
            depth_overlaps.append(quality.overlap)
            depth_errors_mm.append(quality.rmse_m * 1000.0)
            previous_depth = current_depth
    duration = max(
        (phase.frames[-1].timestamp_us - phase.frames[0].timestamp_us) / 1_000_000.0,
        0.1,
    )
    average_speed = path_length / duration
    if max_speed > 2.4 or average_speed > 1.35 or max_angular_speed > 190.0:
        return None
    confidence = round(
        96
        - min(max_speed / 2.4, 1.0) * 12
        - min(average_speed / 1.35, 1.0) * 8
        - min(max_angular_speed / 190.0, 1.0) * 5
    )
    quality_suffix = ""
    if realtime_tracking and phase.tracking_quality:
        overlap = phase.tracking_quality.get("meanOverlap")
        error_mm = phase.tracking_quality.get("meanDepthRmseMm")
        metrics = []
        if overlap is not None:
            metrics.append(f"{overlap * 100:.0f}% mean overlap")
        if error_mm is not None:
            metrics.append(f"{error_mm:.1f} mm mean depth error")
        if metrics:
            quality_suffix = "; " + ", ".join(metrics)
    elif depth_overlaps:
        quality_suffix = (
            f"; {np.mean(depth_overlaps) * 100:.0f}% mean overlap, "
            f"{np.mean(depth_errors_mm):.1f} mm mean depth error"
        )
    detail = (
        f"Validated {source_label} motion "
        f"({average_speed:.2f} m/s average, {max_speed:.2f} m/s peak{quality_suffix})"
    )
    return poses, max(65, confidence), detail


def _estimate_offline_poses(
    o3d: Any,
    phase: PhaseData,
    voxel_size_m: float,
    backend: ComputeBackend,
    progress: ProgressCallback | None,
) -> tuple[list[np.ndarray], int, str]:
    intrinsic = _intrinsic(o3d, phase)
    poses = [np.eye(4)]
    odometry = np.eye(4)
    previous = None if backend.uses_cuda else _rgbd(o3d, phase, 0)
    previous_tensor = (
        tensor_rgbd(o3d, phase, 0, backend.device) if backend.uses_cuda else None
    )
    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_diff_max = max(voxel_size_m * 3.0, 0.06)
    imu_aided_frames = 0
    camera = StreamCamera(
        phase.camera.width,
        phase.camera.height,
        phase.camera.fx,
        phase.camera.fy,
        phase.camera.cx,
        phase.camera.cy,
        phase.camera.depth_scale,
        0.1,
        phase.camera.max_depth_m,
    )
    previous_depth = load_depth(phase.frames[0], phase.camera)
    for index in range(1, len(phase.frames)):
        current_depth = load_depth(phase.frames[index], phase.camera)

        def alignment_is_credible(candidate: np.ndarray) -> bool:
            quality = evaluate_depth_alignment(
                previous_depth,
                current_depth,
                camera,
                candidate,
                depth_threshold_m=option.depth_diff_max,
                minimum_samples=min(
                    350,
                    max(40, phase.camera.width * phase.camera.height // 80),
                ),
            )
            if not quality.accepted:
                return False
            elapsed = max(
                (
                    phase.frames[index].timestamp_us
                    - phase.frames[index - 1].timestamp_us
                )
                / 1_000_000.0,
                1.0 / 30.0,
            )
            return (
                float(np.linalg.norm(candidate[:3, 3])) / elapsed <= 2.4
                and _rotation_degrees(candidate) / elapsed <= 220.0
            )

        initial = odometry_rotation_prior(
            phase.imu_samples,
            phase.frames[index - 1].timestamp_us,
            phase.frames[index].timestamp_us,
        )
        used_cpu_fallback = False
        success = False
        transformation = np.eye(4)
        current_tensor = None
        if backend.uses_cuda:
            current_tensor = tensor_rgbd(o3d, phase, index, backend.device)
            try:
                transformation = tensor_odometry(
                    o3d,
                    previous_tensor,
                    current_tensor,
                    phase,
                    initial if initial is not None else np.eye(4),
                    option.depth_diff_max,
                    backend,
                )
                success = alignment_is_credible(transformation)
                if initial is not None and success:
                    imu_aided_frames += 1
            except RuntimeError:
                if initial is not None:
                    try:
                        transformation = tensor_odometry(
                            o3d,
                            previous_tensor,
                            current_tensor,
                            phase,
                            np.eye(4),
                            option.depth_diff_max,
                            backend,
                        )
                        success = alignment_is_credible(transformation)
                    except RuntimeError:
                        pass
            if not success:
                used_cpu_fallback = True
                previous = _rgbd(o3d, phase, index - 1)
                current = _rgbd(o3d, phase, index)
        else:
            current = _rgbd(o3d, phase, index)

        if not success:
            success, transformation, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                previous,
                current,
                intrinsic,
                initial if initial is not None else np.eye(4),
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                option,
            )
            success = success and alignment_is_credible(transformation)
            if not success and initial is not None:
                # A bad or temporally incomplete IMU window must never make RGB-D
                # tracking less robust than the original identity initialization.
                success, transformation, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                    previous,
                    current,
                    intrinsic,
                    np.eye(4),
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                    option,
                )
                success = success and alignment_is_credible(transformation)
            elif initial is not None:
                imu_aided_frames += 1
        if not success:
            raise RuntimeError(
                f"Odometry was lost in {phase.manifest['name']} near frame {index}. "
                "Capture more slowly or increase overlap."
            )
        odometry = transformation @ odometry
        poses.append(np.linalg.inv(odometry))
        if backend.uses_cuda:
            previous_tensor = current_tensor
        else:
            previous = current
        previous_depth = current_depth
        if progress:
            progress(
                "Tracking frames",
                f"Tracked frame {index + 1} of {len(phase.frames)} in {phase.manifest['name']}"
                + (" · CPU fallback" if used_cpu_fallback else ""),
                1,
                None,
                (index + 1) / len(phase.frames),
            )
    positions = np.asarray([pose[:3, 3] for pose in poses])
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    duration = max(
        (phase.frames[-1].timestamp_us - phase.frames[0].timestamp_us) / 1_000_000.0,
        0.1,
    )
    extent = float(np.ptp(positions, axis=0).max())
    if path_length / duration > 1.5 or extent > max(6.0, duration * 0.8):
        raise RuntimeError(
            f"Tracking drifted in {phase.manifest['name']} (camera path became physically implausible). "
            "This phase was rejected instead of producing duplicated geometry."
        )
    if imu_aided_frames:
        coverage = imu_aided_frames / max(len(phase.frames) - 1, 1)
        return (
            poses,
            min(78, round(68 + coverage * 10)),
            f"IMU-aided RGB-D odometry passed drift checks ({coverage:.0%} gyro coverage)",
        )
    return poses, 68, "RGB-D odometry passed physical drift checks"


def _select_keyframes(phase: PhaseData, poses: list[np.ndarray]) -> list[int]:
    selected = [0]
    for index in range(1, len(poses)):
        previous_index = selected[-1]
        elapsed = (
            phase.frames[index].timestamp_us - phase.frames[previous_index].timestamp_us
        ) / 1_000_000.0
        relative = np.linalg.inv(poses[previous_index]) @ poses[index]
        moved = float(np.linalg.norm(relative[:3, 3])) >= 0.04
        turned = _rotation_degrees(relative) >= 3.0
        # At most five keyframes per second unless the camera made a large move.
        large_move = float(np.linalg.norm(relative[:3, 3])) >= 0.10 or _rotation_degrees(relative) >= 8.0
        if elapsed >= 0.18 and (moved or turned or elapsed >= 0.8 or large_move):
            selected.append(index)
    if selected[-1] != len(poses) - 1:
        selected.append(len(poses) - 1)
    return selected


def _tracking_fragment_ranges(keyframe_count: int) -> list[tuple[int, int]]:
    starts = list(range(0, keyframe_count, TRACKING_FRAGMENT_KEYFRAMES))
    if (
        len(starts) > 1
        and keyframe_count - starts[-1] < TRACKING_FRAGMENT_MIN_REMAINDER
    ):
        starts.pop()
    return [
        (start, starts[index + 1] if index + 1 < len(starts) else keyframe_count)
        for index, start in enumerate(starts)
    ]


def _tracking_fragment_cloud(
    o3d: Any,
    phase: PhaseData,
    intrinsic: Any,
    frame_indices: list[int],
    camera_to_phase: list[np.ndarray],
    start: int,
    end: int,
) -> Any:
    origin_inverse = np.linalg.inv(camera_to_phase[start])
    sample_count = min(TRACKING_FRAGMENT_MATCH_FRAMES, end - start)
    sample_positions = np.linspace(start, end - 1, sample_count, dtype=np.int64)
    cloud = o3d.geometry.PointCloud()
    for position in np.unique(sample_positions):
        frame_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            _rgbd(o3d, phase, frame_indices[int(position)]),
            intrinsic,
            project_valid_depth_only=True,
        )
        if len(frame_cloud.points) > 150_000:
            frame_cloud = frame_cloud.uniform_down_sample(
                math.ceil(len(frame_cloud.points) / 150_000)
            )
        frame_cloud = frame_cloud.voxel_down_sample(0.035)
        frame_cloud.transform(origin_inverse @ camera_to_phase[int(position)])
        cloud += frame_cloud
    cloud = cloud.voxel_down_sample(TRACKING_FRAGMENT_VOXEL_M)
    if not len(cloud.points):
        raise RuntimeError(
            f"No usable depth remained while stabilizing {phase.manifest['name']}"
        )
    return cloud


def _trajectory_alignment_acceptable(
    alignment: PhaseAlignment,
    incremental_correction: np.ndarray,
) -> bool:
    return (
        alignment.fitness >= 0.07
        and alignment.source_overlap >= 0.08
        and alignment.inlier_rmse_m <= 0.045
        and float(np.linalg.norm(incremental_correction[:3, 3])) <= 1.2
        and _rotation_degrees(incremental_correction) <= 35.0
    )


def _prefer_alignment(candidate: PhaseAlignment, current: PhaseAlignment) -> bool:
    return candidate.score > current.score or (
        candidate.score == current.score
        and candidate.inlier_rmse_m < current.inlier_rmse_m
    )


def _fragment_information(
    o3d: Any,
    source: Any,
    target: Any,
    transformation: np.ndarray,
) -> Any:
    return o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source,
        target,
        0.08,
        transformation,
    )


def _optimize_fragment_pose_graph(
    o3d: Any,
    phase: PhaseData,
    fragments: list[Any],
    initial_poses: list[np.ndarray],
    ranges: list[tuple[int, int]],
    frame_indices: list[int],
    backend: ComputeBackend,
    progress: ProgressCallback | None,
) -> tuple[list[np.ndarray], int]:
    """Add bounded non-local constraints and globally distribute loop error."""
    if len(fragments) < LOOP_CLOSURE_MIN_FRAGMENT_GAP + 1:
        return initial_poses, 0

    pose_graph = o3d.pipelines.registration.PoseGraph()
    for pose in initial_poses:
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.asarray(pose).copy())
        )

    for current in range(1, len(fragments)):
        previous = current - 1
        previous_to_current = (
            np.linalg.inv(initial_poses[current]) @ initial_poses[previous]
        )
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                previous,
                current,
                previous_to_current,
                _fragment_information(
                    o3d,
                    fragments[previous],
                    fragments[current],
                    previous_to_current,
                ),
                False,
            )
        )

    accepted_loops = 0
    for current in range(LOOP_CLOSURE_MIN_FRAGMENT_GAP, len(fragments)):
        if current != len(fragments) - 1 and current % 3:
            continue
        candidates: list[tuple[float, int, np.ndarray]] = []
        current_position = initial_poses[current][:3, 3]
        for previous in range(0, current - LOOP_CLOSURE_MIN_FRAGMENT_GAP + 1):
            distance = float(
                np.linalg.norm(current_position - initial_poses[previous][:3, 3])
            )
            if distance > LOOP_CLOSURE_SEARCH_RADIUS_M:
                continue
            current_to_previous = (
                np.linalg.inv(initial_poses[previous]) @ initial_poses[current]
            )
            prescore = o3d.pipelines.registration.evaluate_registration(
                fragments[current],
                fragments[previous],
                0.12,
                current_to_previous,
            )
            if float(prescore.fitness) >= 0.04:
                candidates.append(
                    (float(prescore.fitness), previous, current_to_previous)
                )

        candidates.sort(reverse=True, key=lambda value: value[0])
        for _, previous, current_to_previous in candidates[:LOOP_CLOSURE_MAX_CANDIDATES]:
            if progress:
                progress(
                    "Stabilizing trajectory",
                    f"Testing loop closure near frame {frame_indices[ranges[current][0]]} "
                    f"against frame {frame_indices[ranges[previous][0]]}",
                    0,
                    None,
                )
            try:
                refined, source_fine, target_fine = _refine_registration(
                    o3d,
                    fragments[current],
                    fragments[previous],
                    current_to_previous,
                    backend,
                )
            except RuntimeError:
                continue
            quality = _alignment_quality(
                o3d,
                current + 1,
                "loop closure + ICP",
                refined,
                source_fine,
                target_fine,
            )
            correction = refined.transformation @ np.linalg.inv(current_to_previous)
            if not (
                quality.fitness >= 0.12
                and quality.source_overlap >= 0.12
                and quality.inlier_rmse_m <= 0.035
                and quality.color_consistency >= 0.52
                and quality.score >= 55
                and float(np.linalg.norm(correction[:3, 3])) <= 0.75
                and _rotation_degrees(correction) <= 20.0
            ):
                continue

            previous_to_current = np.linalg.inv(refined.transformation)
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    previous,
                    current,
                    previous_to_current,
                    _fragment_information(
                        o3d,
                        fragments[previous],
                        fragments[current],
                        previous_to_current,
                    ),
                    True,
                )
            )
            accepted_loops += 1

    if not accepted_loops:
        return initial_poses, 0

    if progress:
        progress(
            "Stabilizing trajectory",
            f"Globally optimizing {phase.manifest['name']} across {len(initial_poses)} local maps with "
            f"{accepted_loops} loop closure{'s' if accepted_loops != 1 else ''}",
            0,
            None,
        )
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.06,
        edge_prune_threshold=0.25,
        preference_loop_closure=0.2,
        reference_node=0,
    )
    try:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )
    except RuntimeError:
        return initial_poses, 0
    optimized = [np.asarray(node.pose).copy() for node in pose_graph.nodes]
    deltas = [
        candidate @ np.linalg.inv(initial)
        for candidate, initial in zip(optimized, initial_poses, strict=True)
    ]
    if (
        not all(np.isfinite(value).all() for value in optimized)
        or max(float(np.linalg.norm(value[:3, 3])) for value in deltas) > 1.5
        or max(_rotation_degrees(value) for value in deltas) > 30.0
    ):
        if progress:
            progress(
                "Stabilizing trajectory",
                "Rejected an excessive global loop correction; keeping the validated local trajectory",
                0,
                None,
            )
        return initial_poses, 0
    return optimized, accepted_loops


def _stabilize_offline_trajectory(
    o3d: Any,
    phase: PhaseData,
    poses: list[np.ndarray],
    frame_indices: list[int],
    backend: ComputeBackend,
    progress: ProgressCallback | None,
) -> TrajectoryStabilization | None:
    ranges = _tracking_fragment_ranges(len(frame_indices))
    if len(ranges) < 2:
        return None

    selected_poses = [poses[frame_index] for frame_index in frame_indices]
    intrinsic = _intrinsic(o3d, phase)
    fragments: list[Any] = []
    for fragment_index, (start, end) in enumerate(ranges):
        if progress:
            progress(
                "Stabilizing trajectory",
                f"Building local map {fragment_index + 1} of {len(ranges)} "
                f"for {phase.manifest['name']}",
                0,
                None,
            )
        fragments.append(
            _tracking_fragment_cloud(
                o3d,
                phase,
                intrinsic,
                frame_indices,
                selected_poses,
                start,
                end,
            )
        )

    fragment_transforms = [selected_poses[ranges[0][0]]]
    accumulated = o3d.geometry.PointCloud(fragments[0])
    accumulated.transform(fragment_transforms[0])
    alignments: list[PhaseAlignment] = []
    relocalization_count = 0
    for fragment_index in range(1, len(ranges)):
        start, _ = ranges[fragment_index]
        previous_start, _ = ranges[fragment_index - 1]
        initial = (
            fragment_transforms[-1]
            @ np.linalg.inv(selected_poses[previous_start])
            @ selected_poses[start]
        )
        if progress:
            progress(
                "Stabilizing trajectory",
                f"Matching local map {fragment_index + 1} of {len(ranges)} "
                "to geometry already seen",
                0,
                len(accumulated.points),
            )
        refined, source_fine, target_fine = _refine_registration(
            o3d,
            fragments[fragment_index],
            accumulated,
            initial,
            backend,
        )
        alignment = _alignment_quality(
            o3d,
            fragment_index + 1,
            "local map + ICP",
            refined,
            source_fine,
            target_fine,
        )
        incremental_correction = refined.transformation @ np.linalg.inv(initial)
        acceptable = _trajectory_alignment_acceptable(
            alignment,
            incremental_correction,
        )
        if not acceptable:
            if progress:
                progress(
                    "Stabilizing trajectory",
                    f"Local ICP was uncertain near frame {frame_indices[start]}; "
                    "trying bounded room relocalization",
                    0,
                    len(accumulated.points),
                )
            try:
                o3d.utility.random.seed(1)
                registration_voxel = 0.08
                source_down, source_features = _preprocess(
                    o3d,
                    fragments[fragment_index],
                    registration_voxel,
                )
                target_down, target_features = _preprocess(
                    o3d,
                    accumulated,
                    registration_voxel,
                )
                coarse = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                    source_down,
                    target_down,
                    source_features,
                    target_features,
                    o3d.pipelines.registration.FastGlobalRegistrationOption(
                        maximum_correspondence_distance=registration_voxel * 1.55,
                        iteration_number=32,
                        maximum_tuple_count=500,
                    ),
                )
                candidate, candidate_source, candidate_target = _refine_registration(
                    o3d,
                    fragments[fragment_index],
                    accumulated,
                    coarse.transformation,
                    backend,
                )
                candidate_alignment = _alignment_quality(
                    o3d,
                    fragment_index + 1,
                    "feature relocalization + ICP",
                    candidate,
                    candidate_source,
                    candidate_target,
                )
                candidate_correction = candidate.transformation @ np.linalg.inv(initial)
                candidate_acceptable = _trajectory_alignment_acceptable(
                    candidate_alignment,
                    candidate_correction,
                )
                if candidate_acceptable and (
                    not acceptable or _prefer_alignment(candidate_alignment, alignment)
                ):
                    refined = candidate
                    alignment = candidate_alignment
                    incremental_correction = candidate_correction
                    acceptable = True
                    relocalization_count += 1
            except RuntimeError:
                # The original confidence-checked ICP result remains the
                # fallback and will be rejected below when it is unsafe.
                pass
        if not acceptable:
            first_frame = frame_indices[start]
            raise RuntimeError(
                f"Trajectory stabilization lost the room in {phase.manifest['name']} "
                f"near frame {first_frame} ({alignment.source_overlap * 100:.0f}% overlap, "
                f"{alignment.inlier_rmse_m * 1000:.0f} mm error). This phase was rejected "
                "instead of producing offset duplicate geometry; move more slowly or keep "
                "more of the previous view visible."
            )

        fragment_transforms.append(refined.transformation)
        alignments.append(alignment)
        aligned = o3d.geometry.PointCloud(fragments[fragment_index])
        aligned.transform(refined.transformation)
        accumulated += aligned
        accumulated = accumulated.voxel_down_sample(TRACKING_FRAGMENT_VOXEL_M)
        if progress:
            progress(
                "Stabilizing trajectory",
                f"Local map {fragment_index + 1} accepted at "
                f"{alignment.score}/100 confidence",
                0,
                len(accumulated.points),
            )

    fragment_transforms, loop_closure_count = _optimize_fragment_pose_graph(
        o3d,
        phase,
        fragments,
        fragment_transforms,
        ranges,
        frame_indices,
        backend,
        progress,
    )
    anchor_frames = [frame_indices[start] for start, _ in ranges]
    corrections = [
        transform @ np.linalg.inv(selected_poses[start])
        for transform, (start, _) in zip(fragment_transforms, ranges, strict=True)
    ]
    corrected_poses = _apply_fragment_corrections(poses, anchor_frames, corrections)
    maximum_correction_m = max(
        float(np.linalg.norm(correction[:3, 3])) for correction in corrections
    )
    return TrajectoryStabilization(
        poses=corrected_poses,
        fragment_count=len(ranges),
        weakest_score=min(value.score for value in alignments),
        maximum_correction_m=maximum_correction_m,
        relocalization_count=relocalization_count,
        loop_closure_count=loop_closure_count,
    )


def estimate_local_phase(
    phase: PhaseData,
    voxel_size_m: float,
    backend: ComputeBackend,
    progress: ProgressCallback | None = None,
    cache_root: Path | None = None,
) -> LocalPhase:
    o3d = _import_open3d()
    cache_path = (
        _local_cache_path(cache_root, phase, voxel_size_m, backend)
        if cache_root is not None
        else None
    )
    if cache_path is not None:
        cached = _load_local_phase_cache(o3d, cache_path, phase)
        if cached is not None:
            if progress:
                cached_units = len(phase.frames) * 3 - len(cached.frame_indices) + 2
                progress(
                    "Loading cache",
                    f"Reused tracking and {len(cached.frame_indices)} fused keyframes for {phase.manifest['name']}",
                    cached_units,
                    len(cached.cloud.points),
                )
            return cached
    captured = _captured_poses(phase)
    if captured is not None:
        poses, tracking_confidence, tracking_detail = captured
        if progress:
            progress(
                "Validating tracking",
                f"Validated {len(poses)} captured camera poses in {phase.manifest['name']}",
                len(poses),
                None,
            )
    else:
        poses, tracking_confidence, tracking_detail = _estimate_offline_poses(
            o3d, phase, voxel_size_m, backend, progress
        )
        # Offline tracking reports every frame after the first.
        if progress:
            progress(
                "Tracking frames",
                f"Established the reference pose for {phase.manifest['name']}",
                1,
                None,
                1 / max(len(phase.frames), 1),
            )

    frame_indices = _select_keyframes(phase, poses)
    skipped = len(phase.frames) - len(frame_indices)
    if progress:
        progress(
            "Selecting keyframes",
            f"Using {len(frame_indices)} of {len(phase.frames)} frames from {phase.manifest['name']}",
            skipped * 2,
            None,
        )

    stabilization = _stabilize_offline_trajectory(
        o3d,
        phase,
        poses,
        frame_indices,
        backend,
        progress,
    )
    if stabilization is not None:
        poses = stabilization.poses
        tracking_confidence = min(
            tracking_confidence,
            max(60, stabilization.weakest_score + 8),
        )
        tracking_detail += (
            f"; stabilized with {stabilization.fragment_count} local maps "
            f"(maximum correction {stabilization.maximum_correction_m:.2f} m)"
        )
        if stabilization.relocalization_count:
            tracking_detail += (
                f"; relocalized {stabilization.relocalization_count} local map"
                + ("s" if stabilization.relocalization_count != 1 else "")
            )
        if stabilization.loop_closure_count:
            tracking_detail += (
                f"; globally optimized {stabilization.loop_closure_count} loop closure"
                + ("s" if stabilization.loop_closure_count != 1 else "")
            )

    # Matching does not benefit from a multi-million-point 5 mm cloud. Build a
    # 10 mm local TSDF for registration; the requested precision is retained in
    # the final global TSDF below.
    local_voxel = max(voxel_size_m, 0.01)
    selected_poses = [poses[frame_index] for frame_index in frame_indices]
    entries = [
        (phase, frame_index, np.linalg.inv(camera_to_phase))
        for frame_index, camera_to_phase in zip(frame_indices, selected_poses, strict=True)
    ]

    def integrated(entry_index: int, repeated: bool) -> None:
        if progress:
            progress(
                "Fusing keyframes",
                f"Integrated keyframe {entry_index + 1} of {len(frame_indices)} in {phase.manifest['name']}",
                0 if repeated else 1,
                None,
                (entry_index + 1) / len(frame_indices),
            )

    def fallback(detail: str) -> None:
        if progress:
            progress(
                "Fusing keyframes",
                f"CUDA fusion unavailable for {phase.manifest['name']}; retrying on CPU · {detail}",
                0,
                None,
            )

    cloud = integrate_tsdf(
        o3d,
        entries,
        local_voxel,
        max(local_voxel * 4.0, 0.04),
        backend,
        integrated,
        fallback,
    ).voxel_down_sample(max(local_voxel * 2.5, 0.025))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=40))
    if progress:
        progress(
            "Fusing keyframes",
            f"Local cloud contains {len(cloud.points):,} points from {len(frame_indices)} keyframes",
            2,
            len(cloud.points),
        )
    local = LocalPhase(
        source=phase,
        frame_indices=frame_indices,
        camera_to_phase=selected_poses,
        cloud=cloud,
        tracking_confidence=tracking_confidence,
        tracking_detail=tracking_detail,
    )
    if cache_path is not None:
        _save_local_phase_cache(cache_path, local)
    return local


def _preprocess(o3d: Any, cloud: Any, voxel_size_m: float) -> tuple[Any, Any]:
    down = cloud.voxel_down_sample(voxel_size_m)
    if len(down.points) > 18_000:
        down = down.uniform_down_sample(math.ceil(len(down.points) / 18_000))
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_m * 2.2, max_nn=30))
    features = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_m * 5.0, max_nn=100),
    )
    return down, features


def _manual_transform(phase: PhaseData) -> np.ndarray | None:
    path = Path(phase.root) / "manual_transform.json"
    if not path.exists():
        return None
    import json

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    matrix = np.asarray(value["toPrevious"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Invalid manual transform in {path}")
    return matrix


def _run_with_feedback(
    work: Callable[[], Any],
    progress: ProgressCallback | None,
    detail: str,
    expected_seconds: int,
    budget: int,
) -> tuple[Any, int]:
    if progress is None:
        return work(), 0
    started = perf_counter()
    advanced = 0
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(work)
        while True:
            try:
                result = future.result(timeout=0.75)
                elapsed = max(0, round(perf_counter() - started))
                progress("Aligning phases", detail, max(0, budget - advanced), None, 1.0, 0, elapsed)
                return result, budget
            except TimeoutError:
                elapsed_float = perf_counter() - started
                elapsed = max(1, round(elapsed_float))
                stage_progress = min(0.92, elapsed_float / max(expected_seconds, 1) * 0.85)
                stage_eta = (
                    max(1, round(expected_seconds - elapsed_float))
                    if elapsed_float < expected_seconds
                    else None
                )
                advance = 1 if advanced < max(0, budget - 1) else 0
                advanced += advance
                progress(
                    "Aligning phases",
                    f"{detail} · {elapsed}s elapsed",
                    advance,
                    None,
                    stage_progress,
                    stage_eta,
                    elapsed,
                )


def _refine_registration(
    o3d: Any,
    source: Any,
    target: Any,
    initial: np.ndarray,
    backend: ComputeBackend,
) -> tuple[Any, Any, Any]:
    if backend.uses_cuda:
        try:
            return tensor_refine_registration(o3d, source, target, initial, backend)
        except RuntimeError:
            # Registration remains fully functional if a particular CUDA kernel
            # or allocation is unavailable for the current capture.
            pass
    transformation = initial
    refined = None
    source_fine = None
    target_fine = None
    # A broad first pass recovers from a coarse feature transform; progressively
    # smaller correspondence windows then prevent the result from settling on a
    # nearby duplicate wall/floor plane.
    for fine_voxel, distance, iterations in [
        (0.06, 0.16, 35),
        (0.035, 0.085, 45),
        (0.025, 0.05, 35),
    ]:
        source_fine = source.voxel_down_sample(fine_voxel)
        target_fine = target.voxel_down_sample(fine_voxel)
        if len(source_fine.points) > 120_000:
            source_fine = source_fine.uniform_down_sample(
                math.ceil(len(source_fine.points) / 120_000)
            )
        if len(target_fine.points) > 120_000:
            target_fine = target_fine.uniform_down_sample(
                math.ceil(len(target_fine.points) / 120_000)
            )
        normal_search = o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(fine_voxel * 3.0, 0.08), max_nn=50
        )
        source_fine.estimate_normals(normal_search)
        target_fine.estimate_normals(normal_search)
        refined = o3d.pipelines.registration.registration_icp(
            source_fine,
            target_fine,
            distance,
            transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations),
        )
        transformation = refined.transformation
    assert refined is not None and source_fine is not None and target_fine is not None
    # Geometry alone is ambiguous in rooms full of parallel planes. A short
    # color-guided pass helps keep a chair matched to the same chair instead of
    # a geometrically similar patch elsewhere in the room.
    try:
        geometric = refined
        colored = o3d.pipelines.registration.registration_colored_icp(
            source_fine,
            target_fine,
            0.05,
            transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(
                lambda_geometric=0.94
            ),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
        )
        if (
            colored.fitness >= geometric.fitness * 0.98
            and colored.inlier_rmse <= geometric.inlier_rmse
        ):
            refined = colored
    except RuntimeError:
        # Some synthetic or depth-only sources do not carry useful colors.
        pass
    return refined, source_fine, target_fine


def _alignment_quality(
    o3d: Any,
    phase_number: int,
    method: str,
    refined: Any,
    source: Any,
    target: Any,
) -> PhaseAlignment:
    transformed = o3d.geometry.PointCloud(source)
    transformed.transform(refined.transformation)
    threshold = 0.06
    source_distances = np.asarray(transformed.compute_point_cloud_distance(target))
    target_distances = np.asarray(target.compute_point_cloud_distance(transformed))
    source_overlap = float(np.mean(source_distances <= threshold)) if source_distances.size else 0.0
    target_overlap = float(np.mean(target_distances <= threshold)) if target_distances.size else 0.0

    color_consistency = 0.5
    pairs = np.asarray(refined.correspondence_set)
    if pairs.size and len(source.colors) and len(target.colors):
        pairs = pairs.reshape(-1, 2)
        if len(pairs) > 8_000:
            pairs = pairs[np.linspace(0, len(pairs) - 1, 8_000, dtype=np.int64)]
        source_colors = np.asarray(source.colors)[pairs[:, 0]]
        target_colors = np.asarray(target.colors)[pairs[:, 1]]
        color_error = float(np.median(np.linalg.norm(source_colors - target_colors, axis=1)))
        color_consistency = float(np.clip(1.0 - color_error / math.sqrt(3.0), 0.0, 1.0))

    fitness_score = min(float(refined.fitness) / 0.50, 1.0)
    source_score = min(source_overlap / 0.45, 1.0)
    target_score = min(target_overlap / 0.20, 1.0)
    rmse_score = float(np.clip(1.0 - float(refined.inlier_rmse) / 0.06, 0.0, 1.0))
    geometry_score = 100.0 * (
        0.25 * fitness_score
        + 0.25 * source_score
        + 0.15 * target_score
        + 0.20 * rmse_score
        + 0.15 * color_consistency
    )
    transformed_up = refined.transformation[:3, 1]
    up_tilt_degrees = math.degrees(
        math.acos(float(np.clip(transformed_up[1] / np.linalg.norm(transformed_up), -1.0, 1.0)))
    )
    orientation_score = float(np.clip(1.0 - max(0.0, up_tilt_degrees - 20.0) / 35.0, 0.0, 1.0))
    score = round(
        geometry_score * 0.85 + orientation_score * 15.0
    )
    return PhaseAlignment(
        phase=phase_number,
        method=method,
        fitness=round(float(refined.fitness), 4),
        inlier_rmse_m=round(float(refined.inlier_rmse), 5),
        source_overlap=round(source_overlap, 4),
        target_overlap=round(target_overlap, 4),
        color_consistency=round(color_consistency, 4),
        up_tilt_degrees=round(up_tilt_degrees, 2),
        score=max(0, min(100, score)),
    )


def _acceptable_alignment(value: PhaseAlignment) -> bool:
    return (
        value.fitness >= 0.10
        and value.source_overlap >= 0.10
        and value.inlier_rmse_m <= 0.04
        and value.up_tilt_degrees <= 45.0
        and value.score >= 48
    )


def align_phases(
    local_phases: list[LocalPhase],
    voxel_size_m: float,
    backend: ComputeBackend,
    progress: ProgressCallback | None = None,
) -> tuple[list[np.ndarray], list[PhaseAlignment]]:
    del voxel_size_m  # Matching resolution is intentionally independent from final output detail.
    o3d = _import_open3d()
    transforms = [np.eye(4)]
    alignments: list[PhaseAlignment] = []
    accumulated = o3d.geometry.PointCloud(local_phases[0].cloud)
    registration_voxel = 0.10
    for index in range(1, len(local_phases)):
        phase_number = index + 1
        # FGR/RANSAC sample feature tuples internally. Fix the seed so rebuilding
        # the same captures produces the same transform and confidence every time.
        o3d.utility.random.seed(1)
        used_budget = 0
        if progress:
            progress(
                "Aligning phases",
                f"Preparing bounded match clouds for phase {phase_number} of {len(local_phases)}",
                0,
                None,
                0.0,
                18,
                0,
            )
        manual = _manual_transform(local_phases[index].source)
        if manual is not None:
            to_global = transforms[index - 1] @ manual
            alignment = PhaseAlignment(phase_number, "manual", 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 100)
        else:
            continuity_initial = (
                transforms[index - 1] @ local_phases[index - 1].camera_to_phase[-1]
            )
            continuity_bundle, spent = _run_with_feedback(
                lambda: _refine_registration(
                    o3d,
                    local_phases[index].cloud,
                    accumulated,
                    continuity_initial,
                    backend,
                ),
                progress,
                f"Testing capture continuity for phase {phase_number}",
                5,
                5,
            )
            used_budget += spent
            continuity_refined, continuity_source, continuity_target = continuity_bundle
            best_result = continuity_refined
            alignment = _alignment_quality(
                o3d,
                phase_number,
                "capture continuity + ICP",
                continuity_refined,
                continuity_source,
                continuity_target,
            )
            (features, spent) = _run_with_feedback(
                lambda: (
                    _preprocess(o3d, local_phases[index].cloud, registration_voxel),
                    _preprocess(o3d, accumulated, registration_voxel),
                ),
                progress,
                f"Computing coarse features for phase {phase_number}",
                5,
                5,
            )
            used_budget += spent
            (source_pair, target_pair) = features
            source_down, source_features = source_pair
            target_down, target_features = target_pair
            distance = registration_voxel * 1.55
            coarse, spent = _run_with_feedback(
                lambda: o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                    source_down,
                    target_down,
                    source_features,
                    target_features,
                    o3d.pipelines.registration.FastGlobalRegistrationOption(
                        maximum_correspondence_distance=distance,
                        iteration_number=32,
                        maximum_tuple_count=500,
                    ),
                ),
                progress,
                f"Fast feature match for phase {phase_number}",
                6,
                6,
            )
            used_budget += spent
            refined_bundle, spent = _run_with_feedback(
                lambda: _refine_registration(
                    o3d,
                    local_phases[index].cloud,
                    accumulated,
                    coarse.transformation,
                    backend,
                ),
                progress,
                f"Refining phase {phase_number} against the accumulated room",
                6,
                5,
            )
            used_budget += spent
            refined, source_fine, target_fine = refined_bundle
            feature_alignment = _alignment_quality(
                o3d, phase_number, "fast + ICP", refined, source_fine, target_fine
            )
            if (
                feature_alignment.score > alignment.score
                or (
                    feature_alignment.score == alignment.score
                    and feature_alignment.inlier_rmse_m < alignment.inlier_rmse_m
                )
            ):
                best_result = refined
                alignment = feature_alignment

            if not _acceptable_alignment(alignment):
                if progress:
                    progress(
                        "Aligning phases",
                        f"Fast match scored {alignment.score}/100; trying capped robust match for phase {phase_number}",
                        0,
                        None,
                        0.0,
                        15,
                        0,
                    )
                robust, spent = _run_with_feedback(
                    lambda: o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                        source_down,
                        target_down,
                        source_features,
                        target_features,
                        True,
                        distance,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                        4,
                        [
                            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance),
                        ],
                        o3d.pipelines.registration.RANSACConvergenceCriteria(2_500, 0.99),
                    ),
                    progress,
                    f"Capped robust match for phase {phase_number}",
                    15,
                    min(6, max(1, 30 - used_budget)),
                )
                used_budget += spent
                robust_bundle, spent = _run_with_feedback(
                    lambda: _refine_registration(
                        o3d,
                        local_phases[index].cloud,
                        accumulated,
                        robust.transformation,
                        backend,
                    ),
                    progress,
                    f"Refining robust match for phase {phase_number}",
                    6,
                    max(1, 30 - used_budget),
                )
                used_budget += spent
                robust_refined, robust_source, robust_target = robust_bundle
                robust_quality = _alignment_quality(
                    o3d,
                    phase_number,
                    "robust + ICP",
                    robust_refined,
                    robust_source,
                    robust_target,
                )
                if (
                    robust_quality.score > alignment.score
                    or (
                        robust_quality.score == alignment.score
                        and robust_quality.inlier_rmse_m < alignment.inlier_rmse_m
                    )
                ):
                    best_result = robust_refined
                    alignment = robust_quality

            if not _acceptable_alignment(alignment):
                raise RuntimeError(
                    f"Phase {phase_number} alignment confidence was only {alignment.score}/100 "
                    f"({alignment.source_overlap * 100:.0f}% overlap, "
                    f"{alignment.inlier_rmse_m * 1000:.0f} mm error). "
                    "It was rejected to prevent duplicated floors or furniture; capture more overlap."
                )
            to_global = best_result.transformation

        transforms.append(to_global)
        alignments.append(alignment)
        aligned = o3d.geometry.PointCloud(local_phases[index].cloud)
        aligned.transform(to_global)
        accumulated += aligned
        accumulated = accumulated.voxel_down_sample(0.025)
        if progress:
            progress(
                "Aligning phases",
                f"Phase {phase_number} accepted at {alignment.score}/100 confidence",
                max(0, 30 - used_budget),
                len(accumulated.points),
                1.0,
                0,
                None,
            )
    return transforms, alignments


def _quality_summary(
    local_phases: list[LocalPhase],
    alignments: list[PhaseAlignment],
) -> dict[str, Any]:
    tracking_scores = [phase.tracking_confidence for phase in local_phases]
    if alignments:
        alignment_scores = [value.score for value in alignments]
        score = round(0.72 * min(alignment_scores) + 0.28 * float(np.mean(tracking_scores)))
    else:
        score = round(float(np.mean(tracking_scores)))
    score = max(0, min(100, score))
    label = "High" if score >= 80 else "Medium" if score >= 60 else "Low"
    frames_used = sum(len(phase.frame_indices) for phase in local_phases)
    frames_captured = sum(len(phase.source.frames) for phase in local_phases)
    if alignments:
        weakest = min(alignments, key=lambda value: value.score)
        detail = (
            f"{label} confidence: weakest phase match {weakest.score}/100, "
            f"{weakest.source_overlap * 100:.0f}% overlap and "
            f"{weakest.inlier_rmse_m * 1000:.0f} mm error; "
            f"used {frames_used} of {frames_captured} frames."
        )
    else:
        detail = (
            f"{label} confidence from validated camera tracking; "
            f"used {frames_used} of {frames_captured} frames."
        )
    return {
        "score": score,
        "label": label,
        "detail": detail,
        "framesUsed": frames_used,
        "framesCaptured": frames_captured,
        "tracking": [
            {
                "phase": index + 1,
                "score": phase.tracking_confidence,
                "detail": phase.tracking_detail,
                "framesUsed": len(phase.frame_indices),
                "framesCaptured": len(phase.source.frames),
            }
            for index, phase in enumerate(local_phases)
        ],
        "phaseMatches": [asdict(value) for value in alignments],
    }


def reconstruct_open3d(
    phases: list[PhaseData],
    voxel_size_m: float,
    progress: ProgressCallback | None = None,
    preview_path: Path | None = None,
    requested_device: str = "auto",
    cache_root: Path | None = None,
    artifact_context: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not phases:
        raise ValueError("At least one capture phase is required")
    flip_x = all(
        phase.manifest.get("sensor", {}).get("kind", "kinect_v2") == "kinect_v2"
        for phase in phases
    )
    o3d = _import_open3d()
    backend = select_compute_backend(o3d, requested_device)
    if progress:
        progress(
            "Preparing",
            f"Compute backend: {backend.label}",
            0,
            None,
            compute_backend=backend.label,
        )
    if cache_root is None and preview_path is not None:
        cache_root = preview_path.parent / "cache"
    local_phases = [
        estimate_local_phase(phase, voxel_size_m, backend, progress, cache_root)
        for phase in phases
    ]
    first_count = _publish_preview(local_phases[0].cloud, preview_path, flip_x)
    if progress:
        progress(
            "Previewing build",
            "Showing the reconstructed reference phase",
            0,
            first_count,
            1 / len(local_phases),
        )
    phase_to_global, alignments = align_phases(
        local_phases, voxel_size_m, backend, progress
    )

    if artifact_context is not None:
        from .mesh import PosedFrame

        display_axes = (-1.0, -1.0, -1.0) if flip_x else (1.0, -1.0, -1.0)
        artifact_context["posed_frames"] = [
            PosedFrame(
                phase_name=str(local.source.manifest.get("name", f"Phase {phase_index + 1}")),
                phase_id=str(local.source.manifest.get("id", local.source.root.name)),
                source=local.source,
                frame_index=frame_index,
                camera_to_global=phase_transform @ camera_to_phase,
                display_axes=display_axes,
                image_y_up=False,
            )
            for phase_index, (local, phase_transform) in enumerate(
                zip(local_phases, phase_to_global, strict=True)
            )
            for frame_index, camera_to_phase in zip(
                local.frame_indices, local.camera_to_phase, strict=True
            )
        ]

    aligned_preview = o3d.geometry.PointCloud()
    for index, (local, transform) in enumerate(zip(local_phases, phase_to_global, strict=True)):
        phase_cloud = o3d.geometry.PointCloud(local.cloud)
        phase_cloud.transform(transform)
        aligned_preview += phase_cloud
        aligned_preview = aligned_preview.voxel_down_sample(0.02)
        preview_count = _publish_preview(aligned_preview, preview_path, flip_x)
        if progress:
            progress(
                "Previewing build",
                f"Showing {index + 1} of {len(local_phases)} confidence-checked phases",
                0,
                preview_count,
                (index + 1) / len(local_phases),
            )

    total_final_frames = sum(len(local.frame_indices) for local in local_phases)
    needs_mesh = bool(artifact_context and artifact_context.get("needs_mesh"))
    final_fusion_method = "shared_tsdf_cuda" if backend.uses_cuda else "shared_tsdf_cpu"
    if voxel_size_m < 0.01:
        # A room-scale high-detail TSDF can exceed 10 GB even though only its
        # surface is needed. Fuse posed RGB-D surfels into a streaming voxel map
        # instead. CUDA keeps both per-frame reduction and the persistent map on
        # the GPU; CPU retains the original memory-bounded fallback.
        spacing_mm = voxel_size_m * 1000
        detail_label = f"Building {spacing_mm:g} mm cloud"
        entries = []
        entry_details: list[tuple[int, int, str]] = []
        for local, transform in zip(local_phases, phase_to_global, strict=True):
            for selected_number, (frame_index, camera_to_phase) in enumerate(
                zip(local.frame_indices, local.camera_to_phase, strict=True), start=1
            ):
                camera_to_global = transform @ camera_to_phase
                entries.append(
                    (local.source, frame_index, np.linalg.inv(camera_to_global))
                )
                entry_details.append(
                    (
                        selected_number,
                        len(local.frame_indices),
                        local.source.manifest["name"],
                    )
                )

        def merged(entry_index: int, repeated: bool) -> None:
            if progress:
                selected_number, phase_frames, phase_name = entry_details[entry_index]
                progress(
                    detail_label,
                    f"Merged keyframe {entry_index + 1} of {total_final_frames} "
                    f"({selected_number}/{phase_frames} in {phase_name})",
                    0 if repeated else 1,
                    None,
                    (entry_index + 1) / total_final_frames,
                )

        def fallback(detail: str) -> None:
            if progress:
                progress(
                    detail_label,
                    f"CUDA fine-cloud merge unavailable; retrying on CPU · {detail}",
                    0,
                    None,
                )

        cloud = merge_surfel_cloud(
            o3d,
            entries,
            voxel_size_m,
            backend,
            merged,
            fallback,
        )
        if needs_mesh:
            mesh_voxel_size = float(
                artifact_context.get("mesh_voxel_size_m", max(voxel_size_m, 0.008))
            )

            def meshed(entry_index: int, repeated: bool) -> None:
                if progress:
                    progress(
                        "Meshing geometry",
                        f"Integrated mesh keyframe {entry_index + 1} of {total_final_frames}",
                        0,
                        None,
                        (entry_index + 1) / total_final_frames,
                    )

            def mesh_fallback(detail: str) -> None:
                nonlocal final_fusion_method
                final_fusion_method = "shared_tsdf_cpu"
                if progress:
                    progress(
                        "Meshing geometry",
                        f"CUDA mesh fusion unavailable; retrying on CPU · {detail}",
                        0,
                        None,
                    )

            _, fused_mesh = integrate_tsdf(
                o3d,
                entries,
                mesh_voxel_size,
                max(mesh_voxel_size * 4.0, 0.03),
                backend,
                meshed,
                mesh_fallback,
                extract_mesh=True,
            )
            artifact_context["fused_mesh"] = fused_mesh
            artifact_context["fused_mesh_method"] = final_fusion_method
        preview_count = _publish_preview(cloud, preview_path, flip_x)
        if progress:
            progress(
                detail_label,
                f"Preview updated after {len(local_phases)} confidence-checked phases",
                0,
                preview_count,
            )
    else:
        entries = []
        entry_details: list[tuple[int, int, str]] = []
        for local, transform in zip(local_phases, phase_to_global, strict=True):
            for selected_number, (frame_index, camera_to_phase) in enumerate(
                zip(local.frame_indices, local.camera_to_phase, strict=True), start=1
            ):
                camera_to_global = transform @ camera_to_phase
                entries.append(
                    (local.source, frame_index, np.linalg.inv(camera_to_global))
                )
                entry_details.append(
                    (selected_number, len(local.frame_indices), local.source.manifest["name"])
                )

        def integrated(entry_index: int, repeated: bool) -> None:
            if progress:
                selected_number, phase_frames, phase_name = entry_details[entry_index]
                progress(
                    "Building final cloud",
                    f"Integrated keyframe {entry_index + 1} of {total_final_frames} "
                    f"({selected_number}/{phase_frames} in {phase_name})",
                    0 if repeated else 1,
                    None,
                    (entry_index + 1) / total_final_frames,
                )

        def fallback(detail: str) -> None:
            nonlocal final_fusion_method
            final_fusion_method = "shared_tsdf_cpu"
            if progress:
                progress(
                    "Building final cloud",
                    f"CUDA final fusion unavailable; retrying on CPU · {detail}",
                    0,
                    None,
                )

        fused = integrate_tsdf(
            o3d,
            entries,
            voxel_size_m,
            max(voxel_size_m * 4.0, 0.03),
            backend,
            integrated,
            fallback,
            extract_mesh=needs_mesh,
        )
        if needs_mesh:
            cloud, fused_mesh = fused
            artifact_context["fused_mesh"] = fused_mesh
            artifact_context["fused_mesh_method"] = final_fusion_method
        else:
            cloud = fused
        cloud = cloud.voxel_down_sample(voxel_size_m)
    # Statistical cleanup on a multi-million-point 5 mm cloud can take longer
    # than fusion itself. TSDF integration already rejects isolated depth noise;
    # reserve the expensive filter for smaller results.
    if progress:
        progress(
            "Cleaning cloud",
            f"Checking {len(cloud.points):,} fused points for isolated noise",
            0,
            len(cloud.points),
            0.0,
        )
    if 80 < len(cloud.points) <= 2_000_000:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.2)
    points = _display_points(cloud, flip_x)
    colors = np.rint(np.asarray(cloud.colors) * 255.0).clip(0, 255).astype(np.uint8)
    quality = _quality_summary(local_phases, alignments)
    quality["computeBackend"] = backend.label
    quality["localCacheVersion"] = LOCAL_CACHE_VERSION
    if progress:
        progress(
            "Cleaning cloud",
            f"Final cloud contains {len(points):,} points · {quality['score']}/100 confidence",
            4,
            len(points),
            1.0,
        )
    return points, colors, quality
