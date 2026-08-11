from __future__ import annotations

import json
import math
import gc
import os
import queue
import struct
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, Callable

import numpy as np

from .compute import ComputeBackend, select_compute_backend
from .live_contract import (
    CoverageSummary,
    LiveFailurePolicy,
    LiveSubmapDescriptor,
    TrackingState,
    contract_status,
    pose_uncertainty,
    tracking_confidence,
    submap_message,
)
from .live_mapping import (
    AdaptiveBudgetController,
    CoverageField,
    SubmapLimits,
    VOXEL_BLOCK_BYTES,
    rotation_degrees as live_rotation_degrees,
    tracking_colors,
)
from .live_loop import (
    LocalAnchorDatabase,
    PoseGraphLoop,
    SubmapPoseGraph,
    interpolate_transform,
    loop_event,
    submap_odometry_information,
    transform_delta,
    verify_loop_candidate,
)
from .stream import (
    LatestFrameQueue,
    RgbdFrame,
    StreamCamera,
    read_rgbd_frame,
    reject_depth_speckles,
)


ENGINE_MAGIC = b"SCANENG1"
ENGINE_VERSION = 1
ENGINE_STATUS = 1
ENGINE_POINTS = 2
ENGINE_MESH = 3
ENGINE_CAMERA_POINTS = 4
ENGINE_COVERAGE = 5
ENGINE_SUBMAPS = 6
ENGINE_COVERAGE_POINTS = 7
ENGINE_TRACKING_POINTS = 8
ENGINE_HEADER = struct.Struct("<8sHHIQ")

POINT_MAGIC = b"K2P1"
MESH_MAGIC = b"K2M2"
MAX_PREVIEW_POINTS = 150_000
MAX_PREVIEW_TRIANGLES = 150_000
TRACKING_ANCHOR_TRANSLATION_M = 0.08
TRACKING_ANCHOR_ROTATION_DEGREES = 6.0
MAX_TRACKING_ANCHORS = 48
RECENT_TRACKING_ANCHORS = 8
RELOCALIZATION_CANDIDATES_PER_FRAME = 4
RECOVERY_CONFIRMATION_FRAMES = 3
RECOVERY_MAX_TRANSLATION_M = 0.15
RECOVERY_MAX_ROTATION_DEGREES = 10.0
RECOVERY_MAX_GYRO_ERROR_DEGREES = 10.0
RECOVERY_CONFIRM_TRANSLATION_M = 0.05
RECOVERY_CONFIRM_ROTATION_DEGREES = 4.0
RECOVERY_PENDING_MAX_SEQUENCE_GAP = 240
MAX_TRACKING_LINEAR_SPEED_M_S = 1.5
MAX_TRACKING_ANGULAR_SPEED_DEG_S = 120.0
TRACKING_MIN_OVERLAP = 0.28
TRACKING_MIN_INLIER_RATIO = 0.65
TRACKING_MAX_RMSE_FRACTION = 0.62
MAPPING_MIN_OVERLAP = 0.50
MAPPING_MIN_INLIER_RATIO = 0.82


@dataclass(frozen=True)
class AlignmentQuality:
    accepted: bool
    overlap: float
    inlier_ratio: float
    rmse_m: float
    correspondence_count: int
    reason: str


@dataclass(frozen=True)
class TrackedFrame:
    frame: RgbdFrame
    world_to_camera: np.ndarray | None
    quality: AlignmentQuality
    integrate: bool
    state: str
    detail: str


@dataclass(frozen=True)
class TrackingAnchor:
    frame: RgbdFrame
    representation: Any
    world_to_camera: np.ndarray


@dataclass(frozen=True)
class PendingRecovery:
    world_to_camera: np.ndarray
    confirmations: int
    sequence: int
    anchor_sequence: int | None


@dataclass(frozen=True)
class ResetLiveMap:
    sequence: int
    timestamp_us: int


@dataclass(frozen=True)
class TrackingStateUpdate:
    sequence: int
    timestamp_us: int
    state: str
    confidence: float


class TrackingJournal:
    """Persist compact tracking decisions without blocking the tracking loop."""

    def __init__(self, root: Path, capacity: int = 512) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "tracking.jsonl"
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(capacity)
        self.dropped = 0
        self.error: str | None = None
        self._thread = threading.Thread(target=self._write, name="tracking-journal", daemon=True)
        self._thread.start()

    def append(self, tracked: TrackedFrame) -> None:
        entry: dict[str, Any] = {
            "schemaVersion": 1,
            "sequence": tracked.frame.sequence,
            "depthTimestampUs": tracked.frame.depth_timestamp_us,
            "state": tracked.state,
            "accepted": tracked.world_to_camera is not None,
            "integrated": tracked.integrate,
            "reason": tracked.quality.reason,
            "overlap": tracked.quality.overlap,
            "inlierRatio": tracked.quality.inlier_ratio,
            "depthRmseMm": (
                tracked.quality.rmse_m * 1000.0
                if math.isfinite(tracked.quality.rmse_m)
                else None
            ),
            "worldToCamera": (
                tracked.world_to_camera.reshape(-1).tolist()
                if tracked.world_to_camera is not None
                else None
            ),
        }
        while True:
            try:
                self._queue.put_nowait(entry)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def close(self) -> None:
        try:
            self._queue.put(None, timeout=2.0)
        except queue.Full:
            # A blocked filesystem must not hold application shutdown forever.
            # Sacrifice one oldest diagnostic only after giving normal flushes
            # enough time to preserve the complete quality-gated trajectory.
            try:
                self._queue.get_nowait()
                self.dropped += 1
                self._queue.put_nowait(None)
            except queue.Empty:
                pass
        self._thread.join(timeout=5.0)

    def _write(self) -> None:
        try:
            with self.path.open("w", encoding="utf-8", newline="\n", buffering=1) as handle:
                while True:
                    entry = self._queue.get()
                    if entry is None:
                        break
                    handle.write(
                        json.dumps(entry, separators=(",", ":"), allow_nan=False) + "\n"
                    )
        except BaseException as error:
            self.error = str(error)


class LoopJournal:
    """Persist sparse loop decisions for production revalidation."""

    def __init__(self, root: Path, capacity: int = 64) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "live_loops.jsonl"
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(capacity)
        self.dropped = 0
        self.error: str | None = None
        self._thread = threading.Thread(target=self._write, name="loop-journal", daemon=True)
        self._thread.start()

    def append(self, event: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.dropped += 1

    def close(self) -> None:
        try:
            self._queue.put(None, timeout=2.0)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self.dropped += 1
                self._queue.put_nowait(None)
            except queue.Empty:
                pass
        self._thread.join(timeout=5.0)

    def _write(self) -> None:
        try:
            with self.path.open("w", encoding="utf-8", newline="\n", buffering=1) as handle:
                while True:
                    event = self._queue.get()
                    if event is None:
                        break
                    handle.write(
                        json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n"
                    )
        except BaseException as error:
            self.error = str(error)


def _rotation_degrees(matrix: np.ndarray) -> float:
    cosine = np.clip((np.trace(matrix[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _recovery_pose_is_credible(
    last_world_to_camera: np.ndarray,
    candidate_world_to_camera: np.ndarray,
    gyro_predicted_world_to_camera: np.ndarray | None,
) -> bool:
    """Reject recovery poses capable of creating a second copy of the scene."""
    relative = candidate_world_to_camera @ np.linalg.inv(last_world_to_camera)
    if (
        float(np.linalg.norm(relative[:3, 3])) > RECOVERY_MAX_TRANSLATION_M
        or _rotation_degrees(relative) > RECOVERY_MAX_ROTATION_DEGREES
    ):
        return False
    if gyro_predicted_world_to_camera is not None:
        gyro_error = candidate_world_to_camera @ np.linalg.inv(
            gyro_predicted_world_to_camera
        )
        if _rotation_degrees(gyro_error) > RECOVERY_MAX_GYRO_ERROR_DEGREES:
            return False
    return True


def evaluate_depth_alignment(
    source_depth: np.ndarray,
    target_depth: np.ndarray,
    camera: StreamCamera,
    source_to_target: np.ndarray,
    *,
    stride: int = 6,
    depth_threshold_m: float = 0.05,
    minimum_samples: int = 350,
) -> AlignmentQuality:
    """Validate a proposed odometry transform with metric depth correspondences.

    The check is backend-independent, so CUDA odometry cannot silently accept a
    low-quality result merely because Open3D returned a finite matrix.
    """

    source = np.asarray(source_depth, dtype=np.float64)
    target = np.asarray(target_depth, dtype=np.float64)
    transform = np.asarray(source_to_target, dtype=np.float64)
    if source.shape != (camera.height, camera.width) or target.shape != source.shape:
        raise ValueError("Depth alignment inputs do not match the stream camera")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        return AlignmentQuality(False, 0.0, 0.0, math.inf, 0, "non-finite transform")

    rows = np.arange(0, camera.height, max(1, stride), dtype=np.int32)
    columns = np.arange(0, camera.width, max(1, stride), dtype=np.int32)
    xx, yy = np.meshgrid(columns, rows)
    source_z = source[yy, xx] / camera.depth_scale
    valid_source = (
        (source_z >= camera.min_depth_m)
        & (source_z <= camera.max_depth_m)
        & np.isfinite(source_z)
    )
    valid_count = int(valid_source.sum())
    if valid_count < minimum_samples:
        return AlignmentQuality(
            False, 0.0, 0.0, math.inf, 0, f"only {valid_count} valid depth samples"
        )

    u = xx[valid_source].astype(np.float64)
    v = yy[valid_source].astype(np.float64)
    z = source_z[valid_source]
    points = np.column_stack(
        ((u - camera.cx) * z / camera.fx, (v - camera.cy) * z / camera.fy, z)
    )
    transformed = points @ transform[:3, :3].T + transform[:3, 3]
    projected_z = transformed[:, 2]
    in_front = projected_z > max(0.05, camera.min_depth_m * 0.25)
    projected_u = np.rint(
        camera.fx * transformed[:, 0] / np.maximum(projected_z, 1e-6) + camera.cx
    ).astype(np.int32)
    projected_v = np.rint(
        camera.fy * transformed[:, 1] / np.maximum(projected_z, 1e-6) + camera.cy
    ).astype(np.int32)
    inside = (
        in_front
        & (projected_u >= 0)
        & (projected_u < camera.width)
        & (projected_v >= 0)
        & (projected_v < camera.height)
    )
    if not np.any(inside):
        return AlignmentQuality(False, 0.0, 0.0, math.inf, 0, "no projected overlap")

    projected_u = projected_u[inside]
    projected_v = projected_v[inside]
    projected_z = projected_z[inside]
    observed_z = target[projected_v, projected_u] / camera.depth_scale
    valid_target = (
        (observed_z >= camera.min_depth_m)
        & (observed_z <= camera.max_depth_m)
        & np.isfinite(observed_z)
    )
    correspondence_count = int(valid_target.sum())
    overlap = correspondence_count / valid_count
    if correspondence_count < minimum_samples // 2:
        return AlignmentQuality(
            False,
            overlap,
            0.0,
            math.inf,
            correspondence_count,
            "insufficient target overlap",
        )

    residual = observed_z[valid_target] - projected_z[valid_target]
    absolute = np.abs(residual)
    inliers = absolute <= depth_threshold_m
    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / correspondence_count
    rmse_m = (
        float(np.sqrt(np.mean(np.square(residual[inliers]))))
        if inlier_count
        else math.inf
    )
    accepted = (
        overlap >= TRACKING_MIN_OVERLAP
        and inlier_ratio >= TRACKING_MIN_INLIER_RATIO
        and rmse_m <= depth_threshold_m * TRACKING_MAX_RMSE_FRACTION
    )
    if overlap < TRACKING_MIN_OVERLAP:
        reason = f"low overlap ({overlap:.0%})"
    elif inlier_ratio < TRACKING_MIN_INLIER_RATIO:
        reason = f"low depth agreement ({inlier_ratio:.0%})"
    elif rmse_m > depth_threshold_m * TRACKING_MAX_RMSE_FRACTION:
        reason = f"high depth residual ({rmse_m * 1000:.0f} mm)"
    else:
        reason = "depth alignment accepted"
    return AlignmentQuality(
        accepted,
        overlap,
        inlier_ratio,
        rmse_m,
        correspondence_count,
        reason,
    )


def _display_positions(values: np.ndarray, mirror_x: bool) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy().reshape((-1, 3))
    if mirror_x and len(result):
        result[:, 0] *= -1.0
    if len(result):
        result[:, 1:] *= -1.0
    return result


def frame_point_cloud(frame: RgbdFrame) -> tuple[np.ndarray, np.ndarray]:
    """Create an immediate viewport fallback from one calibrated RGB-D frame."""
    depth_m = np.asarray(frame.depth, dtype=np.float32) / frame.camera.depth_scale
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= frame.camera.min_depth_m)
        & (depth_m <= frame.camera.max_depth_m)
    )
    rows, columns = np.nonzero(valid)
    if not len(rows):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    z = depth_m[rows, columns]
    points = np.column_stack(
        (
            (columns.astype(np.float32) - frame.camera.cx) * z / frame.camera.fx,
            (rows.astype(np.float32) - frame.camera.cy) * z / frame.camera.fy,
            z,
        )
    ).astype(np.float32, copy=False)
    if frame.color is not None and frame.color.shape[:2] == depth_m.shape:
        colors = np.asarray(frame.color[rows, columns], dtype=np.uint8)
    else:
        colors = np.full((len(points), 3), 180, dtype=np.uint8)
    return _display_positions(points, frame.mirror_x), colors


def _bounded_indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, limit, dtype=np.int64)


def point_packet(
    frame_count: int,
    timestamp_us: int,
    update_fps: float,
    points: np.ndarray,
    colors: np.ndarray,
) -> bytes:
    points = np.asarray(points, dtype="<f4")
    colors = np.asarray(colors, dtype=np.uint8)
    if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Live point positions and colors must both be N x 3")
    indices = _bounded_indices(len(points), MAX_PREVIEW_POINTS)
    records = np.empty(
        len(indices),
        dtype=np.dtype([("position", "<f4", (3,)), ("color", "u1", (3,))], align=False),
    )
    records["position"] = points[indices]
    records["color"] = colors[indices]
    return struct.pack(
        "<4sIQfI",
        POINT_MAGIC,
        int(frame_count),
        int(timestamp_us),
        float(update_fps),
        len(records),
    ) + records.tobytes()


def mesh_packet(
    frame_count: int,
    vertices: np.ndarray,
    colors: np.ndarray,
    triangles: np.ndarray,
    flip_winding: bool,
) -> bytes:
    vertices = np.asarray(vertices, dtype="<f4")
    colors = np.asarray(colors, dtype=np.uint8)
    triangles = np.asarray(triangles, dtype=np.uint32).reshape((-1, 3))
    if vertices.shape != colors.shape or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Live mesh vertices and colors must both be N x 3")
    if len(triangles) > MAX_PREVIEW_TRIANGLES:
        triangles = triangles[_bounded_indices(len(triangles), MAX_PREVIEW_TRIANGLES)]
    if len(triangles):
        used = np.unique(triangles.reshape(-1))
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[used] = np.arange(len(used), dtype=np.int64)
        vertices = vertices[used]
        colors = colors[used]
        triangles = remap[triangles].astype(np.uint32)
    else:
        vertices = vertices[:0]
        colors = colors[:0]
    if flip_winding and len(triangles):
        triangles = triangles[:, [0, 2, 1]]
    return (
        struct.pack("<4sIII", MESH_MAGIC, int(frame_count), len(vertices), triangles.size)
        + vertices.tobytes()
        + colors.tobytes()
        + triangles.astype("<u4").tobytes()
    )


class EngineMessageWriter:
    def __init__(self, output: BinaryIO) -> None:
        self.output = output
        self._lock = threading.Lock()

    def write(self, kind: int, sequence: int, payload: bytes) -> None:
        header = ENGINE_HEADER.pack(
            ENGINE_MAGIC, ENGINE_VERSION, int(kind), len(payload), int(sequence)
        )
        with self._lock:
            self.output.write(header)
            self.output.write(payload)
            self.output.flush()

    def status(self, sequence: int, value: dict[str, Any]) -> None:
        self.write(
            ENGINE_STATUS,
            sequence,
            json.dumps(
                contract_status(
                    value,
                    failure_policy=(
                        LiveFailurePolicy()
                        if value.get("state") == TrackingState.READY.value
                        else None
                    ),
                ),
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8"),
        )

    def contract_message(self, kind: int, sequence: int, value: dict[str, Any]) -> None:
        if kind not in {ENGINE_COVERAGE, ENGINE_SUBMAPS}:
            raise ValueError("Contract JSON messages are coverage or submaps")
        self.write(
            kind,
            sequence,
            json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8"),
        )


def read_engine_message(source: BinaryIO) -> tuple[int, int, bytes]:
    header = source.read(ENGINE_HEADER.size)
    if not header:
        raise EOFError
    if len(header) != ENGINE_HEADER.size:
        raise RuntimeError("Truncated reconstruction-engine message header")
    magic, version, kind, payload_size, sequence = ENGINE_HEADER.unpack(header)
    if magic != ENGINE_MAGIC or version != ENGINE_VERSION:
        raise RuntimeError("Unknown reconstruction-engine protocol")
    payload = bytearray()
    while len(payload) < payload_size:
        chunk = source.read(payload_size - len(payload))
        if not chunk:
            raise RuntimeError("Truncated reconstruction-engine message payload")
        payload.extend(chunk)
    return int(kind), int(sequence), bytes(payload)


def _cpu_rgbd(o3d: Any, frame: RgbdFrame) -> Any:
    color = frame.color
    if color is None:
        color = np.full((frame.camera.height, frame.camera.width, 3), 160, dtype=np.uint8)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color)),
        o3d.geometry.Image(np.ascontiguousarray(frame.depth)),
        depth_scale=frame.camera.depth_scale,
        depth_trunc=frame.camera.max_depth_m,
        convert_rgb_to_intensity=False,
    )


def _tensor_rgbd(o3d: Any, frame: RgbdFrame, device: Any) -> Any:
    color = frame.color
    if color is None:
        color = np.full((frame.camera.height, frame.camera.width, 3), 160, dtype=np.uint8)
    return o3d.t.geometry.RGBDImage(
        o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(color), device=device)),
        o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(frame.depth), device=device)),
        True,
    )


def _intrinsic_matrix(camera: StreamCamera) -> np.ndarray:
    return np.asarray(
        [[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


class RealtimeTracker:
    def __init__(
        self,
        o3d: Any,
        backend: ComputeBackend,
        voxel_size_m: float,
    ) -> None:
        self.o3d = o3d
        self.backend = backend
        self.voxel_size_m = voxel_size_m
        self.camera: StreamCamera | None = None
        self.first_captured_pose: np.ndarray | None = None
        self.world_to_camera = np.eye(4, dtype=np.float64)
        self.previous_frame: RgbdFrame | None = None
        self.previous_rgbd: Any | None = None
        self.previous_tensor: Any | None = None
        self.anchor_database: LocalAnchorDatabase[TrackingAnchor] = LocalAnchorDatabase(
            MAX_TRACKING_ANCHORS, RECENT_TRACKING_ANCHORS
        )
        self.last_integrated_pose: np.ndarray | None = None
        self.last_integrated_timestamp_us = 0
        self.rejected_since_accept = 0
        self.gyro_since_accept = np.eye(4, dtype=np.float64)
        self.gyro_samples_since_accept = 0
        self.pending_recovery: PendingRecovery | None = None

    @property
    def anchors(self) -> list[TrackingAnchor]:
        return self.anchor_database.entries

    @property
    def relocalization_features(self) -> dict[int, tuple[Any, Any]]:
        return self.anchor_database.features

    def _initialize_camera(self, camera: StreamCamera) -> None:
        if self.camera is None:
            self.camera = camera
            return
        if self.camera != camera:
            raise RuntimeError("RGB-D camera calibration changed during an active scan")

    def _initial_guess(self, frame: RgbdFrame) -> np.ndarray:
        # The stream reserves a calibrated gyro delta. Camera workers that do
        # not yet provide it simply omit the flag and use identity.
        quaternion = frame.gyro_delta_xyzw
        if quaternion is None:
            return np.eye(4, dtype=np.float64)
        x, y, z, w = quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)
        rotation = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        guess = np.eye(4, dtype=np.float64)
        guess[:3, :3] = rotation
        return guess

    def _representation(self, frame: RgbdFrame) -> Any:
        return (
            _tensor_rgbd(self.o3d, frame, self.backend.device)
            if self.backend.uses_cuda
            else _cpu_rgbd(self.o3d, frame)
        )

    def _odometry(
        self,
        source: Any,
        current: Any,
        frame: RgbdFrame,
        initial: np.ndarray,
    ) -> tuple[bool, np.ndarray]:
        camera = frame.camera
        if self.backend.uses_cuda:
            result = self.o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
                source,
                current,
                self.o3d.core.Tensor(
                    _intrinsic_matrix(camera),
                    dtype=self.o3d.core.Dtype.Float64,
                    device=self.backend.device,
                ),
                self.o3d.core.Tensor(
                    np.ascontiguousarray(initial),
                    dtype=self.o3d.core.Dtype.Float64,
                    device=self.backend.device,
                ),
                depth_scale=camera.depth_scale,
                depth_max=camera.max_depth_m,
                method=self.o3d.t.pipelines.odometry.Method.Hybrid,
                params=self.o3d.t.pipelines.odometry.OdometryLossParams(
                    depth_outlier_trunc=max(0.05, self.voxel_size_m * 4.0),
                    depth_huber_delta=0.04,
                    intensity_huber_delta=0.10,
                ),
            )
            transformation = result.transformation.cpu().numpy()
            return bool(np.isfinite(transformation).all()), transformation

        intrinsic = self.o3d.camera.PinholeCameraIntrinsic(
            camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy
        )
        option = self.o3d.pipelines.odometry.OdometryOption()
        option.depth_diff_max = max(0.05, self.voxel_size_m * 4.0)
        success, transformation, _ = self.o3d.pipelines.odometry.compute_rgbd_odometry(
            source,
            current,
            intrinsic,
            initial,
            self.o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            option,
        )
        return bool(success), np.asarray(transformation)

    def _remember(self, frame: RgbdFrame, representation: Any) -> None:
        self.previous_frame = frame
        if self.backend.uses_cuda:
            self.previous_tensor = representation
        else:
            self.previous_rgbd = representation

    def _remember_anchor(
        self,
        frame: RgbdFrame,
        representation: Any,
        world_to_camera: np.ndarray,
    ) -> None:
        if self.anchors:
            relative = (
                np.asarray(world_to_camera)
                @ np.linalg.inv(self.anchors[-1].world_to_camera)
            )
            if (
                float(np.linalg.norm(relative[:3, 3]))
                < TRACKING_ANCHOR_TRANSLATION_M
                and _rotation_degrees(relative) < TRACKING_ANCHOR_ROTATION_DEGREES
            ):
                return
        self.anchor_database.add(
            TrackingAnchor(frame, representation, np.asarray(world_to_camera).copy())
        )

    def _relocalization_anchors(self) -> list[TrackingAnchor]:
        previous_sequence = (
            self.previous_frame.sequence if self.previous_frame is not None else None
        )
        pending_sequence = (
            self.pending_recovery.anchor_sequence
            if self.pending_recovery is not None
            else None
        )
        return self.anchor_database.candidates(
            previous_sequence=previous_sequence,
            pending_sequence=pending_sequence,
            limit=RELOCALIZATION_CANDIDATES_PER_FRAME,
        )

    def _feature_geometry(self, frame: RgbdFrame) -> tuple[Any, Any]:
        voxel_size = max(0.06, self.voxel_size_m * 6.0)
        intrinsic = self.o3d.camera.PinholeCameraIntrinsic(
            frame.camera.width,
            frame.camera.height,
            frame.camera.fx,
            frame.camera.fy,
            frame.camera.cx,
            frame.camera.cy,
        )
        cloud = self.o3d.geometry.PointCloud.create_from_depth_image(
            self.o3d.geometry.Image(np.ascontiguousarray(frame.depth)),
            intrinsic,
            depth_scale=frame.camera.depth_scale,
            depth_trunc=frame.camera.max_depth_m,
            project_valid_depth_only=True,
        ).voxel_down_sample(voxel_size)
        if len(cloud.points) < 80:
            raise RuntimeError("not enough geometry for feature relocalization")
        cloud.estimate_normals(
            self.o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2.3,
                max_nn=30,
            )
        )
        features = self.o3d.pipelines.registration.compute_fpfh_feature(
            cloud,
            self.o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 5.0,
                max_nn=80,
            ),
        )
        return cloud, features

    def _feature_relocalization_transform(
        self,
        anchor: TrackingAnchor,
        current_geometry: tuple[Any, Any],
    ) -> np.ndarray | None:
        cached = self.relocalization_features.get(anchor.frame.sequence)
        if cached is None:
            cached = self._feature_geometry(anchor.frame)
            self.relocalization_features[anchor.frame.sequence] = cached
        source_cloud, source_features = cached
        current_cloud, current_features = current_geometry
        voxel_size = max(0.06, self.voxel_size_m * 6.0)
        result = self.o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_cloud,
            current_cloud,
            source_features,
            current_features,
            self.o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=voxel_size * 1.5,
                iteration_number=48,
                maximum_tuple_count=500,
            ),
        )
        transformation = np.asarray(result.transformation, dtype=np.float64)
        return transformation if np.isfinite(transformation).all() else None

    def _should_integrate(self, world_to_camera: np.ndarray, timestamp_us: int) -> bool:
        camera_to_world = np.linalg.inv(world_to_camera)
        if self.last_integrated_pose is None:
            self.last_integrated_pose = camera_to_world
            self.last_integrated_timestamp_us = timestamp_us
            return True
        relative = np.linalg.inv(self.last_integrated_pose) @ camera_to_world
        elapsed = (timestamp_us - self.last_integrated_timestamp_us) / 1_000_000.0
        if (
            float(np.linalg.norm(relative[:3, 3])) < 0.025
            and _rotation_degrees(relative) < 2.0
            and elapsed < 0.5
        ):
            return False
        self.last_integrated_pose = camera_to_world
        self.last_integrated_timestamp_us = timestamp_us
        return True

    def _quality_is_safe_to_integrate(self, quality: AlignmentQuality) -> bool:
        """Apply a stricter gate to irreversible map writes than pose tracking."""
        maximum_rmse_m = max(0.016, min(0.022, self.voxel_size_m * 2.0))
        return (
            quality.accepted
            and quality.overlap >= MAPPING_MIN_OVERLAP
            and quality.inlier_ratio >= MAPPING_MIN_INLIER_RATIO
            and quality.rmse_m <= maximum_rmse_m
        )

    def track(self, frame: RgbdFrame) -> TrackedFrame:
        self._initialize_camera(frame.camera)
        perfect = AlignmentQuality(True, 1.0, 1.0, 0.0, frame.depth.size, "captured pose")
        if frame.camera_to_world is not None:
            captured = np.asarray(frame.camera_to_world, dtype=np.float64)
            rotation = captured[:3, :3]
            if (
                not np.isfinite(captured).all()
                or abs(float(np.linalg.det(rotation)) - 1.0) > 0.04
                or float(np.linalg.norm(rotation.T @ rotation - np.eye(3))) > 0.07
            ):
                return TrackedFrame(frame, None, AlignmentQuality(False, 0, 0, math.inf, 0, "invalid captured pose"), False, TrackingState.SEARCHING.value, "Captured pose was invalid")
            if self.first_captured_pose is None:
                self.first_captured_pose = captured
            camera_to_world = np.linalg.inv(self.first_captured_pose) @ captured
            proposed = np.linalg.inv(camera_to_world)
            quality = perfect
            if self.previous_frame is not None:
                elapsed = max(
                    (frame.depth_timestamp_us - self.previous_frame.depth_timestamp_us) / 1_000_000.0,
                    1 / 30,
                )
                relative = proposed @ np.linalg.inv(self.world_to_camera)
                if (
                    np.linalg.norm(relative[:3, 3]) / elapsed
                    > MAX_TRACKING_LINEAR_SPEED_M_S
                    or _rotation_degrees(relative) / elapsed
                    > MAX_TRACKING_ANGULAR_SPEED_DEG_S
                ):
                    return TrackedFrame(frame, None, AlignmentQuality(False, 0, 0, math.inf, 0, "captured motion jump"), False, TrackingState.SEARCHING.value, "Captured pose jumped beyond physical limits")
                quality = evaluate_depth_alignment(
                    self.previous_frame.depth,
                    frame.depth,
                    frame.camera,
                    relative,
                    depth_threshold_m=max(0.04, self.voxel_size_m * 4.0),
                )
                if not quality.accepted:
                    return TrackedFrame(
                        frame,
                        None,
                        quality,
                        False,
                        TrackingState.SEARCHING.value,
                        f"Kinect Fusion pose rejected: {quality.reason}; return to recently scanned geometry",
                    )
            self.world_to_camera = proposed
            representation = self._representation(frame)
            self._remember(frame, representation)
            integrate = self._quality_is_safe_to_integrate(quality) and self._should_integrate(
                proposed, frame.depth_timestamp_us
            )
            if integrate:
                self._remember_anchor(frame, representation, proposed)
            detail = "Kinect Fusion pose accepted"
            if quality is not perfect:
                detail += (
                    f" · {quality.overlap:.0%} overlap"
                    f" · {quality.rmse_m * 1000:.0f} mm"
                )
            return TrackedFrame(
                frame,
                proposed.copy(),
                quality,
                integrate,
                TrackingState.TRACKING.value,
                detail,
            )

        if self.previous_frame is None:
            depth_m = np.asarray(frame.depth, dtype=np.float32) / frame.camera.depth_scale
            valid_samples = int(
                np.count_nonzero(
                    np.isfinite(depth_m)
                    & (depth_m >= frame.camera.min_depth_m)
                    & (depth_m <= frame.camera.max_depth_m)
                )
            )
            minimum_samples = min(
                350,
                max(40, frame.camera.width * frame.camera.height // 80),
            )
            if valid_samples < minimum_samples:
                return TrackedFrame(
                    frame,
                    None,
                    AlignmentQuality(
                        False,
                        0.0,
                        0.0,
                        math.inf,
                        valid_samples,
                        f"only {valid_samples} valid depth samples",
                    ),
                    False,
                    TrackingState.SEARCHING.value,
                    f"Waiting for usable depth ({valid_samples} samples); aim beyond the camera's minimum range",
                )
            representation = self._representation(frame)
            self._remember(frame, representation)
            integrate = self._quality_is_safe_to_integrate(perfect) and self._should_integrate(
                self.world_to_camera, frame.depth_timestamp_us
            )
            if integrate:
                self._remember_anchor(frame, representation, self.world_to_camera)
            self.rejected_since_accept = 0
            self.gyro_since_accept = np.eye(4, dtype=np.float64)
            self.gyro_samples_since_accept = 0
            self.pending_recovery = None
            return TrackedFrame(frame, self.world_to_camera.copy(), perfect, integrate, TrackingState.TRACKING.value, "RGB-D odometry initialized")

        current = self._representation(frame)
        previous_representation = (
            self.previous_tensor if self.backend.uses_cuda else self.previous_rgbd
        )
        incremental_gyro = self._initial_guess(frame)
        if frame.gyro_delta_xyzw is not None:
            self.gyro_since_accept = incremental_gyro @ self.gyro_since_accept
            self.gyro_samples_since_accept += 1
        initial = self.gyro_since_accept.copy()
        # Track directly from the latest integration keyframe whenever it is
        # older than the immediately preceding frame.  Chaining every 30 Hz
        # frame compounds sub-millimetre errors into visible duplicate walls;
        # a bounded keyframe baseline estimates the whole local motion in one
        # solve while previous-frame tracking remains the fallback.
        source_frame = self.previous_frame
        source_representation = previous_representation
        source_world_to_camera = self.world_to_camera
        keyframe_tracking = False
        if self.anchors and self.anchors[-1].frame.sequence != self.previous_frame.sequence:
            anchor = self.anchors[-1]
            source_frame = anchor.frame
            source_representation = anchor.representation
            source_world_to_camera = anchor.world_to_camera
            initial = initial @ self.world_to_camera @ np.linalg.inv(anchor.world_to_camera)
            keyframe_tracking = True
        attempts = [initial]
        if not np.allclose(initial, np.eye(4), atol=1e-6):
            attempts.append(np.eye(4, dtype=np.float64))
        best: tuple[AlignmentQuality, np.ndarray] | None = None
        for guess in attempts:
            try:
                success, transformation = self._odometry(
                    source_representation, current, frame, guess
                )
            except RuntimeError:
                continue
            if not success:
                continue
            quality = evaluate_depth_alignment(
                source_frame.depth,
                frame.depth,
                frame.camera,
                transformation,
                depth_threshold_m=max(0.04, self.voxel_size_m * 4.0),
            )
            if best is None or (
                quality.inlier_ratio - quality.rmse_m
                > best[0].inlier_ratio - best[0].rmse_m
            ):
                best = quality, transformation
            if quality.accepted:
                break

        accepted: tuple[AlignmentQuality, np.ndarray, str] | None = None
        accepted_anchor_sequence: int | None = None
        if best is not None and best[0].accepted:
            quality, transformation = best
            elapsed = max(
                (frame.depth_timestamp_us - source_frame.depth_timestamp_us)
                / 1_000_000.0,
                1 / 30,
            )
            distance = float(np.linalg.norm(transformation[:3, 3]))
            angle = _rotation_degrees(transformation)
            if (
                distance / elapsed <= MAX_TRACKING_LINEAR_SPEED_M_S
                and angle / elapsed <= MAX_TRACKING_ANGULAR_SPEED_DEG_S
            ):
                accepted = (
                    quality,
                    transformation @ source_world_to_camera,
                    "Keyframe tracking accepted" if keyframe_tracking else "Tracking accepted",
                )
            else:
                best = (
                    AlignmentQuality(
                        False,
                        quality.overlap,
                        quality.inlier_ratio,
                        quality.rmse_m,
                        quality.correspondence_count,
                        "motion exceeded physical limits",
                    ),
                    transformation,
                )

        if accepted is None and keyframe_tracking:
            # Large motion or poor overlap can make the longer keyframe
            # baseline fail even though the adjacent pair is still usable.
            best = None
            previous_initial = self.gyro_since_accept.copy()
            previous_attempts = [previous_initial]
            if not np.allclose(previous_initial, np.eye(4), atol=1e-6):
                previous_attempts.append(np.eye(4, dtype=np.float64))
            for guess in previous_attempts:
                try:
                    success, transformation = self._odometry(
                        previous_representation, current, frame, guess
                    )
                except RuntimeError:
                    continue
                if not success:
                    continue
                quality = evaluate_depth_alignment(
                    self.previous_frame.depth,
                    frame.depth,
                    frame.camera,
                    transformation,
                    depth_threshold_m=max(0.04, self.voxel_size_m * 4.0),
                )
                if best is None or (
                    quality.inlier_ratio - quality.rmse_m
                    > best[0].inlier_ratio - best[0].rmse_m
                ):
                    best = quality, transformation
                if quality.accepted:
                    break
            if best is not None and best[0].accepted:
                quality, transformation = best
                elapsed = max(
                    (frame.depth_timestamp_us - self.previous_frame.depth_timestamp_us)
                    / 1_000_000.0,
                    1 / 30,
                )
                if (
                    float(np.linalg.norm(transformation[:3, 3])) / elapsed
                    <= MAX_TRACKING_LINEAR_SPEED_M_S
                    and _rotation_degrees(transformation) / elapsed
                    <= MAX_TRACKING_ANGULAR_SPEED_DEG_S
                ):
                    accepted = (
                        quality,
                        transformation @ self.world_to_camera,
                        "Previous-frame fallback accepted",
                    )

        relocalization_anchors: list[TrackingAnchor] = []
        if (
            accepted is None
            or self.rejected_since_accept > 0
            or self.pending_recovery is not None
        ):
            relocalized: list[tuple[AlignmentQuality, np.ndarray, int]] = []
            relocalization_anchors = self._relocalization_anchors()
            for anchor in relocalization_anchors:
                try:
                    success, transformation = self._odometry(
                        anchor.representation,
                        current,
                        frame,
                        np.eye(4, dtype=np.float64),
                    )
                except RuntimeError:
                    continue
                if not success:
                    continue
                quality = evaluate_depth_alignment(
                    anchor.frame.depth,
                    frame.depth,
                    frame.camera,
                    transformation,
                    depth_threshold_m=max(0.04, self.voxel_size_m * 4.0),
                )
                candidate_world_to_camera = transformation @ anchor.world_to_camera
                # A global recovery may legitimately be far from the last
                # accepted pose. Bound the solve relative to the saved view,
                # not relative to the stale pose where tracking was lost.
                if (
                    quality.accepted
                    and quality.overlap >= 0.25
                    and quality.inlier_ratio >= 0.62
                    and float(np.linalg.norm(transformation[:3, 3])) <= 0.9
                    and _rotation_degrees(transformation) <= 60.0
                ):
                    relocalized.append(
                        (quality, candidate_world_to_camera, anchor.frame.sequence)
                    )
            if relocalized:
                pending_anchor_sequence = (
                    self.pending_recovery.anchor_sequence
                    if self.pending_recovery is not None
                    else None
                )
                relocalized.sort(
                    key=lambda value: value[0].inlier_ratio - value[0].rmse_m,
                    reverse=True,
                )
                strongest = relocalized[0]
                strong_enough = lambda candidate: (
                    candidate[0].overlap >= 0.60
                    and candidate[0].inlier_ratio >= 0.88
                    and candidate[0].rmse_m
                    <= max(0.014, self.voxel_size_m * 1.4)
                )
                pending_candidate = next(
                    (
                        candidate
                        for candidate in relocalized
                        if candidate[2] == pending_anchor_sequence
                        and strong_enough(candidate)
                    ),
                    None,
                )
                if pending_candidate is not None:
                    strongest = pending_candidate
                strong_single = strong_enough(strongest)
                consensus = False
                for candidate in relocalized[1:]:
                    difference = candidate[1] @ np.linalg.inv(strongest[1])
                    if (
                        float(np.linalg.norm(difference[:3, 3])) <= 0.20
                        and _rotation_degrees(difference) <= 12.0
                    ):
                        consensus = True
                        break
                if strong_single or consensus:
                    accepted = (
                        strongest[0],
                        strongest[1],
                        "Tracking relocalized to a saved capture keyframe",
                    )
                    accepted_anchor_sequence = strongest[2]

        if (
            accepted_anchor_sequence is None
            and relocalization_anchors
            and self.rejected_since_accept >= 5
        ):
            # Projective RGB-D odometry only converges when the recovered view
            # is already close to a saved view. FPFH supplies the missing
            # coarse transform when the user returns with a different angle;
            # full-resolution hybrid odometry and metric depth gates still make
            # the final decision, so a look-alike wall is not admitted merely
            # because its sparse features happen to match.
            feature_matches: list[tuple[AlignmentQuality, np.ndarray, int]] = []
            try:
                current_geometry = self._feature_geometry(frame)
            except RuntimeError:
                current_geometry = None
            if current_geometry is not None:
                for anchor in relocalization_anchors[:2]:
                    try:
                        coarse = self._feature_relocalization_transform(
                            anchor,
                            current_geometry,
                        )
                        if coarse is None:
                            continue
                        success, transformation = self._odometry(
                            anchor.representation,
                            current,
                            frame,
                            coarse,
                        )
                    except RuntimeError:
                        continue
                    if not success:
                        continue
                    quality = evaluate_depth_alignment(
                        anchor.frame.depth,
                        frame.depth,
                        frame.camera,
                        transformation,
                        depth_threshold_m=max(0.04, self.voxel_size_m * 4.0),
                    )
                    if (
                        quality.accepted
                        and quality.overlap >= 0.38
                        and quality.inlier_ratio >= 0.80
                        and quality.rmse_m <= max(0.022, self.voxel_size_m * 2.2)
                        and float(np.linalg.norm(transformation[:3, 3])) <= 1.5
                        and _rotation_degrees(transformation) <= 90.0
                    ):
                        feature_matches.append(
                            (
                                quality,
                                transformation @ anchor.world_to_camera,
                                anchor.frame.sequence,
                            )
                        )
                if feature_matches:
                    quality, pose, anchor_sequence = max(
                        feature_matches,
                        key=lambda value: value[0].inlier_ratio - value[0].rmse_m,
                    )
                    accepted = (
                        quality,
                        pose,
                        "Tracking globally relocalized to captured geometry",
                    )
                    accepted_anchor_sequence = anchor_sequence

        if accepted is None:
            quality = (
                best[0]
                if best is not None
                else AlignmentQuality(False, 0, 0, math.inf, 0, "odometry failed")
            )
            self.rejected_since_accept += 1
            if (
                self.pending_recovery is not None
                and frame.sequence - self.pending_recovery.sequence
                > RECOVERY_PENDING_MAX_SEQUENCE_GAP
            ):
                self.pending_recovery = None
            return TrackedFrame(
                frame,
                None,
                quality,
                False,
                TrackingState.SEARCHING.value,
                f"Tracking rejected: {quality.reason}; searching all saved capture keyframes",
            )

        quality, proposed_world_to_camera, detail = accepted
        recovery_required = self.rejected_since_accept > 0 or "relocalized" in detail
        if recovery_required:
            anchor_recovery = accepted_anchor_sequence is not None
            gyro_prediction = None
            if not anchor_recovery and self.gyro_samples_since_accept:
                gyro_prediction = self.gyro_since_accept @ self.world_to_camera
            if not anchor_recovery and not _recovery_pose_is_credible(
                self.world_to_camera, proposed_world_to_camera, gyro_prediction
            ):
                self.rejected_since_accept += 1
                rejected_quality = AlignmentQuality(
                    False,
                    quality.overlap,
                    quality.inlier_ratio,
                    quality.rmse_m,
                    quality.correspondence_count,
                    "recovery pose exceeded strict continuity or IMU limits",
                )
                return TrackedFrame(
                    frame,
                    None,
                    rejected_quality,
                    False,
                    TrackingState.SEARCHING.value,
                    "Tracking recovery rejected; return to the last correctly aligned view",
                )

            confirmations = 1
            if self.pending_recovery is not None:
                difference = proposed_world_to_camera @ np.linalg.inv(
                    self.pending_recovery.world_to_camera
                )
                if (
                    self.pending_recovery.anchor_sequence
                    == accepted_anchor_sequence
                    and float(np.linalg.norm(difference[:3, 3]))
                    <= RECOVERY_CONFIRM_TRANSLATION_M
                    and _rotation_degrees(difference)
                    <= RECOVERY_CONFIRM_ROTATION_DEGREES
                ):
                    confirmations = self.pending_recovery.confirmations + 1
            self.pending_recovery = PendingRecovery(
                proposed_world_to_camera.copy(),
                confirmations,
                frame.sequence,
                accepted_anchor_sequence,
            )
            if confirmations < RECOVERY_CONFIRMATION_FRAMES:
                self.rejected_since_accept += 1
                pending_quality = AlignmentQuality(
                    False,
                    quality.overlap,
                    quality.inlier_ratio,
                    quality.rmse_m,
                    quality.correspondence_count,
                    "recovery pose awaiting temporal confirmation",
                )
                return TrackedFrame(
                    frame,
                    None,
                    pending_quality,
                    False,
                    TrackingState.SEARCHING.value,
                    f"Verifying tracking recovery {confirmations}/{RECOVERY_CONFIRMATION_FRAMES}; hold the camera steady",
                )

        self.world_to_camera = proposed_world_to_camera
        self._remember(frame, current)
        fusion_safe = self._quality_is_safe_to_integrate(quality)
        integrate = (
            not recovery_required
            and fusion_safe
            and self._should_integrate(self.world_to_camera, frame.depth_timestamp_us)
        )
        if integrate:
            self._remember_anchor(frame, current, self.world_to_camera)
        self.rejected_since_accept = 0
        self.gyro_since_accept = np.eye(4, dtype=np.float64)
        self.gyro_samples_since_accept = 0
        self.pending_recovery = None
        if recovery_required:
            detail += " - relocalization locked; map resumes after the next validated frame"
        elif not integrate:
            detail += " - map held for fusion quality"
        state = (
            TrackingState.RELOCALIZED.value
            if recovery_required
            else TrackingState.FROZEN.value
            if not fusion_safe
            else TrackingState.TRACKING.value
        )
        return TrackedFrame(
            frame,
            self.world_to_camera.copy(),
            quality,
            integrate,
            state,
            f"{detail} · {quality.overlap:.0%} overlap · {quality.rmse_m * 1000:.0f} mm",
        )


class RealtimeVolume:
    def __init__(
        self,
        o3d: Any,
        frame: RgbdFrame,
        voxel_size_m: float,
        backend: ComputeBackend,
        block_count: int = 24_000,
    ) -> None:
        self.o3d = o3d
        self.camera = frame.camera
        self.mirror_x = frame.mirror_x
        self.voxel_size_m = voxel_size_m
        self.sdf_trunc_m = max(voxel_size_m * 4.0, 0.04)
        self.backend = backend
        self.block_count = block_count
        self.host = o3d.core.Device("CPU:0") if backend.uses_cuda else None
        if backend.uses_cuda:
            self.intrinsic = o3d.core.Tensor(
                _intrinsic_matrix(frame.camera),
                dtype=o3d.core.Dtype.Float64,
                device=self.host,
            )
            self.volume = o3d.t.geometry.VoxelBlockGrid(
                attr_names=("tsdf", "weight", "color"),
                attr_dtypes=(
                    o3d.core.Dtype.Float32,
                    o3d.core.Dtype.UInt16,
                    o3d.core.Dtype.UInt16,
                ),
                attr_channels=((1,), (1,), (3,)),
                voxel_size=voxel_size_m,
                block_resolution=16,
                block_count=block_count,
                device=backend.device,
            )
        else:
            camera = frame.camera
            self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy
            )
            self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=voxel_size_m,
                sdf_trunc=self.sdf_trunc_m,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            )

    def integrate(
        self,
        tracked: TrackedFrame,
        world_to_camera: np.ndarray | None = None,
    ) -> None:
        if tracked.world_to_camera is None:
            return
        if tracked.frame.camera != self.camera:
            raise RuntimeError("Camera calibration changed while fusing a scan")
        extrinsic_values = (
            tracked.world_to_camera if world_to_camera is None else world_to_camera
        )
        if not self.backend.uses_cuda:
            self.volume.integrate(
                _cpu_rgbd(self.o3d, tracked.frame),
                self.intrinsic,
                extrinsic_values,
            )
            return
        rgbd = _tensor_rgbd(self.o3d, tracked.frame, self.backend.device)
        extrinsic = self.o3d.core.Tensor(
            np.ascontiguousarray(extrinsic_values),
            dtype=self.o3d.core.Dtype.Float64,
            device=self.host,
        )
        truncation_multiplier = self.sdf_trunc_m / self.voxel_size_m
        blocks = self.volume.compute_unique_block_coordinates(
            rgbd.depth,
            self.intrinsic,
            extrinsic,
            self.camera.depth_scale,
            self.camera.max_depth_m,
            truncation_multiplier,
        )
        self.volume.integrate(
            blocks,
            rgbd.depth,
            rgbd.color,
            self.intrinsic,
            extrinsic,
            self.camera.depth_scale,
            self.camera.max_depth_m,
            truncation_multiplier,
        )

    def raw_points(self) -> tuple[np.ndarray, np.ndarray]:
        if self.backend.uses_cuda:
            self.o3d.core.cuda.synchronize(self.backend.device)
            # Live preview should become visible after the first integrated
            # camera frame. The production reconstruction still applies its
            # normal confidence thresholds; this only controls the transient
            # viewport extraction from the realtime TSDF.
            cloud = self.volume.extract_point_cloud(weight_threshold=1.0).cpu().to_legacy()
        else:
            cloud = self.volume.extract_point_cloud()
        points = np.asarray(cloud.points, dtype=np.float32)
        colors = np.rint(np.asarray(cloud.colors) * 255.0).clip(0, 255).astype(np.uint8)
        if colors.shape != points.shape:
            colors = np.full(points.shape, 180, dtype=np.uint8)
        return points, colors

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        points, colors = self.raw_points()
        return _display_positions(points, self.mirror_x), colors

    def raw_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.backend.uses_cuda:
            self.o3d.core.cuda.synchronize(self.backend.device)
            mesh = self.volume.extract_triangle_mesh(weight_threshold=3.0).cpu().to_legacy()
        else:
            mesh = self.volume.extract_triangle_mesh()
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        colors = np.rint(np.asarray(mesh.vertex_colors) * 255.0).clip(0, 255).astype(np.uint8)
        if colors.shape != vertices.shape:
            colors = np.full(vertices.shape, 180, dtype=np.uint8)
        return vertices, colors, np.asarray(mesh.triangles, dtype=np.uint32)

    def mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vertices, colors, triangles = self.raw_mesh()
        return (
            _display_positions(vertices, self.mirror_x),
            colors,
            triangles[:, [0, 2, 1]] if self.mirror_x and len(triangles) else triangles,
        )

    def active_block_count(self) -> int:
        if not self.backend.uses_cuda:
            return 0
        return int(self.volume.hashmap().size())


@dataclass
class _CompletedLiveSubmap:
    descriptor: LiveSubmapDescriptor
    points: np.ndarray
    colors: np.ndarray
    vertices: np.ndarray
    vertex_colors: np.ndarray
    triangles: np.ndarray
    tracking_global_from_local: np.ndarray
    display_global_from_local: np.ndarray
    target_global_from_local: np.ndarray
    correction_started_at: float


@dataclass
class _ActiveLiveSubmap:
    id: str
    volume: RealtimeVolume
    global_from_local: np.ndarray
    start_camera_to_world: np.ndarray
    first_sequence: int
    last_sequence: int
    integrated_frames: int
    confidence_sum: float
    confidence_samples: int
    last_point_count: int = 0
    bounds_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_max: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _transform_positions(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if not len(points):
        return np.empty((0, 3), dtype=np.float32)
    return (
        (np.asarray(transform[:3, :3], dtype=np.float64) @ points.T).T
        + np.asarray(transform[:3, 3], dtype=np.float64)
    ).astype(np.float32)


def _bounded_mesh(
    vertices: np.ndarray,
    colors: np.ndarray,
    triangles: np.ndarray,
    triangle_limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = np.asarray(triangles, dtype=np.uint32).reshape((-1, 3))
    if len(triangles) > triangle_limit:
        triangles = triangles[_bounded_indices(len(triangles), triangle_limit)]
    if not len(triangles):
        return vertices[:0], colors[:0], triangles
    used = np.unique(triangles.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return vertices[used], colors[used], remap[triangles].astype(np.uint32)


class LiveSubmapManager:
    def __init__(
        self,
        o3d: Any,
        voxel_size_m: float,
        backend: ComputeBackend,
        mode: str,
        limits: SubmapLimits,
        loop_event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.o3d = o3d
        self.voxel_size_m = voxel_size_m
        self.backend = backend
        self.mode = mode
        self.limits = limits
        self.active: _ActiveLiveSubmap | None = None
        self.completed: list[_CompletedLiveSubmap] = []
        self.coverage = CoverageField(max(0.06, voxel_size_m * 8.0))
        self.next_submap_index = 0
        self.rollover_count = 0
        self.frozen_reason: str | None = None
        self.last_tracking_state = TrackingState.READY.value
        self.last_tracking_confidence = 0.0
        self.mirror_x = False
        self.pose_graph = SubmapPoseGraph(o3d, voxel_size_m)
        self.map_from_tracking_world = np.eye(4, dtype=np.float64)
        self.loop_events: list[dict[str, Any]] = []
        self.queried_loop_pairs: set[tuple[str, str]] = set()
        self.loop_queries_enabled = True
        self.correction_count = 0
        self.correction_duration_seconds = 0.35
        self.loop_event_sink = loop_event_sink

    def _begin(self, tracked: TrackedFrame) -> bool:
        if len(self.completed) >= self.limits.maximum_submaps:
            self.frozen_reason = (
                f"Live map reached its {self.limits.maximum_submaps}-submap hard ceiling"
            )
            return False
        assert tracked.world_to_camera is not None
        camera_to_world = np.linalg.inv(tracked.world_to_camera)
        global_from_local = self.map_from_tracking_world @ camera_to_world
        self.mirror_x = tracked.frame.mirror_x
        self.active = _ActiveLiveSubmap(
            id=f"submap-{self.next_submap_index:04d}",
            volume=RealtimeVolume(
                self.o3d,
                tracked.frame,
                self.voxel_size_m,
                self.backend,
                block_count=self.limits.block_capacity,
            ),
            global_from_local=global_from_local,
            start_camera_to_world=camera_to_world.copy(),
            first_sequence=tracked.frame.sequence,
            last_sequence=tracked.frame.sequence,
            integrated_frames=0,
            confidence_sum=0.0,
            confidence_samples=0,
        )
        self.next_submap_index += 1
        self.frozen_reason = None
        return True

    def _rollover_reason(self, tracked: TrackedFrame) -> str | None:
        if self.active is None or tracked.world_to_camera is None:
            return None
        camera_to_world = np.linalg.inv(tracked.world_to_camera)
        relative = np.linalg.inv(self.active.start_camera_to_world) @ camera_to_world
        if float(np.linalg.norm(relative[:3, 3])) >= self.limits.maximum_distance_m:
            return "travel budget"
        if live_rotation_degrees(relative) >= self.limits.maximum_rotation_degrees:
            return "rotation budget"
        if self.active.integrated_frames >= self.limits.maximum_integrated_frames:
            return "keyframe budget"
        if (
            self.active.volume.active_block_count()
            >= self.limits.rollover_block_count
        ):
            return "voxel-block budget"
        return None

    def force_rollover(self, reason: str) -> None:
        if self.active is not None:
            self.complete_active(reason)

    def integrate(self, tracked: TrackedFrame) -> bool:
        if tracked.world_to_camera is None:
            return False
        reason = self._rollover_reason(tracked)
        if reason is not None:
            self.complete_active(reason)
        if self.active is None and not self._begin(tracked):
            return False
        assert self.active is not None
        local_world_to_camera = (
            tracked.world_to_camera
            @ np.linalg.inv(self.map_from_tracking_world)
            @ self.active.global_from_local
        )
        self.active.volume.integrate(tracked, local_world_to_camera)
        confidence = tracking_confidence(tracked.quality)
        self.active.last_sequence = tracked.frame.sequence
        self.active.integrated_frames += 1
        self.active.confidence_sum += confidence
        self.active.confidence_samples += 1
        self.last_tracking_state = tracked.state
        self.last_tracking_confidence = confidence
        corrected_world_to_camera = (
            tracked.world_to_camera @ np.linalg.inv(self.map_from_tracking_world)
        )
        self.coverage.observe(tracked.frame, corrected_world_to_camera, confidence)
        return True

    def _display_transform(
        self, submap: _CompletedLiveSubmap, now: float | None = None
    ) -> np.ndarray:
        now = time.perf_counter() if now is None else now
        elapsed = now - submap.correction_started_at
        if elapsed >= self.correction_duration_seconds:
            submap.display_global_from_local = submap.target_global_from_local.copy()
            return submap.display_global_from_local
        return interpolate_transform(
            submap.display_global_from_local,
            submap.target_global_from_local,
            elapsed / self.correction_duration_seconds,
        )

    def _record_loop_event(self, event: dict[str, Any]) -> None:
        self.loop_events.append(event)
        if len(self.loop_events) > 64:
            self.loop_events = self.loop_events[-64:]
        if self.loop_event_sink is not None:
            self.loop_event_sink(event)

    def _apply_pose_graph_solution(self, transforms: dict[str, np.ndarray]) -> None:
        now = time.perf_counter()
        previous_map_correction = self.map_from_tracking_world.copy()
        for submap in self.completed:
            corrected = transforms.get(submap.descriptor.id)
            if corrected is None:
                continue
            current = self._display_transform(submap, now).copy()
            submap.display_global_from_local = current
            submap.target_global_from_local = corrected.copy()
            submap.correction_started_at = now
            submap.descriptor = replace(
                submap.descriptor,
                global_from_local=tuple(corrected.reshape(-1).tolist()),
                state="corrected",
            )
        newest = self.completed[-1]
        self.map_from_tracking_world = (
            newest.target_global_from_local
            @ np.linalg.inv(newest.tracking_global_from_local)
        )
        coverage_correction = (
            self.map_from_tracking_world @ np.linalg.inv(previous_map_correction)
        )
        self.coverage.transform(coverage_correction)
        self.correction_count += 1

    def settle_viewport_corrections(self) -> None:
        """Publish exact optimized transforms in the persistent stop artifact."""
        for submap in self.completed:
            submap.display_global_from_local = submap.target_global_from_local.copy()
            submap.correction_started_at = 0.0

    def _query_nonlocal_loops(self, sequence: int) -> None:
        if not self.loop_queries_enabled or len(self.completed) < 3:
            return
        source = self.completed[-1]
        source_center = np.mean(source.points, axis=0) if len(source.points) else np.zeros(3)
        source_world_center = (
            source.target_global_from_local[:3, :3] @ source_center
            + source.target_global_from_local[:3, 3]
        )
        candidates: list[tuple[float, _CompletedLiveSubmap]] = []
        for target in self.completed[:-2]:
            pair = (source.descriptor.id, target.descriptor.id)
            if pair in self.queried_loop_pairs:
                continue
            target_center = np.mean(target.points, axis=0) if len(target.points) else np.zeros(3)
            target_world_center = (
                target.target_global_from_local[:3, :3] @ target_center
                + target.target_global_from_local[:3, 3]
            )
            candidates.append(
                (float(np.linalg.norm(source_world_center - target_world_center)), target)
            )
        for _, target in sorted(candidates, key=lambda item: item[0])[:3]:
            pair = (source.descriptor.id, target.descriptor.id)
            self.queried_loop_pairs.add(pair)
            initial = (
                np.linalg.inv(target.target_global_from_local)
                @ source.target_global_from_local
            )
            verification = verify_loop_candidate(
                self.o3d,
                source_points=source.points,
                target_points=target.points,
                initial_target_from_source=initial,
                voxel_size_m=self.voxel_size_m,
            )
            if not verification.accepted:
                self._record_loop_event(
                    loop_event(
                        sequence=sequence,
                        source_id=source.descriptor.id,
                        target_id=target.descriptor.id,
                        verification=verification,
                        solution=None,
                    )
                )
                continue
            constraint = PoseGraphLoop(
                source.descriptor.id,
                target.descriptor.id,
                verification.target_from_source,
                verification.information,
                verification.fitness,
                verification.rmse_m,
                sequence,
            )
            self.pose_graph.add_loop(constraint)
            solution = self.pose_graph.optimize()
            if not solution.accepted:
                self.pose_graph.loops.pop()
            event = loop_event(
                sequence=sequence,
                source_id=source.descriptor.id,
                target_id=target.descriptor.id,
                verification=verification,
                solution=solution,
            )
            self._record_loop_event(event)
            if solution.accepted:
                self._apply_pose_graph_solution(solution.transforms)
                break

    def _compact_host_points(self) -> None:
        total = sum(len(submap.points) for submap in self.completed)
        if total <= self.limits.maximum_host_points:
            return
        ratio = self.limits.maximum_host_points / max(total, 1)
        for submap in self.completed:
            keep = max(1, int(len(submap.points) * ratio))
            indices = _bounded_indices(len(submap.points), keep)
            submap.points = submap.points[indices]
            submap.colors = submap.colors[indices]

    def _bound_host_mesh(self) -> None:
        total = sum(len(submap.triangles) for submap in self.completed)
        if total <= self.limits.maximum_host_triangles:
            return
        # Mesh is an optional low-rate diagnostic. Retain newer submaps and
        # discard old triangle caches before point guidance or tracking memory.
        for submap in self.completed:
            if total <= self.limits.maximum_host_triangles:
                break
            total -= len(submap.triangles)
            submap.vertices = submap.vertices[:0]
            submap.vertex_colors = submap.vertex_colors[:0]
            submap.triangles = submap.triangles[:0]

    def complete_active(self, reason: str) -> None:
        active = self.active
        if active is None:
            return
        points, colors = active.volume.raw_points()
        indices = _bounded_indices(len(points), min(len(points), 150_000))
        points = points[indices]
        colors = colors[indices]
        if self.mode == "mesh":
            vertices, vertex_colors, triangles = active.volume.raw_mesh()
            vertices, vertex_colors, triangles = _bounded_mesh(
                vertices, vertex_colors, triangles, 150_000
            )
        else:
            vertices = np.empty((0, 3), dtype=np.float32)
            vertex_colors = np.empty((0, 3), dtype=np.uint8)
            triangles = np.empty((0, 3), dtype=np.uint32)
        bounds_min = tuple(np.min(points, axis=0).tolist()) if len(points) else (0.0, 0.0, 0.0)
        bounds_max = tuple(np.max(points, axis=0).tolist()) if len(points) else (0.0, 0.0, 0.0)
        confidence = active.confidence_sum / max(active.confidence_samples, 1)
        descriptor = LiveSubmapDescriptor(
            id=active.id,
            local_origin=tuple(np.eye(4, dtype=np.float64).reshape(-1).tolist()),
            global_from_local=tuple(active.global_from_local.reshape(-1).tolist()),
            state="complete",
            first_sequence=active.first_sequence,
            last_sequence=active.last_sequence,
            voxel_size_m=self.voxel_size_m,
            voxel_count=active.volume.active_block_count() * 16**3,
            point_count=len(points),
            observation_count=active.integrated_frames,
            confidence=confidence,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            resident="host",
        )
        self.completed.append(
            _CompletedLiveSubmap(
                descriptor,
                points,
                colors,
                vertices,
                vertex_colors,
                triangles,
                active.start_camera_to_world.copy(),
                active.global_from_local.copy(),
                active.global_from_local.copy(),
                0.0,
            )
        )
        odometry_information = None
        if len(self.completed) >= 2:
            previous = self.completed[-2]
            current = self.completed[-1]
            current_from_previous = (
                np.linalg.inv(current.target_global_from_local)
                @ previous.target_global_from_local
            )
            odometry_information = submap_odometry_information(
                self.o3d,
                source_points=previous.points,
                target_points=current.points,
                target_from_source=current_from_previous,
                voxel_size_m=self.voxel_size_m,
            )
        self.pose_graph.add_submap(
            active.id, active.global_from_local, odometry_information
        )
        self.active = None
        self.rollover_count += int(reason != "capture stop")
        self._compact_host_points()
        self._bound_host_mesh()
        self._query_nonlocal_loops(active.last_sequence)
        gc.collect()

    def _active_descriptor(self, frame_sequence: int) -> LiveSubmapDescriptor | None:
        active = self.active
        if active is None:
            return None
        confidence = active.confidence_sum / max(active.confidence_samples, 1)
        return LiveSubmapDescriptor(
            id=active.id,
            local_origin=tuple(np.eye(4, dtype=np.float64).reshape(-1).tolist()),
            global_from_local=tuple(active.global_from_local.reshape(-1).tolist()),
            state="active",
            first_sequence=active.first_sequence,
            last_sequence=max(active.last_sequence, frame_sequence),
            voxel_size_m=self.voxel_size_m,
            voxel_count=active.volume.active_block_count() * 16**3,
            point_count=active.last_point_count,
            observation_count=active.integrated_frames,
            confidence=confidence,
            bounds_min=active.bounds_min,
            bounds_max=active.bounds_max,
            resident="gpu" if self.backend.uses_cuda else "host",
        )

    def descriptors(self, frame_sequence: int) -> dict[str, Any]:
        now = time.perf_counter()
        values = [
            replace(
                submap.descriptor,
                global_from_local=tuple(
                    self._display_transform(submap, now).reshape(-1).tolist()
                ),
            )
            for submap in self.completed
        ]
        active = self._active_descriptor(frame_sequence)
        if active is not None:
            values.append(active)
        message = submap_message(frame_sequence, values)
        message["poseGraph"] = {
            "nodeCount": len(self.pose_graph.ids),
            "loopConstraintCount": len(self.pose_graph.loops),
            "acceptedCorrectionCount": self.correction_count,
            "mapFromTrackingWorld": self.map_from_tracking_world.reshape(-1).tolist(),
        }
        message["recentLoopEvents"] = self.loop_events[-16:]
        message["viewportCorrection"] = {
            "durationMs": int(self.correction_duration_seconds * 1000),
            "active": any(
                time.perf_counter() - submap.correction_started_at
                < self.correction_duration_seconds
                for submap in self.completed
                if submap.correction_started_at > 0
            ),
        }
        return message

    def coverage_summary(self) -> CoverageSummary:
        return self.coverage.summary(self.last_tracking_confidence)

    def world_points(self) -> tuple[np.ndarray, np.ndarray]:
        point_batches: list[np.ndarray] = []
        color_batches: list[np.ndarray] = []
        for submap in self.completed:
            point_batches.append(
                _transform_positions(
                    submap.points,
                    self._display_transform(submap),
                )
            )
            color_batches.append(submap.colors)
        if self.active is not None:
            points, colors = self.active.volume.raw_points()
            self.active.last_point_count = len(points)
            if len(points):
                self.active.bounds_min = tuple(np.min(points, axis=0).tolist())
                self.active.bounds_max = tuple(np.max(points, axis=0).tolist())
            point_batches.append(_transform_positions(points, self.active.global_from_local))
            color_batches.append(colors)
        if not point_batches:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        return np.concatenate(point_batches, axis=0), np.concatenate(color_batches, axis=0)

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        points, colors = self.world_points()
        return _display_positions(points, self.mirror_x), colors

    def mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vertex_batches: list[np.ndarray] = []
        color_batches: list[np.ndarray] = []
        triangle_batches: list[np.ndarray] = []
        offset = 0
        for submap in self.completed:
            if not len(submap.triangles):
                continue
            transform = self._display_transform(submap)
            vertices = _transform_positions(submap.vertices, transform)
            vertex_batches.append(vertices)
            color_batches.append(submap.vertex_colors)
            triangle_batches.append(submap.triangles + offset)
            offset += len(vertices)
        if self.active is not None:
            vertices, colors, triangles = self.active.volume.raw_mesh()
            vertices = _transform_positions(vertices, self.active.global_from_local)
            vertex_batches.append(vertices)
            color_batches.append(colors)
            triangle_batches.append(triangles + offset)
        if not vertex_batches:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0, 3), dtype=np.uint32),
            )
        vertices = _display_positions(np.concatenate(vertex_batches), self.mirror_x)
        triangles = np.concatenate(triangle_batches)
        if self.mirror_x and len(triangles):
            triangles = triangles[:, [0, 2, 1]]
        return vertices, np.concatenate(color_batches), triangles

    def telemetry(self) -> dict[str, int]:
        active_blocks = self.active.volume.active_block_count() if self.active else 0
        host_bytes = sum(
            submap.points.nbytes
            + submap.colors.nbytes
            + submap.vertices.nbytes
            + submap.vertex_colors.nbytes
            + submap.triangles.nbytes
            for submap in self.completed
        )
        active_bytes = (
            self.limits.block_capacity * VOXEL_BLOCK_BYTES if self.active else 0
        )
        return {
            "allocatedLiveMapBytes": active_bytes + host_bytes,
            "activeVoxelCount": active_blocks * 16**3,
            "activeSurfelCount": self.active.last_point_count if self.active else 0,
            "residentSubmapCount": int(self.active is not None),
            "hostCachedSubmapCount": len(self.completed),
            "loopClosureCount": self.correction_count,
            "loopCorrectionActive": int(
                any(
                    time.perf_counter() - submap.correction_started_at
                    < self.correction_duration_seconds
                    for submap in self.completed
                    if submap.correction_started_at > 0
                )
            ),
        }


def run_realtime_engine(
    source: BinaryIO,
    output: BinaryIO,
    *,
    mode: str = "mesh",
    voxel_size_m: float = 0.01,
    requested_device: str = "auto",
    session_root: Path | None = None,
    live_map_mib: int = 1024,
    sensor_kind: str = "unknown",
    expected_frame_count: int | None = None,
) -> dict[str, Any]:
    if mode not in {"points", "mesh"}:
        raise ValueError("Realtime mode must be points or mesh")
    if not 0.005 <= voxel_size_m <= 0.08:
        raise ValueError("Realtime voxel size must be between 5 and 80 mm")
    limits = SubmapLimits.from_mebibytes(live_map_mib)
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D is required for realtime RGB-D reconstruction") from error

    effective_device = os.environ.get("SCANLAN_DEVICE", requested_device).strip().lower()
    backend = select_compute_backend(o3d, effective_device)
    from .backend_policy import select_live_backend

    selected_device, _backend_policy = select_live_backend(
        session_root,
        sensor_kind=sensor_kind,
        requested_device=effective_device,
        cuda_active=backend.uses_cuda,
        open3d_revision=str(o3d.__version__),
        expected_frame_count=expected_frame_count,
    )
    if selected_device != ("cuda" if backend.uses_cuda else "cpu"):
        backend = select_compute_backend(o3d, selected_device)
    writer = EngineMessageWriter(output)
    writer.status(
        0,
        {
            "active": True,
            "state": TrackingState.READY.value,
            "detail": "Realtime RGB-D engine ready",
            "backend": backend.label,
            "processedFrames": 0,
            "acceptedFrames": 0,
            "rejectedFrames": 0,
            "integratedFrames": 0,
            "sourceDrops": 0,
            "trackingQueueDrops": 0,
            "mappingDrops": 0,
            "pointCount": 0,
            "triangleCount": 0,
            "trackingFps": 0.0,
            "trackingQueueDepth": 0,
            "mappingQueueDepth": 0,
            "trackingConfidence": 0.0,
            "poseUncertaintyMm": None,
            "poseUncertaintyDegrees": None,
            "poseLatencyMs": None,
            "mapUpdateLatencyMs": None,
            "mapUpdateHz": 0.0,
            "allocatedLiveMapBytes": 0,
            "activeVoxelCount": 0,
            "activeSurfelCount": 0,
            "residentSubmapCount": 0,
            "hostCachedSubmapCount": 0,
            "droppedPreviewJobs": 0,
            "degradationLevel": 0,
            "loopClosureCount": 0,
            "loopCorrectionActive": False,
        },
    )
    journal = TrackingJournal(session_root) if session_root is not None else None
    loop_journal = LoopJournal(session_root) if session_root is not None else None
    frame_queue = LatestFrameQueue(capacity=4)
    map_queue: queue.Queue[
        TrackedFrame | ResetLiveMap | TrackingStateUpdate | None
    ] = queue.Queue(maxsize=8)
    rollover_request: list[str] = []
    rollover_lock = threading.Lock()
    reader_done = threading.Event()
    stop = threading.Event()
    failure: list[BaseException] = []
    counters = {
        "processed": 0,
        "accepted": 0,
        "rejected": 0,
        "integrated": 0,
        "sourceDrops": 0,
        "mappingDrops": 0,
        "pointCount": 0,
        "triangleCount": 0,
        "mapUpdateLatencyMs": 0.0,
        "mapUpdateHz": 0.0,
        "allocatedLiveMapBytes": 0,
        "activeVoxelCount": 0,
        "activeSurfelCount": 0,
        "residentSubmapCount": 0,
        "hostCachedSubmapCount": 0,
        "droppedPreviewJobs": 0,
        "degradationLevel": 0,
        "coverageCellCount": 0,
        "submapCount": 0,
        "mapCapacityFrozen": False,
        "loopClosureCount": 0,
        "loopCorrectionActive": False,
    }
    counters_lock = threading.Lock()
    started = time.perf_counter()

    def read_frames() -> None:
        previous_sequence: int | None = None
        try:
            while not stop.is_set():
                try:
                    frame = read_rgbd_frame(source)
                except EOFError:
                    break
                frame = RgbdFrame(
                    frame.sequence,
                    frame.depth_timestamp_us,
                    frame.color_timestamp_us,
                    frame.camera,
                    reject_depth_speckles(frame.depth, frame.camera),
                    frame.color,
                    frame.gyro_delta_xyzw,
                    frame.camera_to_world,
                    frame.mirror_x,
                )
                if previous_sequence is not None and frame.sequence > previous_sequence + 1:
                    with counters_lock:
                        counters["sourceDrops"] += frame.sequence - previous_sequence - 1
                previous_sequence = frame.sequence
                frame_queue.put(frame)
        except BaseException as error:
            failure.append(error)
            stop.set()
        finally:
            reader_done.set()

    def enqueue_map(
        item: TrackedFrame | TrackingStateUpdate,
    ) -> None:
        while True:
            try:
                map_queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    map_queue.get_nowait()
                    with counters_lock:
                        counters["mappingDrops"] += 1
                except queue.Empty:
                    pass

    def request_submap_rollover(reason: str) -> None:
        with rollover_lock:
            rollover_request[:] = [reason]

    def map_frames() -> None:
        manager = LiveSubmapManager(
            o3d,
            voxel_size_m,
            backend,
            mode,
            limits,
            loop_journal.append if loop_journal is not None else None,
        )
        controller = AdaptiveBudgetController()
        last_points = 0.0
        last_mesh = 0.0
        last_coverage = 0.0
        last_tracking_overlay = 0.0
        last_tracked: TrackedFrame | None = None
        integration_number = 0

        def publish_points(tracked: TrackedFrame, now: float) -> None:
            nonlocal last_points
            points, colors = manager.points()
            map_update_hz = counters["integrated"] / max(now - started, 1e-3)
            writer.write(
                ENGINE_POINTS,
                tracked.frame.sequence,
                point_packet(
                    tracked.frame.sequence,
                    tracked.frame.depth_timestamp_us,
                    map_update_hz,
                    points,
                    colors,
                ),
            )
            writer.write(
                ENGINE_TRACKING_POINTS,
                tracked.frame.sequence,
                point_packet(
                    tracked.frame.sequence,
                    tracked.frame.depth_timestamp_us,
                    map_update_hz,
                    points,
                    tracking_colors(
                        len(points),
                        manager.last_tracking_state,
                        manager.last_tracking_confidence,
                    ),
                ),
            )
            with counters_lock:
                counters["pointCount"] = len(points)
            last_points = now

        def publish_coverage(tracked: TrackedFrame, now: float) -> None:
            nonlocal last_coverage
            world_points, _ = manager.world_points()
            points = _display_positions(world_points, manager.mirror_x)
            writer.write(
                ENGINE_COVERAGE_POINTS,
                tracked.frame.sequence,
                point_packet(
                    tracked.frame.sequence,
                    tracked.frame.depth_timestamp_us,
                    counters["mapUpdateHz"],
                    points,
                    manager.coverage.colors(world_points),
                ),
            )
            summary = manager.coverage_summary()
            writer.contract_message(
                ENGINE_COVERAGE,
                tracked.frame.sequence,
                summary.to_message(tracked.frame.sequence),
            )
            writer.contract_message(
                ENGINE_SUBMAPS,
                tracked.frame.sequence,
                manager.descriptors(tracked.frame.sequence),
            )
            last_coverage = now

        try:
            while not stop.is_set():
                with rollover_lock:
                    pending_rollover = rollover_request.pop() if rollover_request else None
                if pending_rollover is not None:
                    manager.force_rollover(pending_rollover)
                try:
                    item = map_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    break
                if isinstance(item, ResetLiveMap):
                    manager = LiveSubmapManager(
                        o3d,
                        voxel_size_m,
                        backend,
                        mode,
                        limits,
                        loop_journal.append if loop_journal is not None else None,
                    )
                    last_tracked = None
                    last_points = 0.0
                    last_mesh = 0.0
                    last_coverage = 0.0
                    writer.write(
                        ENGINE_POINTS,
                        item.sequence,
                        point_packet(
                            0,
                            item.timestamp_us,
                            0.0,
                            np.empty((0, 3), dtype=np.float32),
                            np.empty((0, 3), dtype=np.uint8),
                        ),
                    )
                    if mode == "mesh":
                        writer.write(
                            ENGINE_MESH,
                            item.sequence,
                            mesh_packet(
                                0,
                                np.empty((0, 3), dtype=np.float32),
                                np.empty((0, 3), dtype=np.uint8),
                                np.empty((0, 3), dtype=np.uint32),
                                flip_winding=False,
                            ),
                        )
                    continue
                if isinstance(item, TrackingStateUpdate):
                    manager.last_tracking_state = item.state
                    manager.last_tracking_confidence = item.confidence
                    now = time.perf_counter()
                    if now - last_tracking_overlay >= 0.10:
                        points, _ = manager.points()
                        writer.write(
                            ENGINE_TRACKING_POINTS,
                            item.sequence,
                            point_packet(
                                item.sequence,
                                item.timestamp_us,
                                counters["mapUpdateHz"],
                                points,
                                tracking_colors(
                                    len(points), item.state, item.confidence
                                ),
                            ),
                        )
                        last_tracking_overlay = now
                    continue
                tracked = item
                last_tracked = tracked
                map_started = time.perf_counter()
                integration_number += 1
                manager.loop_queries_enabled = controller.level < 5
                if integration_number % controller.integration_stride != 0:
                    with counters_lock:
                        counters["mappingDrops"] += 1
                        counters["droppedPreviewJobs"] += 1
                    continue
                integrated_successfully = manager.integrate(tracked)
                with counters_lock:
                    counters["integrated"] += int(integrated_successfully)
                    integrated = counters["integrated"]
                now = time.perf_counter()
                if now - last_points >= controller.point_interval_seconds:
                    publish_points(tracked, now)
                if (
                    mode == "mesh"
                    and controller.mesh_enabled
                    and now - last_mesh >= 1.0
                ):
                    vertices, colors, triangles = manager.mesh()
                    with counters_lock:
                        counters["triangleCount"] = len(triangles)
                    writer.write(
                        ENGINE_MESH,
                        tracked.frame.sequence,
                        mesh_packet(
                            tracked.frame.sequence,
                            vertices,
                            colors,
                            triangles,
                            flip_winding=False,
                        ),
                    )
                    last_mesh = now
                if now - last_coverage >= controller.coverage_interval_seconds:
                    publish_coverage(tracked, now)
                elapsed_map_ms = (time.perf_counter() - map_started) * 1000.0
                telemetry = manager.telemetry()
                memory_ratio = telemetry["activeVoxelCount"] / max(
                    limits.rollover_block_count * 16**3,
                    1,
                )
                level = controller.observe(
                    map_latency_ms=elapsed_map_ms,
                    mapping_queue_ratio=map_queue.qsize() / map_queue.maxsize,
                    memory_ratio=memory_ratio,
                )
                with counters_lock:
                    counters["mapUpdateLatencyMs"] = elapsed_map_ms
                    counters["mapUpdateHz"] = integrated / max(
                        time.perf_counter() - started, 1e-3
                    )
                    counters.update(telemetry)
                    counters["degradationLevel"] = level
                    counters["coverageCellCount"] = len(manager.coverage.cells)
                    counters["submapCount"] = len(manager.completed) + int(
                        manager.active is not None
                    )
                    counters["mapCapacityFrozen"] = manager.frozen_reason is not None
            if last_tracked is not None and not stop.is_set():
                manager.complete_active("capture stop")
                manager.settle_viewport_corrections()
                with counters_lock:
                    counters.update(manager.telemetry())
                    counters["coverageCellCount"] = len(manager.coverage.cells)
                    counters["submapCount"] = len(manager.completed)
                publish_points(last_tracked, time.perf_counter())
                publish_coverage(last_tracked, time.perf_counter())
                if mode == "mesh":
                    vertices, colors, triangles = manager.mesh()
                    writer.write(
                        ENGINE_MESH,
                        last_tracked.frame.sequence,
                        mesh_packet(
                            last_tracked.frame.sequence,
                            vertices,
                            colors,
                            triangles,
                            flip_winding=False,
                        ),
                    )
                    with counters_lock:
                        counters["triangleCount"] = len(triangles)
        except BaseException as error:
            failure.append(error)
            stop.set()

    reader_thread = threading.Thread(target=read_frames, name="rgbd-reader", daemon=True)
    mapper_thread = threading.Thread(target=map_frames, name="rgbd-mapper", daemon=True)
    reader_thread.start()
    mapper_thread.start()
    tracker = RealtimeTracker(o3d, backend, voxel_size_m)
    last_sequence: int | None = None
    last_status_at = 0.0
    last_raw_preview_at = 0.0
    recording_path = session_root / "recording.flag" if session_root else None
    preview_path = session_root / "preview.flag" if session_root else None
    preview_session = False
    recording_sequence: int | None = None
    capture_map_started = False

    try:
        while not stop.is_set() and (not reader_done.is_set() or frame_queue.qsize()):
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            last_sequence = frame.sequence
            if preview_path is not None and preview_path.exists():
                preview_session = True
            if (
                recording_sequence is None
                and recording_path is not None
                and recording_path.exists()
            ):
                try:
                    recording_sequence = int(recording_path.read_text(encoding="ascii").strip())
                except (OSError, ValueError):
                    # The native worker writes then closes this small handoff
                    # file before removing preview.flag. Never guess an early
                    # sequence while that write is still becoming visible.
                    pass
            if (
                not capture_map_started
                and recording_sequence is not None
                and frame.sequence >= recording_sequence
            ):
                capture_map_started = True
                while True:
                    try:
                        pending = map_queue.get_nowait()
                        if isinstance(pending, TrackedFrame):
                            with counters_lock:
                                counters["mappingDrops"] += 1
                    except queue.Empty:
                        break
                # Preview intentionally performs no odometry. Begin capture
                # with a completely fresh tracker so its first usable recorded
                # frame is identity and no preview reference can reject it.
                tracker = RealtimeTracker(o3d, backend, voxel_size_m)
                started = time.perf_counter()
                with counters_lock:
                    for name in (
                        "processed",
                        "accepted",
                        "rejected",
                        "integrated",
                        "sourceDrops",
                        "mappingDrops",
                        "pointCount",
                        "triangleCount",
                    ):
                        counters[name] = 0
                map_queue.put_nowait(ResetLiveMap(frame.sequence, frame.depth_timestamp_us))
                last_status_at = 0.0

            if preview_session and not capture_map_started:
                now = time.perf_counter()
                with counters_lock:
                    counters["processed"] += 1
                    snapshot = dict(counters)
                if now - last_raw_preview_at >= 0.075:
                    points, colors = frame_point_cloud(frame)
                    writer.write(
                        ENGINE_CAMERA_POINTS,
                        frame.sequence,
                        point_packet(
                            frame.sequence,
                            frame.depth_timestamp_us,
                            snapshot["processed"] / max(now - started, 1e-3),
                            points,
                            colors,
                        ),
                    )
                    last_raw_preview_at = now
                if now - last_status_at >= 0.09:
                    writer.status(
                        frame.sequence,
                        {
                            "active": True,
                            "state": TrackingState.PREVIEW.value,
                            "detail": "Raw camera preview; tracking starts from the first recorded frame",
                            "backend": backend.label,
                            "processedFrames": snapshot["processed"],
                            "acceptedFrames": 0,
                            "rejectedFrames": 0,
                            "integratedFrames": 0,
                            "sourceDrops": snapshot["sourceDrops"],
                            "trackingQueueDrops": frame_queue.dropped,
                            "mappingDrops": 0,
                            "journalDrops": journal.dropped if journal is not None else 0,
                            "pointCount": 0,
                            "triangleCount": 0,
                            "overlap": 0.0,
                            "inlierRatio": 0.0,
                            "depthRmseMm": None,
                            "trackingFps": 0.0,
                            "trackingQueueDepth": frame_queue.qsize(),
                            "mappingQueueDepth": 0,
                            "trackingConfidence": 0.0,
                            "poseUncertaintyMm": None,
                            "poseUncertaintyDegrees": None,
                            "poseLatencyMs": None,
                            "mapUpdateLatencyMs": snapshot["mapUpdateLatencyMs"],
                            "mapUpdateHz": snapshot["mapUpdateHz"],
                            "allocatedLiveMapBytes": 0,
                            "activeVoxelCount": 0,
                            "activeSurfelCount": 0,
                            "residentSubmapCount": 0,
                            "hostCachedSubmapCount": 0,
                            "droppedPreviewJobs": 0,
                            "degradationLevel": 0,
                            "loopClosureCount": 0,
                            "loopCorrectionActive": False,
                        },
                    )
                    last_status_at = now
                continue

            pose_started = time.perf_counter()
            tracked = tracker.track(frame)
            pose_latency_ms = (time.perf_counter() - pose_started) * 1000.0
            position_uncertainty_mm, rotation_uncertainty_degrees = pose_uncertainty(
                tracked.quality
            )
            if journal is not None:
                journal.append(tracked)
            with counters_lock:
                counters["processed"] += 1
                if tracked.world_to_camera is None:
                    counters["rejected"] += 1
                else:
                    counters["accepted"] += 1
                snapshot = dict(counters)
            if tracked.integrate:
                enqueue_map(tracked)
            elif tracked.state == TrackingState.RELOCALIZED.value:
                request_submap_rollover("tracking discontinuity")
            if (
                not tracked.integrate
                and tracked.state != TrackingState.TRACKING.value
            ):
                enqueue_map(
                    TrackingStateUpdate(
                        tracked.frame.sequence,
                        tracked.frame.depth_timestamp_us,
                        tracked.state,
                        tracking_confidence(tracked.quality),
                    )
                )
            now = time.perf_counter()
            if now - last_status_at >= 0.09 or tracked.world_to_camera is None:
                writer.status(
                    frame.sequence,
                    {
                        "active": True,
                        "state": tracked.state,
                        "detail": tracked.detail,
                        "backend": backend.label,
                        "processedFrames": snapshot["processed"],
                        "acceptedFrames": snapshot["accepted"],
                        "rejectedFrames": snapshot["rejected"],
                        "integratedFrames": snapshot["integrated"],
                        "sourceDrops": snapshot["sourceDrops"],
                        "trackingQueueDrops": frame_queue.dropped,
                        "mappingDrops": snapshot["mappingDrops"],
                        "journalDrops": journal.dropped if journal is not None else 0,
                        "pointCount": snapshot["pointCount"],
                        "triangleCount": snapshot["triangleCount"],
                        "overlap": tracked.quality.overlap,
                        "inlierRatio": tracked.quality.inlier_ratio,
                        "depthRmseMm": (
                            tracked.quality.rmse_m * 1000.0
                            if math.isfinite(tracked.quality.rmse_m)
                            else None
                        ),
                        "trackingFps": snapshot["processed"] / max(now - started, 1e-3),
                        "trackingQueueDepth": frame_queue.qsize(),
                        "mappingQueueDepth": map_queue.qsize(),
                        "trackingConfidence": tracking_confidence(tracked.quality),
                        "poseUncertaintyMm": position_uncertainty_mm,
                        "poseUncertaintyDegrees": rotation_uncertainty_degrees,
                        "poseLatencyMs": pose_latency_ms,
                        "mapUpdateLatencyMs": snapshot["mapUpdateLatencyMs"],
                        "mapUpdateHz": snapshot["mapUpdateHz"],
                        "allocatedLiveMapBytes": snapshot["allocatedLiveMapBytes"],
                        "activeVoxelCount": snapshot["activeVoxelCount"],
                        "activeSurfelCount": snapshot["activeSurfelCount"],
                        "residentSubmapCount": snapshot["residentSubmapCount"],
                        "hostCachedSubmapCount": snapshot["hostCachedSubmapCount"],
                        "droppedPreviewJobs": snapshot["droppedPreviewJobs"],
                        "degradationLevel": snapshot["degradationLevel"],
                        "loopClosureCount": snapshot["loopClosureCount"],
                        "loopCorrectionActive": bool(snapshot["loopCorrectionActive"]),
                        "integrationFrozen": (
                            tracked.world_to_camera is None
                            or tracked.state == TrackingState.FROZEN.value
                            or snapshot["mapCapacityFrozen"]
                        ),
                    },
                )
                last_status_at = now
    finally:
        reader_done.set()
        try:
            map_queue.put(None, timeout=5.0)
        except queue.Full:
            while mapper_thread.is_alive():
                try:
                    map_queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    try:
                        map_queue.get_nowait()
                        with counters_lock:
                            counters["mappingDrops"] += 1
                    except queue.Empty:
                        continue
        reader_thread.join(timeout=2.0)
        mapper_thread.join(timeout=30.0)
        stop.set()
        if journal is not None:
            journal.close()
        if loop_journal is not None:
            loop_journal.close()

    if failure:
        raise RuntimeError(str(failure[0])) from failure[0]
    with counters_lock:
        result = dict(counters)
    result["journalDrops"] = journal.dropped if journal is not None else 0
    if journal is not None and journal.error is not None:
        result["journalError"] = journal.error
    result["loopJournalDrops"] = loop_journal.dropped if loop_journal is not None else 0
    if loop_journal is not None and loop_journal.error is not None:
        result["loopJournalError"] = loop_journal.error
    result.update(backend=backend.label, elapsedSeconds=time.perf_counter() - started)
    writer.status(
        last_sequence or 0,
        {
            "active": False,
            "state": TrackingState.COMPLETE.value,
            "detail": "Realtime RGB-D engine stopped cleanly",
            "backend": backend.label,
            "processedFrames": result["processed"],
            "acceptedFrames": result["accepted"],
            "rejectedFrames": result["rejected"],
            "integratedFrames": result["integrated"],
            "sourceDrops": result["sourceDrops"],
            "trackingQueueDrops": frame_queue.dropped,
            "mappingDrops": result["mappingDrops"],
            "journalDrops": result["journalDrops"],
            "pointCount": result["pointCount"],
            "triangleCount": result["triangleCount"],
            "trackingConfidence": 0.0,
            "poseUncertaintyMm": None,
            "poseUncertaintyDegrees": None,
            "poseLatencyMs": None,
            "mapUpdateLatencyMs": result["mapUpdateLatencyMs"],
            "mapUpdateHz": result["mapUpdateHz"],
            "allocatedLiveMapBytes": result["allocatedLiveMapBytes"],
            "activeVoxelCount": result["activeVoxelCount"],
            "activeSurfelCount": result["activeSurfelCount"],
            "residentSubmapCount": result["residentSubmapCount"],
            "hostCachedSubmapCount": result["hostCachedSubmapCount"],
            "droppedPreviewJobs": result["droppedPreviewJobs"],
            "degradationLevel": result["degradationLevel"],
            "loopClosureCount": result["loopClosureCount"],
            "loopCorrectionActive": bool(result["loopCorrectionActive"]),
            "integrationFrozen": bool(result["mapCapacityFrozen"]),
        },
    )
    return result
