from __future__ import annotations

import json
import math
import queue
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .compute import ComputeBackend, select_compute_backend
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
ENGINE_HEADER = struct.Struct("<8sHHIQ")

POINT_MAGIC = b"K2P1"
MESH_MAGIC = b"K2M2"
MAX_PREVIEW_POINTS = 150_000
MAX_PREVIEW_TRIANGLES = 150_000


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


def _rotation_degrees(matrix: np.ndarray) -> float:
    cosine = np.clip((np.trace(matrix[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


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
    accepted = overlap >= 0.18 and inlier_ratio >= 0.52 and rmse_m <= depth_threshold_m * 0.72
    if overlap < 0.18:
        reason = f"low overlap ({overlap:.0%})"
    elif inlier_ratio < 0.52:
        reason = f"low depth agreement ({inlier_ratio:.0%})"
    elif rmse_m > depth_threshold_m * 0.72:
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
        self.anchors: list[TrackingAnchor] = []
        self.last_integrated_pose: np.ndarray | None = None
        self.last_integrated_timestamp_us = 0

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
        self.anchors.append(
            TrackingAnchor(frame, representation, np.asarray(world_to_camera).copy())
        )
        if len(self.anchors) > 8:
            del self.anchors[0]

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
                return TrackedFrame(frame, None, AlignmentQuality(False, 0, 0, math.inf, 0, "invalid captured pose"), False, "lost", "Captured pose was invalid")
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
                    np.linalg.norm(relative[:3, 3]) / elapsed > 2.4
                    or _rotation_degrees(relative) / elapsed > 220.0
                ):
                    return TrackedFrame(frame, None, AlignmentQuality(False, 0, 0, math.inf, 0, "captured motion jump"), False, "lost", "Captured pose jumped beyond physical limits")
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
                        "lost",
                        f"Kinect Fusion pose rejected: {quality.reason}; return to recently scanned geometry",
                    )
            self.world_to_camera = proposed
            representation = self._representation(frame)
            self._remember(frame, representation)
            integrate = self._should_integrate(proposed, frame.depth_timestamp_us)
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
                "tracking",
                detail,
            )

        if self.previous_frame is None:
            representation = self._representation(frame)
            self._remember(frame, representation)
            integrate = self._should_integrate(self.world_to_camera, frame.depth_timestamp_us)
            if integrate:
                self._remember_anchor(frame, representation, self.world_to_camera)
            return TrackedFrame(frame, self.world_to_camera.copy(), perfect, integrate, "tracking", "RGB-D odometry initialized")

        current = self._representation(frame)
        previous_representation = (
            self.previous_tensor if self.backend.uses_cuda else self.previous_rgbd
        )
        initial = self._initial_guess(frame)
        attempts = [initial]
        if not np.allclose(initial, np.eye(4), atol=1e-6):
            attempts.append(np.eye(4, dtype=np.float64))
        best: tuple[AlignmentQuality, np.ndarray] | None = None
        for guess in attempts:
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

        accepted: tuple[AlignmentQuality, np.ndarray, str] | None = None
        if best is not None and best[0].accepted:
            quality, transformation = best
            elapsed = max(
                (frame.depth_timestamp_us - self.previous_frame.depth_timestamp_us)
                / 1_000_000.0,
                1 / 30,
            )
            distance = float(np.linalg.norm(transformation[:3, 3]))
            angle = _rotation_degrees(transformation)
            if distance / elapsed <= 2.4 and angle / elapsed <= 220.0:
                accepted = (
                    quality,
                    transformation @ self.world_to_camera,
                    "Tracking accepted",
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

        if accepted is None:
            strongest: tuple[AlignmentQuality, np.ndarray] | None = None
            attempts_used = 0
            for anchor in reversed(self.anchors):
                if anchor.frame.sequence == self.previous_frame.sequence:
                    continue
                attempts_used += 1
                if attempts_used > 3:
                    break
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
                pose_jump = candidate_world_to_camera @ np.linalg.inv(self.world_to_camera)
                if (
                    quality.accepted
                    and quality.overlap >= 0.25
                    and quality.inlier_ratio >= 0.62
                    and float(np.linalg.norm(pose_jump[:3, 3])) <= 1.5
                    and _rotation_degrees(pose_jump) <= 90.0
                    and (
                        strongest is None
                        or quality.inlier_ratio - quality.rmse_m
                        > strongest[0].inlier_ratio - strongest[0].rmse_m
                    )
                ):
                    strongest = quality, candidate_world_to_camera
            if strongest is not None:
                accepted = strongest[0], strongest[1], "Tracking relocalized to a recent keyframe"

        if accepted is None:
            quality = (
                best[0]
                if best is not None
                else AlignmentQuality(False, 0, 0, math.inf, 0, "odometry failed")
            )
            return TrackedFrame(
                frame,
                None,
                quality,
                False,
                "lost",
                f"Tracking rejected: {quality.reason}; return to recently scanned geometry",
            )

        quality, proposed_world_to_camera, detail = accepted
        self.world_to_camera = proposed_world_to_camera
        self._remember(frame, current)
        integrate = self._should_integrate(self.world_to_camera, frame.depth_timestamp_us)
        if integrate:
            self._remember_anchor(frame, current, self.world_to_camera)
        return TrackedFrame(
            frame,
            self.world_to_camera.copy(),
            quality,
            integrate,
            "tracking",
            f"{detail} · {quality.overlap:.0%} overlap · {quality.rmse_m * 1000:.0f} mm",
        )


class RealtimeVolume:
    def __init__(
        self,
        o3d: Any,
        frame: RgbdFrame,
        voxel_size_m: float,
        backend: ComputeBackend,
    ) -> None:
        self.o3d = o3d
        self.camera = frame.camera
        self.mirror_x = frame.mirror_x
        self.voxel_size_m = voxel_size_m
        self.sdf_trunc_m = max(voxel_size_m * 4.0, 0.04)
        self.backend = backend
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
                block_count=24_000,
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

    def integrate(self, tracked: TrackedFrame) -> None:
        if tracked.world_to_camera is None:
            return
        if tracked.frame.camera != self.camera:
            raise RuntimeError("Camera calibration changed while fusing a scan")
        if not self.backend.uses_cuda:
            self.volume.integrate(
                _cpu_rgbd(self.o3d, tracked.frame),
                self.intrinsic,
                tracked.world_to_camera,
            )
            return
        rgbd = _tensor_rgbd(self.o3d, tracked.frame, self.backend.device)
        extrinsic = self.o3d.core.Tensor(
            np.ascontiguousarray(tracked.world_to_camera),
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

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        if self.backend.uses_cuda:
            self.o3d.core.cuda.synchronize(self.backend.device)
            cloud = self.volume.extract_point_cloud(weight_threshold=3.0).cpu().to_legacy()
        else:
            cloud = self.volume.extract_point_cloud()
        points = _display_positions(np.asarray(cloud.points), self.mirror_x)
        colors = np.rint(np.asarray(cloud.colors) * 255.0).clip(0, 255).astype(np.uint8)
        if colors.shape != points.shape:
            colors = np.full(points.shape, 180, dtype=np.uint8)
        return points, colors

    def mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.backend.uses_cuda:
            self.o3d.core.cuda.synchronize(self.backend.device)
            mesh = self.volume.extract_triangle_mesh(weight_threshold=3.0).cpu().to_legacy()
        else:
            mesh = self.volume.extract_triangle_mesh()
        vertices = _display_positions(np.asarray(mesh.vertices), self.mirror_x)
        colors = np.rint(np.asarray(mesh.vertex_colors) * 255.0).clip(0, 255).astype(np.uint8)
        if colors.shape != vertices.shape:
            colors = np.full(vertices.shape, 180, dtype=np.uint8)
        return vertices, colors, np.asarray(mesh.triangles, dtype=np.uint32)


def run_realtime_engine(
    source: BinaryIO,
    output: BinaryIO,
    *,
    mode: str = "mesh",
    voxel_size_m: float = 0.01,
    requested_device: str = "auto",
    session_root: Path | None = None,
) -> dict[str, Any]:
    if mode not in {"points", "mesh"}:
        raise ValueError("Realtime mode must be points or mesh")
    if not 0.005 <= voxel_size_m <= 0.08:
        raise ValueError("Realtime voxel size must be between 5 and 80 mm")
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D is required for realtime RGB-D reconstruction") from error

    backend = select_compute_backend(o3d, requested_device)
    writer = EngineMessageWriter(output)
    writer.status(
        0,
        {
            "active": True,
            "state": "ready",
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
        },
    )
    journal = TrackingJournal(session_root) if session_root is not None else None
    frame_queue = LatestFrameQueue(capacity=4)
    map_queue: queue.Queue[TrackedFrame | None] = queue.Queue(maxsize=8)
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

    def enqueue_map(tracked: TrackedFrame) -> None:
        while True:
            try:
                map_queue.put_nowait(tracked)
                return
            except queue.Full:
                try:
                    map_queue.get_nowait()
                    with counters_lock:
                        counters["mappingDrops"] += 1
                except queue.Empty:
                    pass

    def map_frames() -> None:
        volume: RealtimeVolume | None = None
        last_points = 0.0
        last_mesh = 0.0
        last_tracked: TrackedFrame | None = None
        try:
            while not stop.is_set():
                try:
                    tracked = map_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if tracked is None:
                    break
                last_tracked = tracked
                if volume is None:
                    volume = RealtimeVolume(o3d, tracked.frame, voxel_size_m, backend)
                volume.integrate(tracked)
                with counters_lock:
                    counters["integrated"] += 1
                    integrated = counters["integrated"]
                    accepted = counters["accepted"]
                now = time.perf_counter()
                update_fps = accepted / max(now - started, 1e-3)
                if now - last_points >= 0.35:
                    points, colors = volume.points()
                    with counters_lock:
                        counters["pointCount"] = len(points)
                    writer.write(
                        ENGINE_POINTS,
                        tracked.frame.sequence,
                        point_packet(
                            tracked.frame.sequence,
                            tracked.frame.depth_timestamp_us,
                            update_fps,
                            points,
                            colors,
                        ),
                    )
                    last_points = now
                if mode == "mesh" and now - last_mesh >= 1.0:
                    vertices, colors, triangles = volume.mesh()
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
                            flip_winding=volume.mirror_x,
                        ),
                    )
                    last_mesh = now
            # Always publish the newest geometry once more at clean shutdown.
            if volume is not None and last_tracked is not None and not stop.is_set():
                points, colors = volume.points()
                writer.write(
                    ENGINE_POINTS,
                    last_tracked.frame.sequence,
                    point_packet(
                        last_tracked.frame.sequence,
                        last_tracked.frame.depth_timestamp_us,
                        0.0,
                        points,
                        colors,
                    ),
                )
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

    try:
        while not stop.is_set() and (not reader_done.is_set() or frame_queue.qsize()):
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            last_sequence = frame.sequence
            tracked = tracker.track(frame)
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

    if failure:
        raise RuntimeError(str(failure[0])) from failure[0]
    with counters_lock:
        result = dict(counters)
    result["journalDrops"] = journal.dropped if journal is not None else 0
    if journal is not None and journal.error is not None:
        result["journalError"] = journal.error
    result.update(backend=backend.label, elapsedSeconds=time.perf_counter() - started)
    writer.status(
        last_sequence or 0,
        {
            "active": False,
            "state": "complete",
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
        },
    )
    return result
