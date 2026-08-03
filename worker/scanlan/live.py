from __future__ import annotations

import json
import math
import os
import struct
import time
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .compute import (
    ComputeBackend,
    select_compute_backend,
    tensor_intrinsic,
    tensor_odometry,
    tensor_rgbd,
)
from .imu import odometry_rotation_prior
from .io import PhaseData, load_color, load_depth, read_phase, write_json


POINT_MAGIC = b"K2P1"
MESH_MAGIC = b"K2M2"
MAX_PREVIEW_POINTS = 100_000
MAX_PREVIEW_TRIANGLES = 100_000


def _rotation_degrees(matrix: np.ndarray) -> float:
    cosine = np.clip((np.trace(matrix[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    for attempt in range(40):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(min(0.005 * (attempt + 1), 0.05))


def _display_positions(values: np.ndarray, flip_x: bool) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    if result.size == 0:
        return result.reshape((-1, 3))
    if flip_x:
        result[:, 0] *= -1.0
    result[:, 1:] *= -1.0
    return result


def _bounded_indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    # Even sampling is deterministic and keeps all parts of a room visible.
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
    header = struct.pack(
        "<4sIQfI",
        POINT_MAGIC,
        int(frame_count),
        int(timestamp_us),
        float(update_fps),
        len(records),
    )
    return header + records.tobytes()


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
    header = struct.pack(
        "<4sIII",
        MESH_MAGIC,
        int(frame_count),
        len(vertices),
        triangles.size,
    )
    return header + vertices.tobytes() + colors.tobytes() + triangles.astype("<u4").tobytes()


def _legacy_rgbd(o3d: Any, phase: PhaseData, frame_index: int) -> Any:
    color = np.ascontiguousarray(load_color(phase.frames[frame_index], phase.camera))
    depth = np.ascontiguousarray(load_depth(phase.frames[frame_index], phase.camera))
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color),
        o3d.geometry.Image(depth),
        depth_scale=phase.camera.depth_scale,
        depth_trunc=phase.camera.max_depth_m,
        convert_rgb_to_intensity=False,
    )


class LiveVolume:
    def __init__(
        self,
        o3d: Any,
        phase: PhaseData,
        voxel_size_m: float,
        backend: ComputeBackend,
    ) -> None:
        self.o3d = o3d
        self.phase = phase
        self.voxel_size_m = voxel_size_m
        self.backend = backend
        self.sdf_trunc_m = max(voxel_size_m * 4.0, 0.03)
        self.host = o3d.core.Device("CPU:0") if backend.uses_cuda else None
        if backend.uses_cuda:
            self.intrinsic = tensor_intrinsic(o3d, phase, self.host)
            self.volume = o3d.t.geometry.VoxelBlockGrid(
                attr_names=("tsdf", "weight", "color"),
                attr_dtypes=(
                    o3d.core.Dtype.Float32,
                    o3d.core.Dtype.UInt16,
                    o3d.core.Dtype.UInt16,
                ),
                attr_channels=((1), (1), (3)),
                voxel_size=voxel_size_m,
                block_resolution=16,
                block_count=30_000,
                device=backend.device,
            )
        else:
            camera = phase.camera
            self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                camera.width,
                camera.height,
                camera.fx,
                camera.fy,
                camera.cx,
                camera.cy,
            )
            self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=voxel_size_m,
                sdf_trunc=self.sdf_trunc_m,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            )

    def integrate(self, phase: PhaseData, frame_index: int, world_to_camera: np.ndarray) -> None:
        self.phase = phase
        if not self.backend.uses_cuda:
            self.volume.integrate(
                _legacy_rgbd(self.o3d, phase, frame_index),
                self.intrinsic,
                world_to_camera,
            )
            return
        rgbd = tensor_rgbd(self.o3d, phase, frame_index, self.backend.device)
        extrinsic = self.o3d.core.Tensor(
            np.ascontiguousarray(world_to_camera),
            dtype=self.o3d.core.Dtype.Float64,
            device=self.host,
        )
        truncation_multiplier = self.sdf_trunc_m / self.voxel_size_m
        blocks = self.volume.compute_unique_block_coordinates(
            rgbd.depth,
            self.intrinsic,
            extrinsic,
            phase.camera.depth_scale,
            phase.camera.max_depth_m,
            truncation_multiplier,
        )
        self.volume.integrate(
            blocks,
            rgbd.depth,
            rgbd.color,
            self.intrinsic,
            extrinsic,
            phase.camera.depth_scale,
            phase.camera.max_depth_m,
            truncation_multiplier,
        )

    def point_cloud(self) -> Any:
        if self.backend.uses_cuda:
            self.o3d.core.cuda.synchronize(self.backend.device)
            return self.volume.extract_point_cloud(weight_threshold=1.0).cpu().to_legacy()
        return self.volume.extract_point_cloud()

    def triangle_mesh(self) -> Any:
        if self.backend.uses_cuda:
            self.o3d.core.cuda.synchronize(self.backend.device)
            return self.volume.extract_triangle_mesh(weight_threshold=1.0).cpu().to_legacy()
        return self.volume.extract_triangle_mesh()


class LiveTracker:
    def __init__(self, o3d: Any, phase: PhaseData, backend: ComputeBackend, voxel_size_m: float) -> None:
        self.o3d = o3d
        self.backend = backend
        self.voxel_size_m = voxel_size_m
        self.pose_source = str(phase.manifest.get("poseSource", "estimated_offline"))
        self.first_captured_pose: np.ndarray | None = None
        self.world_to_camera = np.eye(4, dtype=np.float64)
        self.previous_frame_index: int | None = None
        self.previous_rgbd: Any | None = None
        self.previous_tensor: Any | None = None
        self.last_integrated_pose: np.ndarray | None = None
        self.last_integrated_timestamp_us = 0
        camera = phase.camera
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy
        )
        self.option = o3d.pipelines.odometry.OdometryOption()
        self.option.depth_diff_max = max(voxel_size_m * 3.0, 0.06)

    def track(self, phase: PhaseData, frame_index: int) -> tuple[np.ndarray | None, str]:
        frame = phase.frames[frame_index]
        if self.pose_source == "kinect_fusion":
            if frame.pose is None:
                return None, "Tracking lost - return to the last reconstructed surface"
            captured = np.asarray(frame.pose, dtype=np.float64)
            if self.first_captured_pose is None:
                self.first_captured_pose = captured
            camera_to_world = np.linalg.inv(self.first_captured_pose) @ captured
            self.world_to_camera = np.linalg.inv(camera_to_world)
            self.previous_frame_index = frame_index
            return self.world_to_camera.copy(), "Kinect Fusion pose integrated"

        if self.previous_frame_index is None:
            if self.backend.uses_cuda:
                self.previous_tensor = tensor_rgbd(self.o3d, phase, frame_index, self.backend.device)
            else:
                self.previous_rgbd = _legacy_rgbd(self.o3d, phase, frame_index)
            self.previous_frame_index = frame_index
            return self.world_to_camera.copy(), "RGB-D odometry initialized"

        previous_timestamp = phase.frames[self.previous_frame_index].timestamp_us
        initial = odometry_rotation_prior(
            phase.imu_samples, previous_timestamp, frame.timestamp_us
        )
        transformation = np.eye(4, dtype=np.float64)
        success = False
        if self.backend.uses_cuda:
            current_tensor = tensor_rgbd(self.o3d, phase, frame_index, self.backend.device)
            for guess in ([initial, np.eye(4)] if initial is not None else [np.eye(4)]):
                try:
                    transformation = tensor_odometry(
                        self.o3d,
                        self.previous_tensor,
                        current_tensor,
                        phase,
                        guess,
                        self.option.depth_diff_max,
                        self.backend,
                    )
                    success = True
                    break
                except RuntimeError:
                    continue
        else:
            current_rgbd = _legacy_rgbd(self.o3d, phase, frame_index)
            for guess in ([initial, np.eye(4)] if initial is not None else [np.eye(4)]):
                success, transformation, _ = self.o3d.pipelines.odometry.compute_rgbd_odometry(
                    self.previous_rgbd,
                    current_rgbd,
                    self.intrinsic,
                    guess,
                    self.o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                    self.option,
                )
                if success:
                    break
        if not success or not np.isfinite(transformation).all():
            return None, "Tracking lost - return to the last reconstructed surface"

        distance = float(np.linalg.norm(transformation[:3, 3]))
        angle = _rotation_degrees(transformation)
        elapsed = max((frame.timestamp_us - previous_timestamp) / 1_000_000.0, 1 / 30)
        if distance / elapsed > 2.4 or angle / elapsed > 220.0:
            return None, "Motion rejected - move more slowly"

        self.world_to_camera = transformation @ self.world_to_camera
        self.previous_frame_index = frame_index
        if self.backend.uses_cuda:
            self.previous_tensor = current_tensor
        else:
            self.previous_rgbd = current_rgbd
        detail = "IMU-aided live RGB-D tracking" if initial is not None else "Live RGB-D tracking locked"
        return self.world_to_camera.copy(), detail

    def should_integrate(self, world_to_camera: np.ndarray, timestamp_us: int) -> bool:
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
            and elapsed < 0.6
        ):
            return False
        self.last_integrated_pose = camera_to_world
        self.last_integrated_timestamp_us = timestamp_us
        return True


def _cloud_arrays(cloud: Any, flip_x: bool) -> tuple[np.ndarray, np.ndarray]:
    points = _display_positions(np.asarray(cloud.points), flip_x)
    colors = np.rint(np.asarray(cloud.colors) * 255.0).clip(0, 255).astype(np.uint8)
    if colors.shape != points.shape:
        colors = np.full(points.shape, 180, dtype=np.uint8)
    return points, colors


def _mesh_arrays(mesh: Any, flip_x: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = _display_positions(np.asarray(mesh.vertices), flip_x)
    colors = np.rint(np.asarray(mesh.vertex_colors) * 255.0).clip(0, 255).astype(np.uint8)
    if colors.shape != vertices.shape:
        colors = np.full(vertices.shape, 180, dtype=np.uint8)
    triangles = np.asarray(mesh.triangles, dtype=np.uint32)
    return vertices, colors, triangles


def live_reconstruct(
    phase_root: Path,
    voxel_size_m: float,
    mode: str,
    requested_device: str = "auto",
    poll_seconds: float = 0.04,
) -> dict[str, Any]:
    if mode not in {"points", "mesh"}:
        raise ValueError("Live reconstruction mode must be points or mesh")
    if voxel_size_m < 0.005 or voxel_size_m > 0.08:
        raise ValueError("Live reconstruction voxel size must be between 5 and 80 mm")

    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D is required for live reconstruction") from error

    point_path = phase_root / "live-reconstruction.points"
    mesh_path = phase_root / "live-reconstruction.mesh"
    status_path = phase_root / "live-reconstruction.json"
    stop_path = phase_root / "live-reconstruction.stop"
    selection_path = phase_root / "live-frame-selection.csv"
    for path in (point_path, mesh_path, status_path, stop_path):
        path.unlink(missing_ok=True)

    selection = selection_path.open("w", encoding="utf-8", newline="")
    selection_writer = csv.writer(selection, lineterminator="\n")
    selection_writer.writerow(("index", "accepted", "reason"))
    selection.flush()

    phase: PhaseData | None = None
    backend: ComputeBackend | None = None
    tracker: LiveTracker | None = None
    volume: LiveVolume | None = None
    processed = 0
    integrated = 0
    point_count = 0
    triangle_count = 0
    tracking = False
    tracking_detail = "Waiting for the first complete RGB-D frame"
    rejected = 0
    started = time.monotonic()
    last_publish = 0.0
    last_published_integrated = 0

    while not stop_path.exists():
        try:
            candidate = read_phase(phase_root, respect_live_selection=False)
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            time.sleep(poll_seconds)
            continue
        if phase is None:
            phase = candidate
            backend = select_compute_backend(o3d, requested_device)
            tracker = LiveTracker(o3d, phase, backend, voxel_size_m)
            volume = LiveVolume(o3d, phase, voxel_size_m, backend)
        else:
            phase = candidate

        assert backend is not None and tracker is not None and volume is not None
        if processed >= len(phase.frames):
            time.sleep(poll_seconds)
            continue

        for frame_index in range(processed, len(phase.frames)):
            frame = phase.frames[frame_index]
            try:
                world_to_camera, tracking_detail = tracker.track(phase, frame_index)
                tracking = world_to_camera is not None
                if world_to_camera is not None and tracker.should_integrate(
                    world_to_camera, frame.timestamp_us
                ):
                    volume.integrate(phase, frame_index, world_to_camera)
                    integrated += 1
            except (OSError, ValueError, RuntimeError) as error:
                tracking = False
                tracking_detail = f"Live fusion skipped frame {frame.index}: {str(error).splitlines()[0]}"
            selection_writer.writerow(
                (
                    frame.index,
                    "true" if tracking else "false",
                    tracking_detail,
                )
            )
            selection.flush()
            if not tracking:
                rejected += 1
            processed = frame_index + 1
            write_json(
                status_path,
                {
                    "active": True,
                    "mode": mode,
                    "tracking": tracking,
                    "trackingStatus": tracking_detail,
                    "processedFrames": processed,
                    "integratedFrames": integrated,
                    "rejectedFrames": rejected,
                    "pointCount": point_count,
                    "triangleCount": triangle_count,
                    "backend": backend.label,
                    "updateFps": processed / max(time.monotonic() - started, 0.001),
                },
            )

        now = time.monotonic()
        if (
            integrated > last_published_integrated
            and now - last_publish >= (0.9 if mode == "mesh" else 0.45)
        ):
            sensor_manifest = phase.manifest.get("sensor") or {}
            flip_x = str(sensor_manifest.get("kind", "kinect_v2")) == "kinect_v2"
            cloud = volume.point_cloud()
            points, colors = _cloud_arrays(cloud, flip_x)
            point_count = len(points)
            update_fps = processed / max(now - started, 0.001)
            latest = phase.frames[min(processed - 1, len(phase.frames) - 1)]
            _atomic_bytes(
                point_path,
                point_packet(processed, latest.timestamp_us, update_fps, points, colors),
            )
            if mode == "mesh":
                mesh = volume.triangle_mesh()
                vertices, vertex_colors, triangles = _mesh_arrays(mesh, flip_x)
                triangle_count = len(triangles)
                _atomic_bytes(
                    mesh_path,
                    mesh_packet(
                        processed,
                        vertices,
                        vertex_colors,
                        triangles,
                        flip_winding=flip_x,
                    ),
                )
            last_publish = now
            last_published_integrated = integrated

        write_json(
            status_path,
            {
                "active": True,
                "mode": mode,
                "tracking": tracking,
                "trackingStatus": tracking_detail,
                "processedFrames": processed,
                "integratedFrames": integrated,
                "rejectedFrames": rejected,
                "pointCount": point_count,
                "triangleCount": triangle_count,
                "backend": backend.label,
                "updateFps": processed / max(time.monotonic() - started, 0.001),
            },
        )

    selection.close()
    return {
        "processedFrames": processed,
        "integratedFrames": integrated,
        "rejectedFrames": rejected,
        "pointCount": point_count,
        "triangleCount": triangle_count,
        "backend": backend.label if backend is not None else "Open3D",
    }
