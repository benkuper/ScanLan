from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float
    max_depth_m: float


@dataclass(frozen=True)
class RgbCameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str
    distortion: tuple[float, ...]


@dataclass(frozen=True)
class FrameRecord:
    index: int
    source_sequence: int
    timestamp_us: int
    depth_path: Path
    color_path: Path
    rgb_path: Path | None
    rgb_timestamp_us: int | None
    pose: np.ndarray | None


@dataclass(frozen=True)
class ImuSample:
    timestamp_us: int
    kind: str
    value: np.ndarray
    temperature_c: float


@dataclass(frozen=True)
class PhaseData:
    root: Path
    manifest: dict[str, Any]
    camera: CameraModel
    rgb_camera: RgbCameraModel
    rgb_from_depth: np.ndarray
    frames: list[FrameRecord]
    imu_samples: list[ImuSample]
    tracking_camera_to_world: dict[int, np.ndarray] = field(default_factory=dict)
    tracking_rejected_sequences: frozenset[int] = frozenset()
    tracking_quality: dict[str, float] = field(default_factory=dict)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Windows can briefly deny replacement while the Tauri UI has the previous
        # file open for a status read. Keep the update atomic and wait for that tiny
        # sharing window instead of aborting an otherwise healthy reconstruction.
        for attempt in range(40):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(min(0.005 * (attempt + 1), 0.05))
    finally:
        temporary.unlink(missing_ok=True)


def read_project(root: Path) -> dict[str, Any]:
    project = read_json(root / "project.json")
    if int(project.get("schemaVersion", 0)) != 3:
        raise ValueError("ScanLan requires project schema 3")
    project["path"] = str(root.resolve())
    return project


def _read_tracking_journal(
    root: Path,
) -> tuple[dict[int, np.ndarray], frozenset[int], dict[str, float]]:
    path = root / "tracking.jsonl"
    if not path.is_file():
        return {}, frozenset(), {}
    poses: dict[int, np.ndarray] = {}
    rejected: set[int] = set()
    overlaps: list[float] = []
    errors_mm: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
                if int(entry.get("schemaVersion", 0)) != 1:
                    continue
                sequence = int(entry["sequence"])
                if not bool(entry.get("accepted", False)):
                    rejected.add(sequence)
                    continue
                values = entry.get("worldToCamera")
                if not isinstance(values, list) or len(values) != 16:
                    continue
                world_to_camera = np.asarray(values, dtype=np.float64).reshape(4, 4)
                if not np.isfinite(world_to_camera).all():
                    continue
                camera_to_world = np.linalg.inv(world_to_camera)
                if not np.isfinite(camera_to_world).all():
                    continue
                poses[sequence] = camera_to_world
                overlap = float(entry.get("overlap", "nan"))
                error_mm = float(entry.get("depthRmseMm", "nan"))
                if np.isfinite(overlap):
                    overlaps.append(overlap)
                if np.isfinite(error_mm):
                    errors_mm.append(error_mm)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, np.linalg.LinAlgError):
                continue
    quality: dict[str, float] = {}
    if overlaps:
        quality["meanOverlap"] = float(np.mean(overlaps))
    if errors_mm:
        quality["meanDepthRmseMm"] = float(np.mean(errors_mm))
    return poses, frozenset(rejected), quality


def read_phase(root: Path, *, include_tracking_rejected: bool = False) -> PhaseData:
    manifest = read_json(root / "phase.json")
    if int(manifest.get("schemaVersion", 0)) != 3:
        raise ValueError(f"ScanLan requires capture schema 3: {root}")
    raw_camera = manifest["camera"]
    camera = CameraModel(
        width=int(raw_camera["width"]),
        height=int(raw_camera["height"]),
        fx=float(raw_camera["fx"]),
        fy=float(raw_camera["fy"]),
        cx=float(raw_camera["cx"]),
        cy=float(raw_camera["cy"]),
        depth_scale=float(raw_camera.get("depth_scale", 1000.0)),
        max_depth_m=float(raw_camera.get("max_depth_m", 4.5)),
    )
    camera_values = np.asarray(
        [camera.fx, camera.fy, camera.cx, camera.cy, camera.depth_scale, camera.max_depth_m],
        dtype=np.float64,
    )
    if (
        camera.width <= 0
        or camera.height <= 0
        or not np.isfinite(camera_values).all()
        or camera.fx <= 0.0
        or camera.fy <= 0.0
        or camera.depth_scale <= 0.0
        or camera.max_depth_m <= 0.0
    ):
        raise ValueError(f"Capture depth calibration is invalid: {root}")
    raw_rgb_camera = manifest["rgbCamera"]
    rgb_model = str(raw_rgb_camera["model"])
    raw_distortion = raw_rgb_camera["distortion"]
    if not isinstance(raw_distortion, list):
        raise ValueError(f"Capture RGB distortion must be an array: {root}")
    distortion = tuple(float(value) for value in raw_distortion)
    coefficient_counts = {
        "pinhole": {0},
        "none": {0},
        "brown_conrady": {5},
        "opencv_rational": {8},
    }
    if rgb_model not in coefficient_counts or len(distortion) not in coefficient_counts[rgb_model]:
        raise ValueError(f"Capture RGB lens model is unsupported or incomplete: {root}")
    rgb_camera = RgbCameraModel(
        width=int(raw_rgb_camera["width"]),
        height=int(raw_rgb_camera["height"]),
        fx=float(raw_rgb_camera["fx"]),
        fy=float(raw_rgb_camera["fy"]),
        cx=float(raw_rgb_camera["cx"]),
        cy=float(raw_rgb_camera["cy"]),
        model=rgb_model,
        distortion=distortion,
    )
    rgb_values = np.asarray(
        [rgb_camera.fx, rgb_camera.fy, rgb_camera.cx, rgb_camera.cy, *rgb_camera.distortion],
        dtype=np.float64,
    )
    if (
        rgb_camera.width <= 0
        or rgb_camera.height <= 0
        or not np.isfinite(rgb_values).all()
        or rgb_camera.fx <= 0.0
        or rgb_camera.fy <= 0.0
    ):
        raise ValueError(f"Capture RGB calibration is invalid: {root}")
    raw_rgb_from_depth = manifest["rgbFromDepth"]
    if not isinstance(raw_rgb_from_depth, list) or len(raw_rgb_from_depth) != 16:
        raise ValueError(f"Capture has no calibrated depth-to-RGB transform: {root}")
    rgb_from_depth = np.asarray(raw_rgb_from_depth, dtype=np.float64).reshape(4, 4)
    rotation = rgb_from_depth[:3, :3]
    if (
        not np.isfinite(rgb_from_depth).all()
        or not np.allclose(rgb_from_depth[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3)
    ):
        raise ValueError(f"Capture depth-to-RGB transform is invalid: {root}")

    tracking_poses, tracking_rejected, tracking_quality = _read_tracking_journal(root)
    frames: list[FrameRecord] = []
    with (root / "frames.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        matrix_keys = [f"m{row}{column}" for row in range(4) for column in range(4)]
        for row in reader:
            source_sequence = int(row["source_sequence"])
            if not include_tracking_rejected and source_sequence in tracking_rejected:
                continue
            matrix_values = [row.get(key, "").strip() for key in matrix_keys]
            pose = None
            if matrix_values and all(matrix_values):
                pose = np.asarray([float(value) for value in matrix_values], dtype=np.float64).reshape(4, 4)
            frames.append(
                FrameRecord(
                    index=int(row["index"]),
                    source_sequence=source_sequence,
                    timestamp_us=int(row["timestamp_us"]),
                    depth_path=root / row["depth_path"],
                    color_path=root / row["color_path"],
                    rgb_path=(root / row["rgb_path"] if row.get("rgb_path", "").strip() else None),
                    rgb_timestamp_us=(
                        int(row["rgb_timestamp_us"])
                        if row.get("rgb_timestamp_us", "").strip()
                        else None
                    ),
                    pose=pose,
                )
            )
    if not frames:
        raise ValueError(f"Phase contains no frames: {root}")

    imu_samples: list[ImuSample] = []
    raw_imu = manifest.get("imu")
    if isinstance(raw_imu, dict) and raw_imu.get("coordinateFrame") == "depth_camera":
        imu_path = root / str(raw_imu.get("path", "imu.csv"))
        if imu_path.is_file():
            with imu_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    kind = row.get("type", "")
                    if kind not in {"accel", "gyro"}:
                        continue
                    try:
                        value = np.asarray(
                            [float(row["x"]), float(row["y"]), float(row["z"])],
                            dtype=np.float64,
                        )
                        if not np.isfinite(value).all():
                            continue
                        imu_samples.append(
                            ImuSample(
                                timestamp_us=int(row["timestamp_us"]),
                                kind=kind,
                                value=value,
                                temperature_c=float(row.get("temperature_c", "nan")),
                            )
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
    imu_samples.sort(key=lambda sample: sample.timestamp_us)
    return PhaseData(
        root=root,
        manifest=manifest,
        camera=camera,
        rgb_camera=rgb_camera,
        rgb_from_depth=rgb_from_depth,
        frames=frames,
        imu_samples=imu_samples,
        tracking_camera_to_world=tracking_poses,
        tracking_rejected_sequences=tracking_rejected,
        tracking_quality=tracking_quality,
    )


def load_depth(frame: FrameRecord, camera: CameraModel) -> np.ndarray:
    values = np.fromfile(frame.depth_path, dtype="<u2")
    expected = camera.width * camera.height
    if values.size != expected:
        raise ValueError(
            f"Depth frame {frame.depth_path} has {values.size} samples; expected {expected}"
        )
    return values.reshape(camera.height, camera.width)


def load_color(frame: FrameRecord, camera: CameraModel) -> np.ndarray:
    values = np.fromfile(frame.color_path, dtype=np.uint8)
    expected = camera.width * camera.height * 3
    if values.size != expected:
        raise ValueError(
            f"Color frame {frame.color_path} has {values.size} bytes; expected {expected}"
        )
    return values.reshape(camera.height, camera.width, 3)


def load_source_rgb(frame: FrameRecord, phase: PhaseData) -> np.ndarray:
    """Load native RGB, using the aligned frame only after a JPEG write failure."""
    if not frame_uses_native_rgb(frame):
        return load_color(frame, phase.camera)
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required to decode native RGB capture frames") from error
    with Image.open(frame.rgb_path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def frame_uses_native_rgb(frame: FrameRecord) -> bool:
    return frame.rgb_path is not None and frame.rgb_path.is_file()


def frame_rgb_camera(frame: FrameRecord, phase: PhaseData) -> RgbCameraModel:
    if frame_uses_native_rgb(frame):
        return phase.rgb_camera
    camera = phase.camera
    return RgbCameraModel(
        camera.width,
        camera.height,
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        "pinhole",
        (),
    )


def frame_rgb_from_depth(frame: FrameRecord, phase: PhaseData) -> np.ndarray:
    return phase.rgb_from_depth if frame_uses_native_rgb(frame) else np.eye(4, dtype=np.float64)


def save_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points and colors must both be N×3 arrays")
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_type = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(points.shape[0], dtype=vertex_type)
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment generated by ScanLan\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def save_preview(path: Path, points: np.ndarray, colors: np.ndarray, limit: int = 14_000) -> None:
    if points.shape[0] > limit:
        indices = np.linspace(0, points.shape[0] - 1, limit, dtype=np.int64)
        points = points[indices]
        colors = colors[indices]
    payload = [
        {
            "position": [round(float(value), 5) for value in point],
            "color": [int(value) for value in color],
        }
        for point, color in zip(points, colors, strict=True)
    ]
    write_json(path, payload)


def phase_roots(project_root: Path, project: dict[str, Any]) -> Iterable[Path]:
    for phase in project.get("phases", []):
        if phase.get("status", "complete") == "complete" and int(phase.get("frameCount", 0)) > 0:
            yield project_root / "phases" / phase["id"]
