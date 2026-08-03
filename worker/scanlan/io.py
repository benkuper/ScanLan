from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
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
    rgb_camera: RgbCameraModel | None
    rgb_from_depth: np.ndarray
    frames: list[FrameRecord]
    imu_samples: list[ImuSample]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
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


def read_project(root: Path) -> dict[str, Any]:
    project = read_json(root / "project.json")
    project["path"] = str(root.resolve())
    return project


def read_phase(root: Path, respect_live_selection: bool = True) -> PhaseData:
    manifest = read_json(root / "phase.json")
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
    raw_rgb_camera = manifest.get("rgbCamera")
    rgb_camera = None
    if isinstance(raw_rgb_camera, dict):
        rgb_camera = RgbCameraModel(
            width=int(raw_rgb_camera["width"]),
            height=int(raw_rgb_camera["height"]),
            fx=float(raw_rgb_camera["fx"]),
            fy=float(raw_rgb_camera["fy"]),
            cx=float(raw_rgb_camera["cx"]),
            cy=float(raw_rgb_camera["cy"]),
            model=str(raw_rgb_camera.get("model", "brown_conrady")),
            distortion=tuple(float(value) for value in raw_rgb_camera.get("distortion", [])),
        )
    raw_rgb_from_depth = manifest.get("rgbFromDepth")
    rgb_from_depth = np.eye(4, dtype=np.float64)
    if isinstance(raw_rgb_from_depth, list) and len(raw_rgb_from_depth) == 16:
        candidate = np.asarray(raw_rgb_from_depth, dtype=np.float64).reshape(4, 4)
        if np.isfinite(candidate).all():
            rgb_from_depth = candidate

    frames: list[FrameRecord] = []
    with (root / "frames.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        matrix_keys = [f"m{row}{column}" for row in range(4) for column in range(4)]
        for row in reader:
            matrix_values = [row.get(key, "").strip() for key in matrix_keys]
            pose = None
            if matrix_values and all(matrix_values):
                pose = np.asarray([float(value) for value in matrix_values], dtype=np.float64).reshape(4, 4)
            frames.append(
                FrameRecord(
                    index=int(row["index"]),
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
    selection_path = root / "live-frame-selection.csv"
    if respect_live_selection and selection_path.is_file():
        accepted: set[int] = set()
        with selection_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    if str(row.get("accepted", "")).strip().lower() == "true":
                        accepted.add(int(row["index"]))
                except (KeyError, TypeError, ValueError):
                    continue
        # Once live tracking owns frame selection, an unclassified or rejected
        # frame is unsafe to feed into the authoritative offline trajectory.
        frames = [frame for frame in frames if frame.index in accepted]
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
    """Load native sensor RGB, falling back to the legacy depth-aligned RGB frame."""
    if frame.rgb_path is None or not frame.rgb_path.is_file():
        return load_color(frame, phase.camera)
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required to decode native RGB capture frames") from error
    with Image.open(frame.rgb_path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def effective_rgb_camera(phase: PhaseData) -> RgbCameraModel:
    if phase.rgb_camera is not None:
        return phase.rgb_camera
    camera = phase.camera
    return RgbCameraModel(
        width=camera.width,
        height=camera.height,
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
        model="pinhole",
        distortion=(),
    )


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
