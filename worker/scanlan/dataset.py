from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from PIL import Image

from .calibration import (
    rgb_depth_zbuffer,
    robust_depth_mask,
    scaled_pinhole_camera,
    undistort_rgb_to_pinhole,
    world_from_depth_opencv,
    world_from_rgb_camera,
)
from .io import (
    frame_rgb_camera,
    frame_rgb_from_depth,
    frame_uses_native_rgb,
    load_depth,
    load_source_rgb,
    save_binary_ply,
    write_json,
)
from .splat_seed import (
    MAX_INITIAL_GAUSSIANS,
    SEED_VERSION,
    GaussianSeeds,
    compact_seed_batches,
    seed_rgbd_gaussians,
)

if TYPE_CHECKING:
    from .mesh import PosedFrame


DATASET_VERSION = "metric-rgbd-pinhole-720-v6-display-world"
CANONICAL_MAX_DIMENSION = 720
MAX_CANONICAL_FRAMES = 600


def _select_training_frames(frames: list[PosedFrame]) -> list[PosedFrame]:
    if len(frames) <= MAX_CANONICAL_FRAMES:
        return frames
    features: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        pose = np.asarray(frame.camera_to_global, dtype=np.float64)
        temporal = index / max(len(frames) - 1, 1)
        features.append(
            np.concatenate(
                (
                    pose[:3, 3] / 0.10,
                    pose[:3, 2] * 2.0,
                    pose[:3, 1],
                    [temporal * 0.25],
                )
            )
        )
    feature_matrix = np.asarray(features)
    required: set[int] = {0, len(frames) - 1}
    for index in range(1, len(frames)):
        if frames[index].phase_id != frames[index - 1].phase_id:
            required.update((index - 1, index))
    selected = np.zeros(len(frames), dtype=bool)
    minimum_distance = np.full(len(frames), np.inf, dtype=np.float64)
    for index in sorted(required):
        selected[index] = True
        difference = feature_matrix - feature_matrix[index]
        minimum_distance = np.minimum(
            minimum_distance,
            np.einsum("ij,ij->i", difference, difference),
        )
    while int(selected.sum()) < MAX_CANONICAL_FRAMES:
        minimum_distance[selected] = -1.0
        next_index = int(np.argmax(minimum_distance))
        selected[next_index] = True
        difference = feature_matrix - feature_matrix[next_index]
        minimum_distance = np.minimum(
            minimum_distance,
            np.einsum("ij,ij->i", difference, difference),
        )
    return [frame for index, frame in enumerate(frames) if selected[index]]


def _hash_file(digest: Any, path: Path) -> None:
    digest.update(path.name.encode("utf-8"))
    stat = path.stat()
    digest.update(str(stat.st_size).encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


def dataset_fingerprint(frames: list[PosedFrame]) -> str:
    digest = hashlib.sha256()
    digest.update(DATASET_VERSION.encode("ascii"))
    digest.update(SEED_VERSION.encode("ascii"))
    seen_phases: set[Path] = set()
    for frame in frames:
        if frame.source.root not in seen_phases:
            _hash_file(digest, frame.source.root / "phase.json")
            _hash_file(digest, frame.source.root / "frames.csv")
            seen_phases.add(frame.source.root)
        record = frame.source.frames[frame.frame_index]
        digest.update(frame.phase_id.encode("utf-8"))
        digest.update(str(record.index).encode("ascii"))
        digest.update(np.asarray(frame.camera_to_global, dtype="<f8").tobytes())
        digest.update(np.asarray(frame.display_axes, dtype="<f8").tobytes())
        for path in (record.depth_path, record.color_path, record.rgb_path):
            if path is not None and path.is_file():
                stat = path.stat()
                digest.update(path.name.encode("utf-8"))
                digest.update(str(stat.st_size).encode("ascii"))
                digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()[:24]


def resolve_dataset(path: Path) -> Path:
    if path.is_file():
        pointer = json.loads(path.read_text(encoding="utf-8"))
        return (path.parent / pointer["path"]).resolve()
    pointer_path = path / "dataset-link.json"
    if pointer_path.is_file():
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        return (path / pointer["path"]).resolve()
    return path.resolve()


def _save_depth_png(path: Path, depth_m: np.ndarray) -> None:
    millimetres = np.rint(np.clip(depth_m, 0.0, 65.535) * 1000.0).astype(np.uint16)
    Image.fromarray(millimetres).save(path, compress_level=3)


def _display_world_matrix(
    world_from_camera: np.ndarray,
    display_axes: tuple[float, float, float],
) -> np.ndarray:
    """Express a camera pose in the same world axes as ScanLan artifacts."""
    display_from_global = np.diag([*display_axes, 1.0])
    return display_from_global @ np.asarray(world_from_camera, dtype=np.float64)


def build_posed_dataset(
    cache_root: Path,
    frames: list[PosedFrame],
    progress: Callable[..., None] | None = None,
) -> dict[str, Any]:
    if not frames:
        raise ValueError("Optimized camera poses are required to build the canonical dataset")
    fingerprint = dataset_fingerprint(frames)
    training_frames = _select_training_frames(frames)
    datasets_root = cache_root / "datasets"
    dataset_root = datasets_root / fingerprint
    manifest_path = dataset_root / "dataset.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        write_json(datasets_root / "current.json", {"path": fingerprint, "fingerprint": fingerprint})
        if progress:
            progress("Preparing splat data", "Reused canonical posed-frame dataset", 0, None, 1.0)
        return payload

    temporary = datasets_root / f".{fingerprint}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    for name in ("images", "depths", "masks"):
        (temporary / name).mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    seed_batches: list[GaussianSeeds] = []
    for output_index, frame in enumerate(training_frames):
        source_frame = frame.source.frames[frame.frame_index]
        rgb_camera = frame_rgb_camera(source_frame, frame.source)
        rgb_from_depth = frame_rgb_from_depth(source_frame, frame.source)
        image = load_source_rgb(source_frame, frame.source)
        if image.shape[:2] != (rgb_camera.height, rgb_camera.width):
            image = np.asarray(
                Image.fromarray(image).resize((rgb_camera.width, rgb_camera.height), Image.Resampling.LANCZOS),
                dtype=np.uint8,
            )
        depth = load_depth(source_frame, frame.source.camera)
        dataset_camera = scaled_pinhole_camera(
            rgb_camera,
            CANONICAL_MAX_DIMENSION,
        )
        depth_rgb, uv_map, visibility = rgb_depth_zbuffer(
            depth,
            frame.source,
            source_frame,
            output_camera=dataset_camera,
        )
        mask = robust_depth_mask(depth_rgb)
        world_from_depth = _display_world_matrix(
            world_from_depth_opencv(frame.camera_to_global, frame.image_y_up),
            frame.display_axes,
        )
        seeds = seed_rgbd_gaussians(
            depth,
            image,
            uv_map,
            visibility,
            frame.source.camera,
            world_from_depth,
        )
        if len(seeds.points):
            seed_batches.append(seeds)
            if len(seed_batches) >= 8:
                seed_batches[:] = [
                    compact_seed_batches(
                        seed_batches,
                        limit=MAX_INITIAL_GAUSSIANS * 2,
                    )
                ]
        image, rgb_valid = undistort_rgb_to_pinhole(
            image,
            rgb_camera,
            dataset_camera,
        )
        mask &= rgb_valid
        stem = f"{output_index:06}"
        Image.fromarray(image).save(
            temporary / "images" / f"{stem}.jpg",
            quality=95,
            optimize=True,
        )
        _save_depth_png(temporary / "depths" / f"{stem}.png", depth_rgb)
        Image.fromarray(mask.astype(np.uint8) * 255).save(
            temporary / "masks" / f"{stem}.png",
            compress_level=3,
        )
        world_from_rgb = _display_world_matrix(
            world_from_rgb_camera(
                frame.camera_to_global,
                frame.image_y_up,
                rgb_from_depth,
            ),
            frame.display_axes,
        )
        records.append(
            {
                "image": f"images/{stem}.jpg",
                "depth": f"depths/{stem}.png",
                "depthMask": f"masks/{stem}.png",
                "worldFromRgbCamera": [round(float(value), 10) for value in world_from_rgb.reshape(-1)],
                "intrinsics": {
                    "width": dataset_camera.width,
                    "height": dataset_camera.height,
                    "fx": dataset_camera.fx,
                    "fy": dataset_camera.fy,
                    "cx": dataset_camera.cx,
                    "cy": dataset_camera.cy,
                    "model": "pinhole",
                    "distortion": [],
                },
                "timestampUs": (
                    source_frame.rgb_timestamp_us
                    if frame_uses_native_rgb(source_frame) and source_frame.rgb_timestamp_us is not None
                    else source_frame.timestamp_us
                ),
                "phaseId": frame.phase_id,
                "frameIndex": source_frame.index,
                "metric": True,
            }
        )
        if progress:
            progress(
                "Preparing splat data",
                f"Reprojected calibrated RGB depth {output_index + 1} of {len(training_frames)}",
                0,
                None,
                (output_index + 1) / len(training_frames),
            )

    seeds = compact_seed_batches(seed_batches)
    save_binary_ply(temporary / "initialization.ply", seeds.points, seeds.colors)
    np.savez(
        temporary / "initialization-2dgs.npz",
        points=seeds.points,
        colors=seeds.colors,
        scales=seeds.scales,
        quaternions=seeds.quaternions,
    )
    payload: dict[str, Any] = {
        "schemaVersion": 3,
        "fingerprint": fingerprint,
        "coordinateConvention": {
            "handedness": "right",
            "units": "metres",
            "worldAxes": "scanlan_display_x_right_y_up_z_back",
            "cameraAxes": "opencv_x_right_y_down_z_forward",
            "pose": "worldFromCamera",
            "matrixStorage": "row-major",
        },
        "metric": True,
        "sourceFrameCount": len(frames),
        "trainingFrameCount": len(training_frames),
        "frames": records,
        "initialization": "initialization.ply",
        "initializationParameters": "initialization-2dgs.npz",
        "gaussianRepresentation": "2d_surface_discs",
        "seedVersion": SEED_VERSION,
        "initialGaussianCount": int(len(seeds.points)),
    }
    write_json(temporary / "dataset.json", payload)
    datasets_root.mkdir(parents=True, exist_ok=True)
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    temporary.replace(dataset_root)
    write_json(datasets_root / "current.json", {"path": fingerprint, "fingerprint": fingerprint})
    return payload
