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
    world_from_depth_opencv,
    world_from_rgb_camera,
)
from .io import effective_rgb_camera, load_depth, load_source_rgb, save_binary_ply, write_json
from .splat_seed import SEED_VERSION, GaussianSeeds, compact_seed_batches, seed_rgbd_gaussians

if TYPE_CHECKING:
    from .mesh import PosedFrame


def _hash_file(digest: Any, path: Path) -> None:
    digest.update(path.name.encode("utf-8"))
    stat = path.stat()
    digest.update(str(stat.st_size).encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


def dataset_fingerprint(frames: list[PosedFrame]) -> str:
    digest = hashlib.sha256()
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


def build_posed_dataset(
    cache_root: Path,
    frames: list[PosedFrame],
    progress: Callable[..., None] | None = None,
) -> dict[str, Any]:
    if not frames:
        raise ValueError("Optimized camera poses are required to build the canonical dataset")
    fingerprint = dataset_fingerprint(frames)
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
    for output_index, frame in enumerate(frames):
        source_frame = frame.source.frames[frame.frame_index]
        rgb_camera = effective_rgb_camera(frame.source)
        image = load_source_rgb(source_frame, frame.source)
        if image.shape[:2] != (rgb_camera.height, rgb_camera.width):
            image = np.asarray(
                Image.fromarray(image).resize((rgb_camera.width, rgb_camera.height), Image.Resampling.LANCZOS),
                dtype=np.uint8,
            )
        depth = load_depth(source_frame, frame.source.camera)
        depth_rgb, uv_map, visibility = rgb_depth_zbuffer(depth, frame.source)
        mask = robust_depth_mask(depth_rgb)
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
        world_from_rgb = world_from_rgb_camera(
            frame.camera_to_global,
            frame.image_y_up,
            frame.source.rgb_from_depth,
        )
        records.append(
            {
                "image": f"images/{stem}.jpg",
                "depth": f"depths/{stem}.png",
                "depthMask": f"masks/{stem}.png",
                "worldFromRgbCamera": [round(float(value), 10) for value in world_from_rgb.reshape(-1)],
                "intrinsics": {
                    "width": rgb_camera.width,
                    "height": rgb_camera.height,
                    "fx": rgb_camera.fx,
                    "fy": rgb_camera.fy,
                    "cx": rgb_camera.cx,
                    "cy": rgb_camera.cy,
                    "model": rgb_camera.model,
                    "distortion": list(rgb_camera.distortion),
                },
                "timestampUs": source_frame.rgb_timestamp_us or source_frame.timestamp_us,
                "phaseId": frame.phase_id,
                "frameIndex": source_frame.index,
                "metric": True,
            }
        )
        seeds = seed_rgbd_gaussians(
            depth,
            image,
            uv_map,
            visibility,
            frame.source.camera,
            world_from_depth_opencv(frame.camera_to_global, frame.image_y_up),
        )
        if len(seeds.points):
            seed_batches.append(seeds)
        if progress:
            progress(
                "Preparing splat data",
                f"Reprojected native RGB depth {output_index + 1} of {len(frames)}",
                0,
                None,
                (output_index + 1) / len(frames),
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
        "schemaVersion": 2,
        "fingerprint": fingerprint,
        "coordinateConvention": {
            "handedness": "right",
            "units": "metres",
            "cameraAxes": "opencv_x_right_y_down_z_forward",
            "pose": "worldFromCamera",
            "matrixStorage": "row-major",
        },
        "metric": True,
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
