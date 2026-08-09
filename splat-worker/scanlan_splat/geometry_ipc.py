from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .lingbot import (
    LINGBOT_CODE_REVISION,
    LINGBOT_MAX_SEEDS,
    LINGBOT_MODEL_REVISION,
    LINGBOT_MODEL_SHA256,
    LingbotGeometry,
    infer_lingbot_geometry,
)


GEOMETRY_REQUEST_SCHEMA = 1
GEOMETRY_RESULT_SCHEMA = 1
ProgressCallback = Callable[[str, str, float, str, dict[str, Any]], None]


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(value, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save_geometry(path: Path, geometry: LingbotGeometry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            world_from_cameras=geometry.world_from_cameras,
            intrinsics=geometry.intrinsics,
            points=geometry.points,
            colors=geometry.colors,
            scales=geometry.scales,
            quaternions=geometry.quaternions,
            source_frame_indices=geometry.source_frame_indices,
            frame_confidence=geometry.frame_confidence,
        )
    os.replace(temporary, path)


def _load_geometry(path: Path, metadata: dict[str, Any]) -> LingbotGeometry:
    if (
        metadata.get("codeRevision") != LINGBOT_CODE_REVISION
        or metadata.get("modelRevision") != LINGBOT_MODEL_REVISION
        or metadata.get("modelSha256") != LINGBOT_MODEL_SHA256
    ):
        raise RuntimeError("Geometry worker used an incompatible LingBot-Map revision")
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]).copy() for name in archive.files}
    required = {
        "world_from_cameras",
        "intrinsics",
        "points",
        "colors",
        "scales",
        "quaternions",
        "source_frame_indices",
        "frame_confidence",
    }
    if set(values) != required:
        raise RuntimeError("Geometry worker result has an incompatible array contract")
    point_count = len(values["points"])
    if not (
        values["points"].shape == (point_count, 3)
        and values["colors"].shape == (point_count, 3)
        and values["scales"].shape == (point_count, 3)
        and values["quaternions"].shape == (point_count, 4)
        and values["source_frame_indices"].shape == (point_count,)
    ):
        raise RuntimeError("Geometry worker returned inconsistent dense seed arrays")
    camera_count = len(values["world_from_cameras"])
    if not (
        values["world_from_cameras"].shape == (camera_count, 4, 4)
        and values["intrinsics"].shape == (camera_count, 3, 3)
        and values["frame_confidence"].shape == (camera_count,)
    ):
        raise RuntimeError("Geometry worker returned inconsistent camera arrays")
    for name, value in values.items():
        if name != "colors" and not np.isfinite(value).all():
            raise RuntimeError(f"Geometry worker returned non-finite {name}")
    if not np.allclose(
        values["world_from_cameras"][:, 3, :],
        np.asarray((0.0, 0.0, 0.0, 1.0)),
        atol=1e-5,
    ):
        raise RuntimeError("Geometry worker returned non-rigid camera transforms")
    rotations = values["world_from_cameras"][:, :3, :3]
    orthogonality = rotations.transpose(0, 2, 1) @ rotations
    if not np.allclose(orthogonality, np.eye(3), atol=2e-3) or np.any(
        np.linalg.det(rotations) < 0.99
    ):
        raise RuntimeError("Geometry worker returned invalid camera rotations")
    quaternion_norms = np.linalg.norm(values["quaternions"], axis=1)
    if (
        np.any(values["intrinsics"][:, 0, 0] <= 0)
        or np.any(values["intrinsics"][:, 1, 1] <= 0)
        or np.any(values["scales"] <= 0)
        or np.any(quaternion_norms < 0.99)
        or np.any(quaternion_norms > 1.01)
        or np.any(values["source_frame_indices"] < 0)
        or np.any(values["source_frame_indices"] >= camera_count)
        or np.any(values["frame_confidence"] < 0)
        or np.any(values["frame_confidence"] > 1)
    ):
        raise RuntimeError("Geometry worker returned values outside the geometry contract")
    processed_size = metadata.get("processedSize")
    if not (
        isinstance(processed_size, list)
        and len(processed_size) == 2
        and all(int(value) > 0 for value in processed_size)
    ):
        raise RuntimeError("Geometry worker omitted its processed image size")
    return LingbotGeometry(
        world_from_cameras=values["world_from_cameras"],
        intrinsics=values["intrinsics"],
        points=values["points"],
        colors=values["colors"],
        scales=values["scales"],
        quaternions=values["quaternions"],
        source_frame_indices=values["source_frame_indices"],
        frame_confidence=values["frame_confidence"],
        backend=str(metadata.get("backend", "LingBot-Map geometry worker")),
        model_path=str(metadata.get("modelPath", "")),
        processed_size=(int(processed_size[0]), int(processed_size[1])),
    )


def run_lingbot_map_request(
    request_path: Path,
    progress_path: Path,
    *,
    inference: Callable[..., LingbotGeometry] = infer_lingbot_geometry,
) -> dict[str, Any]:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if int(request.get("schemaVersion", 0)) != GEOMETRY_REQUEST_SCHEMA:
        raise ValueError("Unsupported geometry-worker request schema")
    image_paths = [Path(value).resolve(strict=True) for value in request.get("imagePaths", [])]
    if len(image_paths) < 3:
        raise ValueError("LingBot-Map geometry request requires at least three images")
    cancel_value = str(request.get("cancelPath", "")).strip()
    cancel_path = Path(cancel_value) if cancel_value else None
    rays_value = str(request.get("normalizedRaysPath", "")).strip()
    normalized_rays = (
        np.load(Path(rays_value), allow_pickle=False) if rays_value else None
    )

    def report(
        stage: str,
        detail: str,
        progress: float,
        backend: str,
        metrics: dict[str, Any],
    ) -> None:
        if cancel_path is not None and cancel_path.is_file():
            raise RuntimeError("LingBot-Map geometry inference cancelled")
        _write_json_atomic(
            progress_path,
            {
                "schemaVersion": GEOMETRY_REQUEST_SCHEMA,
                "stage": stage,
                "detail": detail,
                "progress": progress,
                "computeBackend": backend,
                "metrics": metrics,
            },
        )

    geometry = inference(
        image_paths,
        maximum_seeds=int(request.get("maximumSeeds", LINGBOT_MAX_SEEDS)),
        normalized_rays=normalized_rays,
        output_indices=request.get("outputIndices"),
        progress=report,
    )
    if cancel_path is not None and cancel_path.is_file():
        raise RuntimeError("LingBot-Map geometry inference cancelled")
    arrays_path = Path(request["arraysPath"])
    _save_geometry(arrays_path, geometry)
    result = {
        "schemaVersion": GEOMETRY_RESULT_SCHEMA,
        "status": "complete",
        "backend": geometry.backend,
        "modelPath": geometry.model_path,
        "codeRevision": LINGBOT_CODE_REVISION,
        "modelRevision": LINGBOT_MODEL_REVISION,
        "modelSha256": LINGBOT_MODEL_SHA256,
        "processedSize": list(geometry.processed_size),
        "cameraCount": len(geometry.world_from_cameras),
        "pointCount": len(geometry.points),
        "arraysPath": str(arrays_path),
    }
    _write_json_atomic(Path(request["resultPath"]), result)
    report(
        "lingbot_geometry",
        f"Published {len(geometry.points):,} confidence-gated dense seeds",
        1.0,
        geometry.backend,
        {"cameraCount": len(geometry.world_from_cameras), "pointCount": len(geometry.points)},
    )
    return result


def infer_lingbot_geometry_isolated(
    executable: Path,
    image_paths: Sequence[Path],
    *,
    work_root: Path,
    cancel_path: Path,
    maximum_seeds: int = LINGBOT_MAX_SEEDS,
    normalized_rays: np.ndarray | None = None,
    output_indices: Sequence[int] | None = None,
    progress: ProgressCallback | None = None,
) -> LingbotGeometry:
    executable = executable.resolve(strict=True)
    request_root = work_root / f"geometry-{uuid.uuid4().hex}"
    request_root.mkdir(parents=True, exist_ok=False)
    request_path = request_root / "request.json"
    progress_path = request_root / "progress.json"
    result_path = request_root / "result.json"
    arrays_path = request_root / "geometry.npz"
    rays_path = request_root / "normalized-rays.npy"
    if normalized_rays is not None:
        np.save(rays_path, np.asarray(normalized_rays), allow_pickle=False)
    request = {
        "schemaVersion": GEOMETRY_REQUEST_SCHEMA,
        "imagePaths": [str(Path(path).resolve(strict=True)) for path in image_paths],
        "maximumSeeds": int(maximum_seeds),
        "normalizedRaysPath": str(rays_path) if normalized_rays is not None else "",
        "outputIndices": None if output_indices is None else list(map(int, output_indices)),
        "cancelPath": str(cancel_path.resolve()),
        "arraysPath": str(arrays_path.resolve()),
        "resultPath": str(result_path.resolve()),
    }
    _write_json_atomic(request_path, request)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    log_path = request_root / "geometry-worker.log"
    last_progress_mtime = -1
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            [
                str(executable),
                "infer-lingbot-map",
                "--request",
                str(request_path),
                "--progress",
                str(progress_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            creationflags=flags,
        )
        while process.poll() is None:
            if progress_path.is_file():
                modified = progress_path.stat().st_mtime_ns
                if modified != last_progress_mtime:
                    last_progress_mtime = modified
                    try:
                        value = json.loads(progress_path.read_text(encoding="utf-8"))
                        if progress:
                            progress(
                                str(value.get("stage", "lingbot_inference")),
                                str(value.get("detail", "Running isolated LingBot-Map inference")),
                                float(value.get("progress", 0.0)),
                                str(value.get("computeBackend", "LingBot geometry worker")),
                                dict(value.get("metrics") or {}),
                            )
                    except (OSError, ValueError, json.JSONDecodeError):
                        pass
            if cancel_path.is_file():
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.terminate()
                break
            time.sleep(0.10)
        return_code = process.wait()
    if return_code != 0:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        detail = lines[-1] if lines else "Geometry worker failed without diagnostics"
        raise RuntimeError(detail)
    metadata = json.loads(result_path.read_text(encoding="utf-8"))
    if int(metadata.get("schemaVersion", 0)) != GEOMETRY_RESULT_SCHEMA:
        raise RuntimeError("Geometry worker returned an unsupported result schema")
    geometry = _load_geometry(arrays_path, metadata)
    shutil.rmtree(request_root, ignore_errors=True)
    return geometry
