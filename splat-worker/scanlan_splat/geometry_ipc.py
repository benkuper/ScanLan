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
PreviewCallback = Callable[[np.ndarray, np.ndarray, dict[str, Any]], None]


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


def _save_preview(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, points=points, colors=colors)
    os.replace(temporary, path)


class ProgressivePreviewAccumulator:
    """Maintain bounded learned-depth submaps and explicit uncertainty telemetry."""

    def __init__(self, *, maximum_points: int = 120_000, resident_submaps: int = 16) -> None:
        self.maximum_points = max(10_000, int(maximum_points))
        self.resident_submaps = max(2, int(resident_submaps))
        self.chunks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.chunk_status: dict[int, dict[str, Any]] = {}
        self.archived_points = np.empty((0, 3), dtype=np.float32)
        self.archived_colors = np.empty((0, 3), dtype=np.uint8)
        self.total_submaps = 0
        self.archived_accepted = 0
        self.archived_rejected = 0
        self.archived_confidence_sum = 0.0
        self.archived_frame_count = 0
        self.archived_drift_risk = 0.0

    @staticmethod
    def _rotation_step_degrees(first: np.ndarray, second: np.ndarray) -> float:
        relative = first[:3, :3].T @ second[:3, :3]
        cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    @staticmethod
    def _bounded(values: np.ndarray, count: int) -> np.ndarray:
        if len(values) <= count:
            return values
        return values[np.linspace(0, len(values) - 1, count, dtype=np.int64)]

    def update(self, geometry: LingbotGeometry, first_frame: int, final_frame: int) -> dict[str, Any]:
        if first_frame == 0 and 0 in self.chunks:
            # A FlashInfer launch can fail after emitting its scale frames and
            # retry from frame zero with SDPA. Restart publication rather than
            # mixing two executions of the same ordered stream.
            self.chunks.clear()
            self.chunk_status.clear()
            self.archived_points = np.empty((0, 3), dtype=np.float32)
            self.archived_colors = np.empty((0, 3), dtype=np.uint8)
            self.total_submaps = 0
            self.archived_accepted = 0
            self.archived_rejected = 0
            self.archived_confidence_sum = 0.0
            self.archived_frame_count = 0
            self.archived_drift_risk = 0.0
        poses = np.asarray(geometry.world_from_cameras, dtype=np.float64)
        confidence = np.asarray(geometry.frame_confidence, dtype=np.float32)
        translations = (
            np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
            if len(poses) > 1
            else np.empty(0, dtype=np.float64)
        )
        typical_step = float(np.median(translations[translations > 1e-6])) if np.any(translations > 1e-6) else 0.0
        accepted = confidence >= 0.50
        pose_risk = 0.0
        for index in range(1, len(poses)):
            distance = float(np.linalg.norm(poses[index, :3, 3] - poses[index - 1, :3, 3]))
            angle = self._rotation_step_degrees(poses[index - 1], poses[index])
            distance_limit = max(2.0, typical_step * 4.0)
            frame_pose_risk = max(distance / distance_limit, angle / 75.0)
            pose_risk = max(pose_risk, frame_pose_risk)
            if distance > distance_limit or angle > 75.0:
                accepted[index] = False
        frame_lookup = {
            first_frame + index: bool(value) for index, value in enumerate(accepted)
        }
        point_mask = np.asarray(
            [frame_lookup.get(int(index), False) for index in geometry.source_frame_indices],
            dtype=bool,
        )
        points = np.asarray(geometry.points[point_mask], dtype=np.float32)
        original_colors = np.asarray(geometry.colors[point_mask], dtype=np.float32)
        point_confidence = np.asarray(
            [
                confidence[int(index) - first_frame]
                for index in geometry.source_frame_indices[point_mask]
            ],
            dtype=np.float32,
        )
        confidence_palette = np.column_stack(
            (
                255.0 * (1.0 - point_confidence),
                255.0 * point_confidence,
                np.full(len(point_confidence), 48.0),
            )
        )
        colors = np.rint(0.55 * original_colors + 0.45 * confidence_palette).astype(np.uint8)
        per_submap_limit = max(4_000, self.maximum_points // self.resident_submaps)
        if len(points) > per_submap_limit:
            indices = np.linspace(0, len(points) - 1, per_submap_limit, dtype=np.int64)
            points, colors = points[indices], colors[indices]
        replacing = first_frame in self.chunks
        self.chunks[first_frame] = (points, colors)
        self.chunk_status[first_frame] = {
            "accepted": int(np.count_nonzero(accepted)),
            "rejected": int(len(accepted) - np.count_nonzero(accepted)),
            "confidence": float(np.mean(confidence)) if len(confidence) else 0.0,
            "driftRisk": float(np.clip(max(1.0 - float(np.mean(confidence)), pose_risk - 1.0), 0.0, 1.0)),
            "finalFrame": final_frame,
        }
        if not replacing:
            self.total_submaps += 1
        while len(self.chunks) > self.resident_submaps:
            oldest = min(self.chunks)
            old_points, old_colors = self.chunks.pop(oldest)
            archived_status = self.chunk_status.pop(oldest)
            archived_frames = int(archived_status["accepted"]) + int(
                archived_status["rejected"]
            )
            self.archived_accepted += int(archived_status["accepted"])
            self.archived_rejected += int(archived_status["rejected"])
            self.archived_confidence_sum += float(archived_status["confidence"]) * archived_frames
            self.archived_frame_count += archived_frames
            self.archived_drift_risk = max(
                self.archived_drift_risk,
                float(archived_status["driftRisk"]),
            )
            self.archived_points = np.concatenate((self.archived_points, old_points))
            self.archived_colors = np.concatenate((self.archived_colors, old_colors))
            archive_limit = self.maximum_points // 3
            if len(self.archived_points) > archive_limit:
                indices = np.linspace(
                    0,
                    len(self.archived_points) - 1,
                    archive_limit,
                    dtype=np.int64,
                )
                self.archived_points = self.archived_points[indices]
                self.archived_colors = self.archived_colors[indices]
        point_groups = [self.archived_points, *(value[0] for value in self.chunks.values())]
        color_groups = [self.archived_colors, *(value[1] for value in self.chunks.values())]
        preview_points = np.concatenate(point_groups)
        preview_colors = np.concatenate(color_groups)
        if len(preview_points) > self.maximum_points:
            indices = np.linspace(0, len(preview_points) - 1, self.maximum_points, dtype=np.int64)
            preview_points, preview_colors = preview_points[indices], preview_colors[indices]
        statuses = list(self.chunk_status.values())
        accepted_count = self.archived_accepted + sum(
            int(value["accepted"]) for value in statuses
        )
        rejected_count = self.archived_rejected + sum(
            int(value["rejected"]) for value in statuses
        )
        resident_frames = sum(
            int(value["accepted"]) + int(value["rejected"]) for value in statuses
        )
        confidence_sum = self.archived_confidence_sum + sum(
            float(value["confidence"])
            * (int(value["accepted"]) + int(value["rejected"]))
            for value in statuses
        )
        status = {
            "schemaVersion": 1,
            "active": True,
            "learnedOnly": True,
            "scaleStatus": "MODEL_METRIC_UNVERIFIED",
            "processedFrameCount": int(final_frame),
            "acceptedFrameCount": accepted_count,
            "rejectedFrameCount": rejected_count,
            "residentSubmapCount": len(self.chunks),
            "archivedSubmapCount": self.total_submaps - len(self.chunks),
            "pointCount": len(preview_points),
            "confidence": float(
                confidence_sum / max(1, self.archived_frame_count + resident_frames)
            ),
            "driftRisk": float(
                max(
                    self.archived_drift_risk,
                    max((value["driftRisk"] for value in statuses), default=0.0),
                )
            ),
            "integrationFrozen": not len(points),
        }
        return {"points": preview_points, "colors": preview_colors, "status": status}


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


def _load_preview(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"points", "colors"}:
            raise RuntimeError("Geometry worker preview has an incompatible array contract")
        points = np.asarray(archive["points"], dtype=np.float32).copy()
        colors = np.asarray(archive["colors"], dtype=np.uint8).copy()
    if points.ndim != 2 or points.shape[1:] != (3,) or colors.shape != points.shape:
        raise RuntimeError("Geometry worker preview has inconsistent point arrays")
    if len(points) > 150_000 or not np.isfinite(points).all():
        raise RuntimeError("Geometry worker preview violates its bounded finite contract")
    return points, colors


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
    preview_value = str(request.get("previewPath", "")).strip()
    preview_status_value = str(request.get("previewStatusPath", "")).strip()
    preview_path = Path(preview_value) if preview_value else None
    preview_status_path = Path(preview_status_value) if preview_status_value else None
    preview_accumulator = (
        ProgressivePreviewAccumulator(
            maximum_points=int(request.get("maximumPreviewPoints", 120_000)),
            resident_submaps=int(request.get("maximumResidentSubmaps", 16)),
        )
        if preview_path is not None and preview_status_path is not None
        else None
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

    def publish_preview(
        chunk: LingbotGeometry,
        first_frame: int,
        final_frame: int,
    ) -> None:
        if preview_accumulator is None or preview_path is None or preview_status_path is None:
            return
        preview = preview_accumulator.update(chunk, first_frame, final_frame)
        _save_preview(preview_path, preview["points"], preview["colors"])
        status = dict(preview["status"])
        status["revision"] = time.time_ns()
        _write_json_atomic(preview_status_path, status)
        report(
            "rgb_preview_streaming",
            (
                f"Provisional learned-depth preview: {status['acceptedFrameCount']} accepted, "
                f"{status['rejectedFrameCount']} rejected frames"
            ),
            0.10 + 0.09 * (final_frame / len(image_paths)),
            chunk.backend,
            {"rgbPreview": status, "previewRevision": status["revision"]},
        )

    geometry = inference(
        image_paths,
        maximum_seeds=int(request.get("maximumSeeds", LINGBOT_MAX_SEEDS)),
        normalized_rays=normalized_rays,
        output_indices=request.get("outputIndices"),
        progress=report,
        stream=publish_preview if preview_accumulator is not None else None,
        stream_chunk_frames=int(request.get("previewChunkFrames", 8)),
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
        "progressivePreview": preview_accumulator is not None,
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
    preview: PreviewCallback | None = None,
) -> LingbotGeometry:
    executable = executable.resolve(strict=True)
    request_root = work_root / f"geometry-{uuid.uuid4().hex}"
    request_root.mkdir(parents=True, exist_ok=False)
    request_path = request_root / "request.json"
    progress_path = request_root / "progress.json"
    result_path = request_root / "result.json"
    arrays_path = request_root / "geometry.npz"
    rays_path = request_root / "normalized-rays.npy"
    preview_path = request_root / "preview.npz"
    preview_status_path = request_root / "preview.json"
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
        "previewPath": str(preview_path.resolve()) if preview is not None else "",
        "previewStatusPath": (
            str(preview_status_path.resolve()) if preview is not None else ""
        ),
        "maximumPreviewPoints": 120_000,
        "maximumResidentSubmaps": 16,
        "previewChunkFrames": 8,
    }
    _write_json_atomic(request_path, request)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    log_path = request_root / "geometry-worker.log"
    last_progress_mtime = -1
    last_preview_revision = -1
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
            if preview is not None and preview_status_path.is_file() and preview_path.is_file():
                try:
                    status = json.loads(preview_status_path.read_text(encoding="utf-8"))
                    revision = int(status.get("revision", 0))
                    if revision > last_preview_revision:
                        points, colors = _load_preview(preview_path)
                        preview(points, colors, status)
                        last_preview_revision = revision
                except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
                    # Publication is latest-wins; a subsequent atomic snapshot
                    # replaces any transient read failure without blocking inference.
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
    if preview is not None and preview_status_path.is_file() and preview_path.is_file():
        status = json.loads(preview_status_path.read_text(encoding="utf-8"))
        revision = int(status.get("revision", 0))
        if revision > last_preview_revision:
            points, colors = _load_preview(preview_path)
            preview(points, colors, status)
    metadata = json.loads(result_path.read_text(encoding="utf-8"))
    if int(metadata.get("schemaVersion", 0)) != GEOMETRY_RESULT_SCHEMA:
        raise RuntimeError("Geometry worker returned an unsupported result schema")
    geometry = _load_geometry(arrays_path, metadata)
    shutil.rmtree(request_root, ignore_errors=True)
    return geometry
