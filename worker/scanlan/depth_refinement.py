from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from scanlan_validation import validate_depth, validate_ray_depths

from .calibration import project_rgb, robust_depth_mask, world_from_depth_opencv
from .io import (
    frame_rgb_camera,
    frame_rgb_from_depth,
    load_depth,
    read_json,
    write_json,
)

if TYPE_CHECKING:
    from .mesh import PosedFrame


ProgressCallback = Callable[..., None]
DEPTH_REFINEMENT_VERSION = "lingbot-depth-v0.5-guarded-multiview-v2"
LINGBOT_DEPTH_CODE_REVISION = "f3a237e434ae987bc38281476d6cfb5df3e4d739"
LINGBOT_DEPTH_MODEL_REVISION = "79204ed6b837f4fdd192cf563e59481fecfa0295"
LINGBOT_DEPTH_MODEL_SHA256 = "b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23"
MAX_NEIGHBORS = 4
MINIMUM_CONFIRMATIONS = 2
GENERATED_DEPTH_CONFIDENCE = 96


@dataclass(frozen=True)
class DepthOverride:
    key: str
    measured_depth_path: Path
    refined_depth_path: Path
    generated_mask_path: Path
    confidence_path: Path
    generated_pixels: int
    metrics: dict[str, Any]


@dataclass(frozen=True)
class DepthRefinementResult:
    overrides: dict[str, DepthOverride]
    report: dict[str, Any]
    cache_root: Path


def frame_depth_key(frame: PosedFrame) -> str:
    record = frame.source.frames[frame.frame_index]
    return f"{frame.phase_id}:{record.index}"


def _cache_fingerprint(frames: list[PosedFrame]) -> str:
    digest = hashlib.sha256()
    digest.update(DEPTH_REFINEMENT_VERSION.encode("ascii"))
    digest.update(LINGBOT_DEPTH_CODE_REVISION.encode("ascii"))
    digest.update(LINGBOT_DEPTH_MODEL_REVISION.encode("ascii"))
    digest.update(LINGBOT_DEPTH_MODEL_SHA256.encode("ascii"))
    for frame in frames:
        record = frame.source.frames[frame.frame_index]
        digest.update(frame_depth_key(frame).encode("utf-8"))
        digest.update(np.asarray(frame.camera_to_global, dtype="<f8").tobytes())
        digest.update(bytes((1 if frame.image_y_up else 0,)))
        for path in (record.depth_path, record.color_path):
            stat = path.stat()
            digest.update(path.name.encode("utf-8"))
            digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode("ascii"))
        camera = frame.source.camera
        digest.update(
            np.asarray(
                [
                    camera.width,
                    camera.height,
                    camera.fx,
                    camera.fy,
                    camera.cx,
                    camera.cy,
                    camera.depth_scale,
                    camera.max_depth_m,
                ],
                dtype="<f8",
            ).tobytes()
        )
        rgb_camera = frame_rgb_camera(record, frame.source)
        digest.update(rgb_camera.model.encode("ascii"))
        digest.update(
            np.asarray(
                [
                    rgb_camera.width,
                    rgb_camera.height,
                    rgb_camera.fx,
                    rgb_camera.fy,
                    rgb_camera.cx,
                    rgb_camera.cy,
                    *rgb_camera.distortion,
                ],
                dtype="<f8",
            ).tobytes()
        )
        digest.update(
            np.asarray(
                frame_rgb_from_depth(record, frame.source), dtype="<f8"
            ).tobytes()
        )
    return digest.hexdigest()[:24]


def _load_cached_result(cache_root: Path) -> DepthRefinementResult | None:
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = read_json(manifest_path)
        if manifest.get("version") != DEPTH_REFINEMENT_VERSION:
            return None
        if manifest.get("modelSha256") != LINGBOT_DEPTH_MODEL_SHA256:
            return None
        overrides: dict[str, DepthOverride] = {}
        for record in manifest.get("frames", []):
            paths = {
                name: cache_root / str(record[name])
                for name in (
                    "measuredDepthPath",
                    "refinedDepthPath",
                    "generatedMaskPath",
                    "confidencePath",
                )
            }
            if not all(path.is_file() for path in paths.values()):
                return None
            key = str(record["key"])
            overrides[key] = DepthOverride(
                key=key,
                measured_depth_path=paths["measuredDepthPath"],
                refined_depth_path=paths["refinedDepthPath"],
                generated_mask_path=paths["generatedMaskPath"],
                confidence_path=paths["confidencePath"],
                generated_pixels=int(record.get("generatedPixels", 0)),
                metrics=dict(record.get("metrics", {})),
            )
        if not overrides:
            return None
        return DepthRefinementResult(
            overrides=overrides,
            report=dict(manifest.get("report", {})),
            cache_root=cache_root,
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_raw(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(np.ascontiguousarray(value).tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reliable_sensor_mask(depth_m: np.ndarray, maximum_depth_m: float) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= maximum_depth_m)
    support = np.zeros(depth_m.shape, dtype=np.uint8)
    threshold = np.maximum(0.024, depth_m * 0.015)

    def compare(center: tuple[slice, slice], neighbor: tuple[slice, slice]) -> None:
        support[center] += (
            valid[center]
            & valid[neighbor]
            & (np.abs(depth_m[center] - depth_m[neighbor]) <= threshold[center])
        )

    compare((slice(None), slice(1, None)), (slice(None), slice(None, -1)))
    compare((slice(None), slice(None, -1)), (slice(None), slice(1, None)))
    compare((slice(1, None), slice(None)), (slice(None, -1), slice(None)))
    compare((slice(None, -1), slice(None)), (slice(1, None), slice(None)))
    return valid & (support > 0)


def _metric_gate(
    measured: np.ndarray,
    reliable: np.ndarray,
    predicted: np.ndarray,
    model_valid: np.ndarray,
) -> dict[str, Any]:
    validation = validate_depth(measured, predicted, reliable, model_valid)
    values = validation.metrics
    accepted = validation.accepted
    return {
        "accepted": accepted,
        "reason": "metric agreement accepted" if accepted else "; ".join(validation.reasons),
        "sampleCount": int(values.get("sampleCount", 0)),
        "medianResidualMm": round(float(values.get("medianResidual", 0.0)) * 1000.0, 2),
        "p90ResidualMm": round(float(values.get("p90Residual", 0.0)) * 1000.0, 2),
        "scaleBiasPercent": round(float(values.get("scaleBias", 0.0)) * 100.0, 3),
        "inlierRatio": round(float(values.get("inlierRatio", 0.0)), 4),
        "validation": validation.to_dict(),
    }


def _true_rgb_coverage(frame: PosedFrame, depth_m: np.ndarray) -> np.ndarray:
    camera = frame.source.camera
    yy, xx = np.indices(depth_m.shape, dtype=np.float64)
    z = depth_m.astype(np.float64, copy=False)
    finite_depth = np.isfinite(z) & (z > 0.0)
    safe_z = np.where(finite_depth, z, 1.0)
    points = np.stack(
        (
            (xx - camera.cx) * safe_z / camera.fx,
            (yy - camera.cy) * safe_z / camera.fy,
            safe_z,
            np.ones(z.shape, dtype=np.float64),
        ),
        axis=-1,
    ).reshape(-1, 4)
    record = frame.source.frames[frame.frame_index]
    rgb_points = (frame_rgb_from_depth(record, frame.source) @ points.T).T[:, :3]
    u, v, projected_z = project_rgb(rgb_points, frame_rgb_camera(record, frame.source))
    rgb_camera = frame_rgb_camera(record, frame.source)
    covered = (
        finite_depth.reshape(-1)
        & np.isfinite(u)
        & np.isfinite(v)
        & (projected_z > 0.0)
        & (u >= 0.0)
        & (u <= rgb_camera.width - 1.0)
        & (v >= 0.0)
        & (v <= rgb_camera.height - 1.0)
    )
    return covered.reshape(depth_m.shape)


def _neighbor_indices(frames: list[PosedFrame]) -> list[list[int]]:
    poses = [world_from_depth_opencv(frame.camera_to_global, frame.image_y_up) for frame in frames]
    centers = np.asarray([pose[:3, 3] for pose in poses])
    forwards = np.asarray([pose[:3, 2] for pose in poses])
    result: list[list[int]] = []
    for index, frame in enumerate(frames):
        ranked: list[tuple[float, int]] = []
        for candidate, other in enumerate(frames):
            if candidate == index:
                continue
            distance = float(np.linalg.norm(centers[candidate] - centers[index]))
            forward_dot = float(
                np.clip(np.dot(forwards[index], forwards[candidate]), -1.0, 1.0)
            )
            # Repeated frames from effectively the same optical center do not
            # independently validate a learned completion.
            if distance < 0.012 and forward_dot > 0.9999:
                continue
            direction_penalty = 1.0 - forward_dot
            same_phase = frame.phase_id == other.phase_id
            temporal = abs(frame.frame_index - other.frame_index)
            score = distance + direction_penalty * 0.35
            if same_phase:
                score += min(temporal, 1000) * 1e-4 - 0.25
            ranked.append((score, candidate))
        result.append([candidate for _, candidate in sorted(ranked)[:MAX_NEIGHBORS]])
    return result


def _projective_support(
    source_frame: PosedFrame,
    target_frame: PosedFrame,
    source_depth: np.ndarray,
    source_candidates: np.ndarray,
    target_measured: np.ndarray,
    target_reliable: np.ndarray,
    target_prediction: np.ndarray,
    target_model_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.nonzero(source_candidates)
    support = np.zeros(source_candidates.shape, dtype=bool)
    free_space_violation = np.zeros(source_candidates.shape, dtype=bool)
    if not len(rows):
        return support, free_space_violation
    source_camera = source_frame.source.camera
    z = source_depth[rows, columns].astype(np.float64)
    camera_points = np.column_stack(
        (
            (columns - source_camera.cx) * z / source_camera.fx,
            (rows - source_camera.cy) * z / source_camera.fy,
            z,
            np.ones(len(z), dtype=np.float64),
        )
    )
    world_from_source = world_from_depth_opencv(
        source_frame.camera_to_global, source_frame.image_y_up
    )
    target_from_world = np.linalg.inv(
        world_from_depth_opencv(target_frame.camera_to_global, target_frame.image_y_up)
    )
    target_points = (target_from_world @ world_from_source @ camera_points.T).T[:, :3]
    projected_z = target_points[:, 2]
    target_camera = target_frame.source.camera
    safe_z = np.where(projected_z > 1e-8, projected_z, 1.0)
    u = np.rint(target_camera.fx * target_points[:, 0] / safe_z + target_camera.cx).astype(np.int64)
    v = np.rint(target_camera.fy * target_points[:, 1] / safe_z + target_camera.cy).astype(np.int64)
    inside = (
        (projected_z > 0.0)
        & (u >= 0)
        & (u < target_camera.width)
        & (v >= 0)
        & (v < target_camera.height)
    )
    if not np.any(inside):
        return support, free_space_violation
    selected = np.flatnonzero(inside)
    target_u = u[selected]
    target_v = v[selected]
    has_measured = target_reliable[target_v, target_u]
    reference = np.where(
        has_measured,
        target_measured[target_v, target_u],
        target_prediction[target_v, target_u],
    )
    reference_valid = has_measured | target_model_valid[target_v, target_u]
    ray_validation = validate_ray_depths(
        projected_z[selected], reference, reference_valid
    )
    supported_source = selected[ray_validation.support_mask]
    violated_source = selected[ray_validation.free_space_violation_mask]
    support[rows[supported_source], columns[supported_source]] = True
    free_space_violation[rows[violated_source], columns[violated_source]] = True
    return support, free_space_violation


def validate_predictions(
    frames: list[PosedFrame],
    predictions: list[np.ndarray],
    model_masks: list[np.ndarray],
    output_root: Path,
) -> tuple[dict[str, DepthOverride], dict[str, Any], list[dict[str, Any]]]:
    state_root = output_root / "validation-state"
    state_root.mkdir(parents=True, exist_ok=True)
    state_paths: list[dict[str, Path]] = []
    metrics: list[dict[str, Any]] = []
    for index, (frame, prediction, model_mask) in enumerate(
        zip(frames, predictions, model_masks, strict=True)
    ):
        camera = frame.source.camera
        record = frame.source.frames[frame.frame_index]
        raw = load_depth(record, camera)
        measured = raw.astype(np.float32) / float(camera.depth_scale)
        measured_valid = (
            np.isfinite(measured)
            & (measured > 0.0)
            & (measured <= camera.max_depth_m)
        )
        reliable = _reliable_sensor_mask(measured, camera.max_depth_m)
        model_valid = (
            np.asarray(model_mask, dtype=bool)
            & np.isfinite(prediction)
            & (prediction > 0.0)
            & (prediction <= camera.max_depth_m)
        )
        gate = _metric_gate(measured, reliable, prediction, model_valid)
        stable_surface = robust_depth_mask(np.where(model_valid, prediction, 0.0))
        rgb_coverage = _true_rgb_coverage(frame, prediction)
        # Completion is strictly hole-only. A nonzero sensor sample remains
        # immutable even when it is outside the configured fusion range.
        candidate = (raw == 0) & model_valid & stable_surface & rgb_coverage
        if not gate["accepted"]:
            candidate.fill(False)
        paths = {
            "measured": state_root / f"{index:06}-measured.f32",
            "measuredValid": state_root / f"{index:06}-measured-valid.u8",
            "reliable": state_root / f"{index:06}-reliable.u8",
            "modelValid": state_root / f"{index:06}-model-valid.u8",
            "candidate": state_root / f"{index:06}-candidate.u8",
        }
        _write_raw(paths["measured"], measured.astype("<f4", copy=False))
        _write_raw(paths["measuredValid"], measured_valid.astype(np.uint8))
        _write_raw(paths["reliable"], reliable.astype(np.uint8))
        _write_raw(paths["modelValid"], model_valid.astype(np.uint8))
        _write_raw(paths["candidate"], candidate.astype(np.uint8))
        state_paths.append(paths)
        metrics.append(
            {
                **gate,
                "rawValidPixels": int((measured > 0.0).sum()),
                "reliableMeasuredPixels": int(reliable.sum()),
                "candidatePixels": int(candidate.sum()),
            }
        )

    neighbors = _neighbor_indices(frames)
    overrides: dict[str, DepthOverride] = {}
    manifest_frames: list[dict[str, Any]] = []
    generated_total = 0
    measured_total = 0
    for index, frame in enumerate(frames):
        shape = predictions[index].shape
        source_candidate = np.memmap(
            state_paths[index]["candidate"], dtype=np.uint8, mode="r", shape=shape
        ) > 0
        confirmation_count = np.zeros(shape, dtype=np.uint8)
        free_space_violation_count = np.zeros(shape, dtype=np.uint8)
        for neighbor in neighbors[index]:
            target_shape = predictions[neighbor].shape
            target_measured = np.memmap(
                state_paths[neighbor]["measured"],
                dtype="<f4",
                mode="r",
                shape=target_shape,
            )
            target_reliable = np.memmap(
                state_paths[neighbor]["reliable"],
                dtype=np.uint8,
                mode="r",
                shape=target_shape,
            ) > 0
            target_model_valid = np.memmap(
                state_paths[neighbor]["modelValid"],
                dtype=np.uint8,
                mode="r",
                shape=target_shape,
            ) > 0
            support, free_space_violation = _projective_support(
                frame,
                frames[neighbor],
                predictions[index],
                source_candidate,
                target_measured,
                target_reliable,
                predictions[neighbor],
                target_model_valid,
            )
            confirmation_count += support
            free_space_violation_count += free_space_violation
            del target_measured, target_reliable, target_model_valid
        required_confirmations = min(MINIMUM_CONFIRMATIONS, len(neighbors[index]))
        accepted = source_candidate & (
            confirmation_count >= max(required_confirmations, 1)
        ) & (free_space_violation_count == 0)
        if len(neighbors[index]) == 0:
            accepted.fill(False)
        camera = frame.source.camera
        raw = load_depth(frame.source.frames[frame.frame_index], camera)
        measured_units = raw.copy()
        refined_units = measured_units.copy()
        predicted_units = np.rint(
            np.clip(predictions[index], 0.0, 65_535.0 / camera.depth_scale)
            * camera.depth_scale
        ).astype(np.uint16)
        refined_units[accepted] = predicted_units[accepted]
        measured_valid = np.memmap(
            state_paths[index]["measuredValid"],
            dtype=np.uint8,
            mode="r",
            shape=shape,
        ) > 0
        confidence = np.zeros(accepted.shape, dtype=np.uint8)
        confidence[measured_valid] = 255
        confidence[accepted] = GENERATED_DEPTH_CONFIDENCE
        key = frame_depth_key(frame)
        stem = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        measured_path = output_root / "measured" / f"{stem}.bin"
        refined_path = output_root / "refined" / f"{stem}.bin"
        generated_mask_path = output_root / "generated-masks" / f"{stem}.bin"
        confidence_path = output_root / "confidence" / f"{stem}.bin"
        _write_raw(measured_path, measured_units.astype("<u2", copy=False))
        _write_raw(refined_path, refined_units.astype("<u2", copy=False))
        _write_raw(generated_mask_path, accepted.astype(np.uint8))
        _write_raw(confidence_path, confidence)
        frame_metrics = {
            **metrics[index],
            "neighborCount": len(neighbors[index]),
            "requiredConfirmations": required_confirmations,
            "freeSpaceViolationPixels": int(
                np.count_nonzero(source_candidate & (free_space_violation_count > 0))
            ),
            "generatedPixels": int(accepted.sum()),
            "generatedCoveragePercent": round(float(accepted.mean()) * 100.0, 3),
        }
        generated_total += int(accepted.sum())
        measured_total += int(metrics[index]["reliableMeasuredPixels"])
        override = DepthOverride(
            key=key,
            measured_depth_path=measured_path,
            refined_depth_path=refined_path,
            generated_mask_path=generated_mask_path,
            confidence_path=confidence_path,
            generated_pixels=int(accepted.sum()),
            metrics=frame_metrics,
        )
        overrides[key] = override
        manifest_frames.append(
            {
                "key": key,
                "measuredDepthPath": measured_path.relative_to(output_root).as_posix(),
                "refinedDepthPath": refined_path.relative_to(output_root).as_posix(),
                "generatedMaskPath": generated_mask_path.relative_to(output_root).as_posix(),
                "confidencePath": confidence_path.relative_to(output_root).as_posix(),
                "generatedPixels": override.generated_pixels,
                "metrics": frame_metrics,
            }
        )
    shutil.rmtree(state_root, ignore_errors=True)
    report = {
        "enabled": True,
        "method": "LingBot-Depth v0.5 with metric and multi-view quality gates",
        "frameCount": len(frames),
        "acceptedFrameCount": sum(bool(value["accepted"]) for value in metrics),
        "generatedPixelCount": generated_total,
        "reliableMeasuredPixelCount": measured_total,
        "generatedToMeasuredPercent": round(
            generated_total * 100.0 / max(measured_total, 1), 3
        ),
        "generatedFusionWeight": 0.5,
        "generatedTrainingConfidence": GENERATED_DEPTH_CONFIDENCE / 255.0,
        "modelRevision": LINGBOT_DEPTH_MODEL_REVISION,
        "modelSha256": LINGBOT_DEPTH_MODEL_SHA256,
    }
    return overrides, report, manifest_frames


def _run_inference_worker(
    executable: Path,
    request_path: Path,
    progress_path: Path,
    log_path: Path,
    cancel_path: Path,
    progress: ProgressCallback | None,
) -> None:
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            [
                str(executable),
                "refine-rgbd-depth",
                "--request",
                str(request_path),
                "--progress",
                str(progress_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            creationflags=creation_flags,
        )
        last_progress_mtime = -1
        while process.poll() is None:
            if progress_path.is_file():
                stat = progress_path.stat()
                if stat.st_mtime_ns != last_progress_mtime:
                    last_progress_mtime = stat.st_mtime_ns
                    try:
                        state = read_json(progress_path)
                        if progress:
                            progress(
                                "LingBot depth refinement",
                                str(state.get("detail", "Refining aligned RGB-D depth")),
                                0,
                                None,
                                stage_progress=float(state.get("progress", 0.0)),
                                compute_backend=state.get("computeBackend"),
                            )
                    except (OSError, ValueError, json.JSONDecodeError):
                        pass
            if cancel_path.is_file():
                # The child observes the same flag between inference frames.
                # Give the active CUDA kernel a short grace period to finish.
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.terminate()
                break
            time.sleep(0.25)
        return_code = process.wait()
    if return_code != 0:
        detail = "LingBot-Depth inference worker failed"
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if lines:
                detail = lines[-1]
        except OSError:
            pass
        raise RuntimeError(detail)


def _close_memmaps(arrays: list[np.ndarray]) -> None:
    """Release Windows file handles before an atomic cache-directory move."""
    for array in arrays:
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()


def prepare_lingbot_depth_refinement(
    frames: list[PosedFrame],
    project_root: Path,
    executable: Path,
    progress: ProgressCallback | None = None,
) -> DepthRefinementResult:
    frames = [frame for frame in frames if not frame.depthless]
    if len(frames) < 2:
        raise RuntimeError("LingBot-Depth refinement requires at least two posed RGB-D keyframes")
    executable = executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"LingBot-Depth runtime is missing: {executable}")
    cache_parent = project_root / "outputs" / "cache" / "lingbot-depth"
    fingerprint = _cache_fingerprint(frames)
    cache_root = cache_parent / fingerprint
    cached = _load_cached_result(cache_root)
    if cached is not None:
        if progress:
            progress(
                "LingBot depth refinement",
                f"Reusing {len(cached.overrides)} quality-gated refined keyframes",
                0,
                None,
                stage_progress=1.0,
            )
        return cached
    temporary = cache_parent / f".{fingerprint}.{os.getpid()}.{time.time_ns()}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        predictions_root = temporary / "predictions"
        request_frames: list[dict[str, Any]] = []
        for index, frame in enumerate(frames):
            record = frame.source.frames[frame.frame_index]
            camera = frame.source.camera
            request_frames.append(
                {
                    "key": frame_depth_key(frame),
                    "colorPath": str(record.color_path.resolve()),
                    "depthPath": str(record.depth_path.resolve()),
                    "predictionPath": str((predictions_root / f"{index:06}.npy").resolve()),
                    "modelMaskPath": str((predictions_root / f"{index:06}-mask.npy").resolve()),
                    "width": camera.width,
                    "height": camera.height,
                    "fx": camera.fx,
                    "fy": camera.fy,
                    "cx": camera.cx,
                    "cy": camera.cy,
                    "depthScale": camera.depth_scale,
                    "maximumDepthM": camera.max_depth_m,
                }
            )
        request = {
            "schemaVersion": 1,
            "modelRevision": LINGBOT_DEPTH_MODEL_REVISION,
            "modelSha256": LINGBOT_DEPTH_MODEL_SHA256,
            "cancelPath": str((project_root / "outputs" / "cancel.flag").resolve()),
            "resultPath": str((temporary / "inference-result.json").resolve()),
            "frames": request_frames,
        }
        request_path = temporary / "request.json"
        write_json(request_path, request)
        _run_inference_worker(
            executable,
            request_path,
            temporary / "inference-progress.json",
            temporary / "inference.log",
            project_root / "outputs" / "cancel.flag",
            progress,
        )
        result = read_json(temporary / "inference-result.json")
        if result.get("modelSha256") != LINGBOT_DEPTH_MODEL_SHA256:
            raise RuntimeError("LingBot-Depth worker used an unpinned model")
        predictions: list[np.ndarray] = []
        model_masks: list[np.ndarray] = []
        try:
            predictions = [
                np.load(record["predictionPath"], allow_pickle=False, mmap_mode="r")
                for record in request_frames
            ]
            model_masks = [
                np.load(record["modelMaskPath"], allow_pickle=False, mmap_mode="r")
                for record in request_frames
            ]
            if progress:
                progress(
                    "LingBot depth validation",
                    "Checking metric agreement, true RGB coverage, and neighbouring views",
                    0,
                    None,
                    stage_progress=0.0,
                    compute_backend=str(result.get("backend", "LingBot-Depth v0.5")),
                )
            overrides, report, manifest_frames = validate_predictions(
                frames,
                predictions,
                model_masks,
                temporary,
            )
        finally:
            _close_memmaps(predictions)
            _close_memmaps(model_masks)
        manifest = {
            "version": DEPTH_REFINEMENT_VERSION,
            "fingerprint": fingerprint,
            "modelCodeRevision": LINGBOT_DEPTH_CODE_REVISION,
            "modelRevision": LINGBOT_DEPTH_MODEL_REVISION,
            "modelSha256": LINGBOT_DEPTH_MODEL_SHA256,
            "backend": result.get("backend"),
            "report": report,
            "frames": manifest_frames,
        }
        write_json(temporary / "manifest.json", manifest)
        cache_parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(temporary, cache_root)
        except FileExistsError:
            existing = _load_cached_result(cache_root)
            if existing is None:
                raise
            shutil.rmtree(temporary, ignore_errors=True)
            return existing
        published = _load_cached_result(cache_root)
        if published is None:
            raise RuntimeError("Published LingBot-Depth cache failed validation")
        if progress:
            progress(
                "LingBot depth validation",
                f"Accepted {report['generatedPixelCount']:,} generated depth pixels across {report['acceptedFrameCount']} keyframes",
                0,
                None,
                stage_progress=1.0,
            )
        return published
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
