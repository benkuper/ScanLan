from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps
from scanlan_material.radiometry import to_canonical_srgb

from .da3 import DA3_CODE_REVISION, DA3_MODEL_REVISION, DA3_MODEL_SHA256
from .lingbot import (
    LingbotGeometry,
    lingbot_processed_size,
    lingbot_source_pixel_grid,
)
from .geometry_ipc import (
    infer_da3_geometry_isolated,
    infer_lingbot_geometry_isolated,
    infer_mapanything_geometry_isolated,
)
from .initialization import (
    GaussianRepresentation,
    InitializationKind,
    initialization_manifest,
)
from .runtime import pycolmap_device, pycolmap_feature_runtime


PHOTO_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".bmp",
}
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
    ".mts",
    ".m2ts",
}


@dataclass(frozen=True)
class MediaPreparationOptions:
    # This is the rate at which video frames are inspected, not the rate at
    # which they are retained. Optical-flow keyframing below keeps views based
    # on camera motion and tracked overlap, so still/slow passages contribute
    # fewer images while fast turns contribute more.
    video_fps: float = 15.0
    # A crash-safety ceiling only; adaptive selection does not aim for it.
    maximum_video_frames: int = 3_000
    maximum_image_dimension: int = 2560
    minimum_image_dimension: int = 480
    maximum_features: int = 8_192


LINGBOT_CONTEXT_FPS = 10.0
LINGBOT_MAX_CONTEXT_FRAMES = 3_000
ADAPTIVE_MINIMUM_GAP_SECONDS = 0.125
ADAPTIVE_MAXIMUM_GAP_SECONDS = 2.0
ADAPTIVE_TARGET_TRACKED_RATIO = 0.72
ADAPTIVE_HARD_TRACKED_RATIO = 0.50
ADAPTIVE_TARGET_MEDIAN_MOTION = 0.055
ADAPTIVE_TARGET_P90_MOTION = 0.12


def adaptive_frame_selection_status() -> dict[str, Any]:
    """Describe the non-optional video keyframing policy in this worker."""
    defaults = MediaPreparationOptions()
    return {
        "enabled": True,
        "mode": "adaptive_optical_flow",
        "analysisFps": defaults.video_fps,
        "maximumFrames": defaults.maximum_video_frames,
        "maximumFramesIsSafetyCeiling": True,
        "signals": [
            "tracked_overlap",
            "spatial_coverage",
            "camera_motion",
            "parallax",
            "blur_guard",
        ],
    }


class _CameraSolveTelemetry:
    """Thread-safe progress counters fed by PyCOLMAP mapper callbacks."""

    def __init__(self, image_count: int) -> None:
        self.image_count = max(1, image_count)
        self.model_attempts = 0
        self.current_registered = 0
        self.best_registered = 0
        self._lock = threading.Lock()

    def initial_pair_registered(self) -> None:
        with self._lock:
            self.best_registered = max(self.best_registered, self.current_registered)
            self.model_attempts += 1
            self.current_registered = 2

    def next_image_registered(self) -> None:
        with self._lock:
            self.current_registered += 1
            self.best_registered = max(self.best_registered, self.current_registered)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            best = max(self.best_registered, self.current_registered)
            return {
                "imageCount": self.image_count,
                "registeredCameras": self.current_registered,
                "bestRegisteredCameras": best,
                "modelAttempts": self.model_attempts,
            }

    def progress(self) -> float:
        best = self.snapshot()["bestRegisteredCameras"]
        return 0.45 + 0.23 * min(1.0, best / self.image_count)

    def detail(self, elapsed: int) -> str:
        values = self.snapshot()
        minutes, seconds = divmod(elapsed, 60)
        elapsed_label = f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"
        current = values["registeredCameras"]
        best = values["bestRegisteredCameras"]
        attempt = values["modelAttempts"]
        if attempt == 0:
            return f"Finding a stable initial camera pair · {elapsed_label} elapsed"
        best_label = f" · best {best:,}" if best > current else ""
        return (
            f"Solving cameras · {current:,}/{self.image_count:,} registered in model "
            f"{attempt:,}{best_label} · {elapsed_label} elapsed"
        )


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # The desktop UI polls this file for status. On Windows, a read can
        # briefly hold the destination open and make an otherwise atomic
        # replacement fail with ERROR_ACCESS_DENIED. Preserve atomic updates,
        # but wait out that short sharing window instead of aborting the job.
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


def _progress(
    project_root: Path,
    stage: str,
    detail: str,
    progress: float,
    *,
    stage_eta_seconds: int | None = None,
    compute_backend: str = "COLMAP structure-from-motion",
    metrics: dict[str, Any] | None = None,
) -> None:
    payload = {
        "stage": stage,
        "detail": detail,
        "progress": float(np.clip(progress, 0.0, 1.0)),
        "stageProgress": float(np.clip(progress, 0.0, 1.0)),
        "iteration": None,
        "totalIterations": None,
        "etaSeconds": None,
        "stageEtaSeconds": stage_eta_seconds,
        "computeBackend": compute_backend,
    }
    if metrics:
        payload["metrics"] = metrics
    _write_json_atomic(
        project_root / "outputs" / "splat-progress.json",
        payload,
    )


@contextmanager
def _progress_heartbeat(
    project_root: Path,
    stage: str,
    detail: str,
    progress: float,
    *,
    compute_backend: str,
    metrics: dict[str, Any] | None = None,
    progress_provider: Callable[[], float] | None = None,
    detail_provider: Callable[[int], str] | None = None,
    metrics_provider: Callable[[], dict[str, Any]] | None = None,
) -> Iterable[None]:
    """Keep opaque native COLMAP calls visibly alive without inventing ETA."""
    started = time.perf_counter()
    stopped = threading.Event()
    base_metrics = dict(metrics or {})

    def publish_once(elapsed: int) -> None:
        minutes, seconds = divmod(elapsed, 60)
        elapsed_label = f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"
        live_metrics = metrics_provider() if metrics_provider is not None else {}
        _progress(
            project_root,
            stage,
            (
                detail_provider(elapsed)
                if detail_provider is not None
                else f"{detail} · {elapsed_label} elapsed"
            ),
            progress_provider() if progress_provider is not None else progress,
            compute_backend=compute_backend,
            metrics={**base_metrics, **live_metrics, "elapsedSeconds": elapsed},
        )

    def publish() -> None:
        while not stopped.wait(1.0):
            publish_once(round(time.perf_counter() - started))

    publish_once(0)
    thread = threading.Thread(target=publish, name=f"scanlan-{stage}-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=2.0)


def _check_cancelled(project_root: Path) -> None:
    if (project_root / "outputs" / "cancel.flag").is_file():
        raise RuntimeError("Media reconstruction cancelled")


def _media_sources(project_root: Path, explicit: Sequence[Path]) -> list[Path]:
    if explicit:
        sources = [path.resolve() for path in explicit]
    else:
        project_path = project_root / "project.json"
        if not project_path.is_file():
            raise FileNotFoundError(f"Project manifest is missing: {project_path}")
        project = json.loads(project_path.read_text(encoding="utf-8"))
        sources = []
        for entry in project.get("mediaSources", []):
            relative = Path(str(entry.get("path", "")))
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Project contains an unsafe media source path")
            sources.append((project_root / relative).resolve())
    if not sources:
        raise ValueError("Choose photos or a video before building a Gaussian splat")
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Media source is missing: {missing[0]}")
    unsupported = [str(path) for path in sources if path.suffix.lower() not in PHOTO_EXTENSIONS | VIDEO_EXTENSIONS]
    if unsupported:
        raise ValueError(f"Unsupported photo or video format: {unsupported[0]}")
    return sources


def _source_fingerprint(sources: Sequence[Path], options: MediaPreparationOptions) -> str:
    """Fingerprint only work needed to decode and select canonical media views."""
    digest = hashlib.sha256()
    digest.update(b"scanlan-media-observations-v2-adaptive-flow-jpeg95\0")
    digest.update(
        json.dumps(
            {
                "videoFps": options.video_fps,
                "maximumVideoFrames": options.maximum_video_frames,
                "maximumImageDimension": options.maximum_image_dimension,
                "minimumImageDimension": options.minimum_image_dimension,
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    for path in sources:
        stat = path.stat()
        digest.update(path.name.lower().encode("utf-8", errors="replace"))
        digest.update(stat.st_size.to_bytes(8, "little", signed=False))
        # Content at both ends catches replaced videos without hashing multi-GB
        # captures on every resume. Photos are hashed completely.
        with path.open("rb") as handle:
            if stat.st_size <= 16 * 1024 * 1024:
                digest.update(handle.read())
            else:
                digest.update(handle.read(8 * 1024 * 1024))
                handle.seek(max(0, stat.st_size - 8 * 1024 * 1024))
                digest.update(handle.read(8 * 1024 * 1024))
    return digest.hexdigest()


def _media_dataset_fingerprint(
    observation_fingerprint: str,
    options: MediaPreparationOptions,
) -> str:
    """Fingerprint camera analysis independently from expensive media decoding."""
    digest = hashlib.sha256()
    # P10 extends the learned-first dataset with a versioned dense-fusion
    # sidecar. Older solutions lack independent fusion confidence/ownership.
    digest.update(b"scanlan-media-dataset-v20-gaussian-init-contract\0")
    digest.update(observation_fingerprint.encode("ascii"))
    digest.update(DA3_CODE_REVISION.encode("ascii"))
    digest.update(DA3_MODEL_REVISION.encode("ascii"))
    digest.update(DA3_MODEL_SHA256.encode("ascii"))
    digest.update(f"{LINGBOT_CONTEXT_FPS}:{LINGBOT_MAX_CONTEXT_FRAMES}".encode("ascii"))
    digest.update(str(options.maximum_features).encode("ascii"))
    return digest.hexdigest()


def _limited_size(width: int, height: int, maximum_dimension: int) -> tuple[int, int]:
    scale = min(1.0, maximum_dimension / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def _save_canonical_image(
    image: Image.Image,
    destination: Path,
    maximum_dimension: int,
    *,
    exif: bytes | None = None,
) -> tuple[int, int]:
    image, _color_metadata = to_canonical_srgb(image)
    size = _limited_size(image.width, image.height, maximum_dimension)
    if size != image.size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_options: dict[str, Any] = {
        "format": "JPEG",
        "quality": 95,
        "subsampling": 0,
        "optimize": False,
    }
    if exif:
        save_options["exif"] = exif
    save_options["icc_profile"] = image.info["icc_profile"]
    image.save(destination, **save_options)
    return image.size


def _sharpness_and_descriptor(image: Image.Image) -> tuple[float, np.ndarray]:
    sample = ImageOps.grayscale(image)
    sample.thumbnail((320, 320), Image.Resampling.LANCZOS)
    pixels = np.asarray(sample, dtype=np.float32) / 255.0
    if min(pixels.shape) < 3:
        return 0.0, np.zeros(256, dtype=np.float32)
    laplacian = (
        -4.0 * pixels[1:-1, 1:-1]
        + pixels[:-2, 1:-1]
        + pixels[2:, 1:-1]
        + pixels[1:-1, :-2]
        + pixels[1:-1, 2:]
    )
    sharpness = float(np.var(laplacian))
    descriptor_image = sample.resize((16, 16), Image.Resampling.BILINEAR)
    descriptor = np.asarray(descriptor_image, dtype=np.float32).reshape(-1) / 255.0
    descriptor -= float(np.mean(descriptor))
    norm = float(np.linalg.norm(descriptor))
    if norm > 1e-8:
        descriptor /= norm
    return sharpness, descriptor


def _descriptor_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(1.0 - np.clip(np.dot(left, right), -1.0, 1.0))


def _tracked_visual_motion(
    reference: np.ndarray,
    current: np.ndarray,
) -> dict[str, float | int | bool]:
    """Measure retained visual overlap and image motion with forward/backward LK.

    The small grayscale inputs make this cheap enough to evaluate throughout a
    long video. Forward/backward validation rejects unstable tracks from blur,
    occlusion, and independently moving objects instead of mistaking them for
    useful camera overlap.
    """
    import cv2

    if reference.ndim != 2 or current.ndim != 2:
        raise ValueError("Adaptive video tracking requires grayscale images")
    if current.shape != reference.shape:
        current = cv2.resize(
            current,
            (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    corners = cv2.goodFeaturesToTrack(
        reference,
        maxCorners=600,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=5,
    )
    feature_count = 0 if corners is None else len(corners)
    if feature_count < 24:
        return {
            "reliable": False,
            "featureCount": feature_count,
            "trackedCount": 0,
            "trackedRatio": 0.0,
            "coverageRatio": 0.0,
            "overlapScore": 0.0,
            "medianMotion": 0.0,
            "p90Motion": 0.0,
        }
    flow_options = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    }
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        reference,
        current,
        corners,
        None,
        **flow_options,
    )
    if forward is None or forward_status is None:
        return {
            "reliable": False,
            "featureCount": feature_count,
            "trackedCount": 0,
            "trackedRatio": 0.0,
            "coverageRatio": 0.0,
            "overlapScore": 0.0,
            "medianMotion": 0.0,
            "p90Motion": 0.0,
        }
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current,
        reference,
        forward,
        None,
        **flow_options,
    )
    if backward is None or backward_status is None:
        return {
            "reliable": False,
            "featureCount": feature_count,
            "trackedCount": 0,
            "trackedRatio": 0.0,
            "coverageRatio": 0.0,
            "overlapScore": 0.0,
            "medianMotion": 0.0,
            "p90Motion": 0.0,
        }
    source_points = corners.reshape(-1, 2)
    target_points = forward.reshape(-1, 2)
    recovered_points = backward.reshape(-1, 2)
    height, width = reference.shape
    forward_backward_error = np.linalg.norm(recovered_points - source_points, axis=1)
    valid = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & np.isfinite(target_points).all(axis=1)
        & (forward_backward_error <= 1.5)
        & (target_points[:, 0] >= 0.0)
        & (target_points[:, 0] < width)
        & (target_points[:, 1] >= 0.0)
        & (target_points[:, 1] < height)
    )
    tracked_count = int(np.count_nonzero(valid))
    tracked_ratio = tracked_count / feature_count
    grid_columns = 8
    grid_rows = 6
    source_cells = (
        np.clip((source_points[:, 1] * grid_rows / max(height, 1)).astype(np.int32), 0, grid_rows - 1)
        * grid_columns
        + np.clip((source_points[:, 0] * grid_columns / max(width, 1)).astype(np.int32), 0, grid_columns - 1)
    )
    occupied_cells = np.unique(source_cells)
    tracked_cells = np.unique(source_cells[valid])
    coverage_ratio = len(tracked_cells) / max(len(occupied_cells), 1)
    if not tracked_count:
        median_motion = 0.0
        p90_motion = 0.0
    else:
        diagonal = max(float(math.hypot(width, height)), 1.0)
        motion = np.linalg.norm(target_points[valid] - source_points[valid], axis=1) / diagonal
        median_motion = float(np.median(motion))
        p90_motion = float(np.percentile(motion, 90))
    return {
        "reliable": True,
        "featureCount": feature_count,
        "trackedCount": tracked_count,
        "trackedRatio": tracked_ratio,
        "coverageRatio": coverage_ratio,
        "overlapScore": min(tracked_ratio, coverage_ratio),
        "medianMotion": median_motion,
        "p90Motion": p90_motion,
    }


def _adaptive_keyframe_reason(
    elapsed_seconds: float,
    motion: dict[str, float | int | bool],
    descriptor_distance: float,
    *,
    minimum_gap_seconds: float = ADAPTIVE_MINIMUM_GAP_SECONDS,
) -> str | None:
    """Return why the candidate should become a keyframe, if it should."""
    if elapsed_seconds < minimum_gap_seconds:
        return None
    if elapsed_seconds >= ADAPTIVE_MAXIMUM_GAP_SECONDS:
        return "maximum_gap"
    if bool(motion["reliable"]):
        if float(motion["overlapScore"]) <= ADAPTIVE_TARGET_TRACKED_RATIO:
            return "tracked_overlap"
        if float(motion["medianMotion"]) >= ADAPTIVE_TARGET_MEDIAN_MOTION:
            return "camera_motion"
        if float(motion["p90Motion"]) >= ADAPTIVE_TARGET_P90_MOTION:
            return "parallax"
        return None
    # Very low-texture frames cannot support a trustworthy LK estimate. The
    # coarse exposure-invariant descriptor still prevents a long blind gap.
    return "visual_change" if descriptor_distance >= 0.12 else None


def _extract_photo(
    source: Path,
    destination: Path,
    maximum_dimension: int,
) -> dict[str, Any]:
    with Image.open(source) as opened:
        exif = opened.getexif()
        orientation = int(exif.get(274, 1))
        exif[274] = 1
        width, height = _save_canonical_image(
            opened,
            destination,
            maximum_dimension,
            exif=exif.tobytes(),
        )
        with Image.open(destination) as normalized:
            sharpness, descriptor = _sharpness_and_descriptor(normalized)
    return {
        "source": str(source),
        "image": destination.name,
        "width": width,
        "height": height,
        "orientation": orientation,
        "sharpness": sharpness,
        "descriptor": descriptor,
        "timestampSeconds": None,
    }


def _extract_video_streaming(
    source: Path,
    images_root: Path,
    first_frame_index: int,
    options: MediaPreparationOptions,
    maximum_frames: int,
    project_root: Path,
    progress: Callable[[float, str, int | None, dict[str, Any]], None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select motion-adaptive video views with constant full-frame memory.

    The previous implementation retained every evaluated RGB frame until the
    entire video had decoded. A few minutes of 4K video could therefore retain
    many gigabytes and turn normal decoding into paging. This version holds a
    small grayscale reference plus at most one pending decoded frame. Keyframes
    are driven by tracked visual overlap and normalized motion instead of video
    duration or a fixed output FPS.
    """
    try:
        import av
    except ModuleNotFoundError as error:
        raise RuntimeError("Video import requires the bundled PyAV runtime") from error

    container = av.open(str(source))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None and stream.time_base is not None
            else None
        )
        source_rate = float(stream.average_rate) if stream.average_rate else None
        analysis_fps = max(options.video_fps, 1e-4)
        if source_rate:
            analysis_fps = min(analysis_fps, source_rate)
        evaluation_interval = 1.0 / max(analysis_fps, 1e-6)
        minimum_gap = ADAPTIVE_MINIMUM_GAP_SECONDS
        if duration and duration > 0.0 and maximum_frames > 1:
            # Only the crash-safety ceiling may relax the fastest permitted
            # selection rate. Below that ceiling, selection is purely visual.
            minimum_gap = max(minimum_gap, duration / (maximum_frames - 1))
        next_evaluation = 0.0
        reference_gray: np.ndarray | None = None
        reference_descriptor: np.ndarray | None = None
        reference_timestamp: float | None = None
        pending: tuple[float, Any, float, np.ndarray, np.ndarray] | None = None
        pending_motion: dict[str, float | int | bool] | None = None
        records: list[dict[str, Any]] = []
        decoded = 0
        evaluated = 0
        selection_reasons: dict[str, int] = {}
        tracked_overlaps: list[float] = []
        started = time.perf_counter()
        last_progress_at = 0.0

        def write_selected(
            candidate: tuple[float, Any, float, np.ndarray, np.ndarray],
            reason: str,
            motion: dict[str, float | int | bool] | None,
        ) -> bool:
            nonlocal reference_gray, reference_descriptor, reference_timestamp
            if len(records) >= maximum_frames:
                return False
            timestamp, frame, sharpness, descriptor, gray = candidate
            output_size = _limited_size(
                int(frame.width),
                int(frame.height),
                options.maximum_image_dimension,
            )
            image = frame.to_image(width=output_size[0], height=output_size[1]).convert("RGB")
            destination = images_root / f"video-{first_frame_index + len(records):06d}.jpg"
            width, height = _save_canonical_image(
                image,
                destination,
                options.maximum_image_dimension,
            )
            if min(width, height) < options.minimum_image_dimension:
                destination.unlink(missing_ok=True)
                return False
            tracked_ratio = (
                float(motion["overlapScore"])
                if motion is not None and bool(motion["reliable"])
                else None
            )
            records.append(
                {
                    "source": str(source),
                    "image": destination.name,
                    "width": width,
                    "height": height,
                    "orientation": 1,
                    "sharpness": sharpness,
                    "descriptor": descriptor,
                    "timestampSeconds": timestamp,
                    "selectionReason": reason,
                    "trackedOverlap": tracked_ratio,
                }
            )
            selection_reasons[reason] = selection_reasons.get(reason, 0) + 1
            if tracked_ratio is not None:
                tracked_overlaps.append(tracked_ratio)
            reference_gray = gray.copy()
            reference_descriptor = descriptor.copy()
            reference_timestamp = timestamp
            return True

        for frame in container.decode(stream):
            decoded += 1
            if decoded % 30 == 0:
                _check_cancelled(project_root)
            timestamp = float(frame.time) if frame.time is not None else (
                decoded / source_rate if source_rate else float(decoded)
            )
            now = time.perf_counter()
            if now - last_progress_at >= 0.5:
                fraction = (
                    float(np.clip(timestamp / duration, 0.0, 1.0))
                    if duration and duration > 0.0
                    else 0.0
                )
                elapsed = now - started
                eta = (
                    max(0, round(elapsed * (1.0 - fraction) / fraction))
                    if fraction >= 0.02 and fraction < 1.0
                    else None
                )
                detail = (
                    f"Scanning {source.name} · {timestamp:.1f}s of {duration:.1f}s"
                    if duration
                    else f"Scanning {source.name} · {decoded:,} decoded frames"
                )
                progress(
                    fraction,
                    detail,
                    eta,
                    {
                        "decodedFrames": decoded,
                        "evaluatedFrames": evaluated,
                        "selectedFrames": len(records),
                        "sourceTimestampSeconds": round(timestamp, 3),
                        "sourceDurationSeconds": duration,
                        "selectionMode": "adaptive optical flow",
                    },
                )
                last_progress_at = now
            if timestamp + 1e-6 < next_evaluation:
                continue
            next_evaluation = timestamp + evaluation_interval
            sample_size = _limited_size(int(frame.width), int(frame.height), 320)
            sample = frame.to_image(width=sample_size[0], height=sample_size[1]).convert("RGB")
            sharpness, descriptor = _sharpness_and_descriptor(sample)
            gray = np.asarray(ImageOps.grayscale(sample), dtype=np.uint8)
            evaluated += 1
            candidate = (timestamp, frame, sharpness, descriptor, gray)
            if reference_gray is None:
                write_selected(candidate, "first_frame", None)
                pending = None
                pending_motion = None
                continue
            assert reference_timestamp is not None
            assert reference_descriptor is not None
            motion = _tracked_visual_motion(reference_gray, gray)
            elapsed = timestamp - reference_timestamp

            # If the current candidate has already crossed the hard continuity
            # floor, retain the preceding evaluated frame—the last known view
            # with safer overlap—then assess the current one against that new
            # reference. Fast motion therefore creates closer-spaced views.
            if (
                bool(motion["reliable"])
                and float(motion["overlapScore"]) < ADAPTIVE_HARD_TRACKED_RATIO
                and pending is not None
                and pending_motion is not None
                and pending[0] - reference_timestamp >= minimum_gap
            ):
                write_selected(pending, "overlap_guard", pending_motion)
                assert reference_gray is not None
                assert reference_descriptor is not None
                assert reference_timestamp is not None
                motion = _tracked_visual_motion(reference_gray, gray)
                elapsed = timestamp - reference_timestamp
                pending = None
                pending_motion = None

            reason = _adaptive_keyframe_reason(
                elapsed,
                motion,
                _descriptor_distance(reference_descriptor, descriptor),
                minimum_gap_seconds=minimum_gap,
            )
            if (
                reason is not None
                and pending is not None
                and pending_motion is not None
                and pending[0] - reference_timestamp >= minimum_gap
                and bool(pending_motion["reliable"])
                and float(pending_motion["overlapScore"]) >= ADAPTIVE_HARD_TRACKED_RATIO
                and sharpness < pending[2] * 0.70
            ):
                # A blurred frame often appears to have moved because its
                # tracks disappear. Prefer the immediately preceding sharp,
                # well-overlapped view, then re-evaluate the current frame from
                # that safer reference instead of training on motion blur.
                write_selected(pending, "sharpness_guard", pending_motion)
                assert reference_gray is not None
                assert reference_descriptor is not None
                assert reference_timestamp is not None
                motion = _tracked_visual_motion(reference_gray, gray)
                elapsed = timestamp - reference_timestamp
                reason = _adaptive_keyframe_reason(
                    elapsed,
                    motion,
                    _descriptor_distance(reference_descriptor, descriptor),
                    minimum_gap_seconds=minimum_gap,
                )
                pending = None
                pending_motion = None
            if reason is not None:
                write_selected(candidate, reason, motion)
                pending = None
                pending_motion = None
            else:
                pending = candidate
                pending_motion = motion

        if (
            pending is not None
            and reference_timestamp is not None
            and pending[0] - reference_timestamp >= minimum_gap
        ):
            write_selected(pending, "last_frame", pending_motion)
        effective_selection_fps = (
            len(records) / duration
            if duration and duration > 0.0
            else None
        )
        progress(
            1.0,
            f"Selected {len(records):,} motion-adaptive keyframes from {source.name}",
            0,
            {
                "decodedFrames": decoded,
                "evaluatedFrames": evaluated,
                "selectedFrames": len(records),
                "redundantFrames": max(0, evaluated - len(records)),
                "sourceDurationSeconds": duration,
                "selectionMode": "adaptive optical flow",
                "selectionReasons": selection_reasons,
            },
        )
        return records, {
            "path": str(source),
            "durationSeconds": duration,
            "sourceFps": source_rate,
            "analysisFps": analysis_fps,
            "effectiveSelectionFps": effective_selection_fps,
            "decodedFrameCount": decoded,
            "candidateFrameCount": evaluated,
            "selectedFrameCount": len(records),
            "redundantFrameCount": max(0, evaluated - len(records)),
            "selectionMode": "adaptive_optical_flow",
            "selectionReasons": selection_reasons,
            "medianTrackedOverlap": (
                float(np.median(tracked_overlaps)) if tracked_overlaps else None
            ),
            "minimumTrackedOverlap": min(tracked_overlaps, default=None),
            "frameLimitReached": len(records) >= maximum_frames,
        }
    finally:
        container.close()


def _extract_lingbot_context(
    source: Path,
    records: Sequence[dict[str, Any]],
    input_images: Path,
    context_root: Path,
    project_root: Path,
) -> tuple[list[Path], list[int], dict[str, Any]]:
    """Decode a smooth video stream while preserving exact training views.

    COLMAP benefits from a sparse set of sharp, high-resolution frames. A
    streaming trajectory model has the opposite requirement: small temporal
    steps. Build a 10 FPS low-resolution context stream, replace the nearest
    slots with ScanLan's exact selected frames, and remember which predictions
    belong to the Gaussian training set.
    """
    try:
        import av
    except ModuleNotFoundError as error:
        raise RuntimeError("LingBot video context requires the bundled PyAV runtime") from error
    if not records:
        raise ValueError("LingBot video context requires selected training views")
    container = av.open(str(source))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        source_rate = float(stream.average_rate) if stream.average_rate else 30.0
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None and stream.time_base is not None
            else max(float(record.get("timestampSeconds") or 0.0) for record in records)
        )
        effective_fps = LINGBOT_CONTEXT_FPS
        if duration > 0.0:
            effective_fps = min(
                effective_fps,
                max(1.0, (LINGBOT_MAX_CONTEXT_FRAMES - 1) / duration),
            )
        regular_count = max(3, min(
            LINGBOT_MAX_CONTEXT_FRAMES,
            int(math.floor(max(duration, 0.0) * effective_fps)) + 1,
        ))
        schedule: list[dict[str, Any]] = [
            {
                "timestamp": index / effective_fps,
                "trainingIndex": None,
                "path": None,
            }
            for index in range(regular_count)
        ]
        # Replace the closest regular slot with the exact sharp frame used by
        # COLMAP/training. This makes returned depth and color pixel-identical
        # to the selected observation while surrounding it with smooth motion.
        for training_index, record in enumerate(records):
            timestamp = float(record.get("timestampSeconds") or 0.0)
            nearest = min(
                range(len(schedule)),
                key=lambda index: abs(float(schedule[index]["timestamp"]) - timestamp),
            )
            if schedule[nearest]["trainingIndex"] is not None:
                schedule.append(
                    {"timestamp": timestamp, "trainingIndex": training_index, "path": None}
                )
                target = schedule[-1]
            else:
                target = schedule[nearest]
            target["timestamp"] = timestamp
            target["trainingIndex"] = training_index
            target["path"] = input_images / str(record["image"])
        schedule.sort(key=lambda item: (float(item["timestamp"]), item["trainingIndex"] is None))
        context_root.mkdir(parents=True, exist_ok=False)
        pending = [item for item in schedule if item["path"] is None]
        pending_index = 0
        decoded = 0
        saved = 0
        last_report = 0.0
        processed_width, processed_height = lingbot_processed_size(
            int(stream.codec_context.width),
            int(stream.codec_context.height),
        )
        last_image: Image.Image | None = None
        for frame in container.decode(stream):
            decoded += 1
            if decoded % 60 == 0:
                _check_cancelled(project_root)
            timestamp = float(frame.time) if frame.time is not None else decoded / source_rate
            while pending_index < len(pending) and timestamp + 1e-6 >= float(
                pending[pending_index]["timestamp"]
            ):
                image = frame.to_image(
                    width=processed_width,
                    height=processed_height,
                ).convert("RGB")
                destination = context_root / f"context-{saved:06d}.jpg"
                image.save(destination, format="JPEG", quality=95, subsampling=0)
                pending[pending_index]["path"] = destination
                last_image = image
                pending_index += 1
                saved += 1
            now = time.perf_counter()
            if now - last_report >= 0.75:
                fraction = min(1.0, timestamp / max(duration, 1e-6))
                _progress(
                    project_root,
                    "lingbot_context",
                    f"Decoding continuous LingBot context · {timestamp:.1f}s of {duration:.1f}s",
                    0.685 + 0.005 * fraction,
                    compute_backend="PyAV 10 FPS LingBot context",
                    metrics={
                        "decodedFrames": decoded,
                        "contextFrames": saved + len(records),
                        "effectiveContextFps": effective_fps,
                    },
                )
                last_report = now
            if pending_index >= len(pending):
                break
        if pending_index < len(pending):
            if last_image is None:
                raise RuntimeError("Video decoder produced no LingBot context frames")
            while pending_index < len(pending):
                destination = context_root / f"context-{saved:06d}.jpg"
                last_image.save(destination, format="JPEG", quality=95, subsampling=0)
                pending[pending_index]["path"] = destination
                pending_index += 1
                saved += 1
        paths = [Path(item["path"]) for item in schedule]
        output_indices = [
            index
            for index, item in enumerate(schedule)
            if item["trainingIndex"] is not None
        ]
        training_order = [
            int(schedule[index]["trainingIndex"])
            for index in output_indices
        ]
        if training_order != list(range(len(records))):
            raise RuntimeError("LingBot context did not preserve selected video frame order")
        return paths, output_indices, {
            "contextFrameCount": len(paths),
            "contextFps": effective_fps,
            "decodedFrameCount": decoded,
            "trainingViewCount": len(records),
        }
    finally:
        container.close()


def _collect_images(
    sources: Sequence[Path],
    images_root: Path,
    options: MediaPreparationOptions,
    project_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    videos: list[dict[str, Any]] = []
    photo_sources = [path for path in sources if path.suffix.lower() in PHOTO_EXTENSIONS]
    video_sources = [path for path in sources if path.suffix.lower() in VIDEO_EXTENSIONS]
    total_sources = max(1, len(photo_sources) + len(video_sources))
    for source_index, source in enumerate(photo_sources):
        _check_cancelled(project_root)
        destination = images_root / f"photo-{source_index:06d}.jpg"
        record = _extract_photo(source, destination, options.maximum_image_dimension)
        if min(record["width"], record["height"]) < options.minimum_image_dimension:
            destination.unlink(missing_ok=True)
            continue
        records.append(record)
        _progress(
            project_root,
            "media_decode",
            f"Prepared photo {source_index + 1:,} of {len(photo_sources):,}",
            0.08 * (source_index + 1) / total_sources,
        )
    video_frame_index = 0
    for video_index, source in enumerate(video_sources):
        _check_cancelled(project_root)
        maximum_frames = max(
            2,
            options.maximum_video_frames // max(1, len(video_sources)),
        )
        def report_video_progress(
            source_progress: float,
            detail: str,
            eta: int | None,
            metrics: dict[str, Any],
        ) -> None:
            completed_sources = len(photo_sources) + video_index
            _progress(
                project_root,
                "media_decode",
                detail,
                0.08
                * (completed_sources + source_progress)
                / total_sources,
                stage_eta_seconds=eta,
                compute_backend="PyAV multithreaded video decode",
                metrics={
                    **metrics,
                    "sourceIndex": completed_sources + 1,
                    "sourceCount": total_sources,
                },
            )

        video_records, video = _extract_video_streaming(
            source,
            images_root,
            video_frame_index,
            options,
            maximum_frames,
            project_root,
            report_video_progress,
        )
        records.extend(video_records)
        video_frame_index += len(video_records)
        videos.append(video)
        _progress(
            project_root,
            "media_decode",
            f"Selected {len(video_records):,} sharp keyframes from video {video_index + 1:,} of {len(video_sources):,}",
            0.08
            * (len(photo_sources) + video_index + 1)
            / total_sources,
            compute_backend="PyAV multithreaded video decode",
            metrics={
                "selectedFrames": len(video_records),
                "decodedFrames": video["decodedFrameCount"],
                "sourceIndex": len(photo_sources) + video_index + 1,
                "sourceCount": total_sources,
            },
        )
    if len(records) < 3:
        raise ValueError("At least three usable, overlapping images are required")
    return records, videos


def _configure_sfm(
    image_count: int,
    maximum_features: int,
    use_cuda: bool,
    sequential: bool,
    maximum_image_dimension: int = 2560,
    feature_runtime: dict[str, Any] | None = None,
) -> tuple[Any, ...]:
    import pycolmap

    available_threads = os.cpu_count() or 1
    worker_threads = max(1, available_threads - 2 if available_threads > 4 else available_threads)

    reader = pycolmap.ImageReaderOptions()
    # A single radial parameter is substantially better conditioned than the
    # full OPENCV model for casual phone sweeps with only a few dozen views.
    # COLMAP still refines it, then the canonicalizer removes it completely.
    reader.camera_model = "SIMPLE_RADIAL"
    reader.default_focal_length_factor = 1.2

    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = maximum_image_dimension
    extraction.num_threads = worker_threads
    extraction.use_gpu = use_cuda
    learned_features = bool(feature_runtime and feature_runtime.get("learnedValidated"))
    if learned_features:
        extraction.type = pycolmap.FeatureExtractorType.ALIKED_N16ROT
        extraction.aliked.max_num_features = maximum_features
        extraction.aliked.n16rot_model_path = str(feature_runtime["alikedModel"])
    else:
        extraction.sift.max_num_features = maximum_features
        extraction.sift.peak_threshold = 0.004
        extraction.sift.edge_threshold = 12.0
        extraction.sift.max_num_orientations = 1
        # Covariant affine/DSP SIFT is prohibitively slow in the CPU-only
        # Windows build. Dense standard SIFT plus guided geometric matching is
        # the validated fallback when learned ONNX matching is unavailable.
        extraction.sift.estimate_affine_shape = False
        extraction.sift.domain_size_pooling = False

    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = worker_threads
    matching.use_gpu = use_cuda
    matching.max_num_matches = 32_768
    if learned_features:
        matching.type = pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE
        matching.aliked.lightglue.model_path = str(feature_runtime["lightglueModel"])
        # COLMAP does not support its SIFT-only guided rematch for ALIKED.
        matching.guided_matching = False
    else:
        matching.guided_matching = True
        matching.sift.max_ratio = 0.85
        matching.sift.cross_check = True

    verification = pycolmap.TwoViewGeometryOptions()
    verification.min_num_inliers = 20
    verification.ransac.max_error = 3.0
    verification.ransac.confidence = 0.9999

    mapping = pycolmap.IncrementalPipelineOptions()
    mapping.num_threads = worker_threads
    mapping.multiple_models = True
    # Long videos often contain several small disconnected fragments. Three
    # candidate models are enough to recover the dominant path without spending
    # another minute repeatedly retrying tiny four-frame islands.
    mapping.max_num_models = 3 if sequential else min(10, max(2, image_count // 10))
    mapping.min_model_size = min(12, max(3, image_count // 4))
    if sequential:
        # Local BA still runs as cameras are added, while less-frequent global
        # BA avoids solving nearly the same growing system after every few
        # video frames. COLMAP performs a final global refinement per model.
        mapping.ba_global_frames_ratio = 1.25
        mapping.ba_global_points_ratio = 1.25
    mapping.ba_global_max_num_iterations = 50
    mapping.ba_global_max_refinements = 3
    mapping.ba_global_function_tolerance = 1e-7
    mapping.max_runtime_seconds = round(max(120.0, min(600.0, image_count * 1.25)))
    mapping.mapper.init_min_num_inliers = min(100, max(40, maximum_features // 100))
    mapping.mapper.init_min_tri_angle = 8.0
    mapping.mapper.abs_pose_min_num_inliers = 20
    mapping.mapper.abs_pose_min_inlier_ratio = 0.15
    mapping.mapper.filter_max_reproj_error = 3.0
    mapping.mapper.filter_min_tri_angle = 1.0
    mapping.mapper.max_reg_trials = 5
    mapping.mapper.num_threads = worker_threads
    mapping.triangulation.min_angle = 1.0
    return reader, extraction, matching, verification, mapping


def _minimum_useful_registration_count(image_count: int) -> int:
    """Reject small fragments while tolerating a brief break in a long video."""
    return max(3, math.ceil(image_count * 0.45))


def _feature_extraction_groups(
    records: Sequence[dict[str, Any]],
) -> list[tuple[list[str], bool]]:
    """Group each locked video behind one camera while batching photos together.

    COLMAP's AUTO mode creates a separate camera for generated video JPEGs,
    because they intentionally carry no still-camera EXIF identity. Letting
    bundle adjustment refine focal length and distortion independently for
    every frame makes the scene elastic and is catastrophic for Gaussian
    densification. Each source video therefore gets one SINGLE-camera group.
    """
    video_groups: dict[str, list[str]] = {}
    photos: list[str] = []
    for record in records:
        image_name = str(record["image"])
        source = Path(str(record.get("source", "")))
        if source.suffix.lower() in VIDEO_EXTENSIONS:
            source_key = os.path.normcase(str(source.resolve()))
            video_groups.setdefault(source_key, []).append(image_name)
        else:
            photos.append(image_name)
    groups = [
        (sorted(names), True)
        for names in video_groups.values()
        if names
    ]
    if photos:
        groups.append((sorted(photos), False))
    return groups


def _video_intrinsic_spread(frames: Sequence[dict[str, Any]]) -> float:
    """Return the largest normalized-intrinsic spread within one video source."""
    groups: dict[str, list[tuple[float, float, float, float]]] = {}
    for frame in frames:
        source = Path(str(frame.get("sourcePath", "")))
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        intrinsics = frame["intrinsics"]
        width = float(intrinsics["width"])
        height = float(intrinsics["height"])
        groups.setdefault(os.path.normcase(str(source.resolve())), []).append(
            (
                float(intrinsics["fx"]) / width,
                float(intrinsics["fy"]) / height,
                float(intrinsics["cx"]) / width,
                float(intrinsics["cy"]) / height,
            )
        )
    maximum_spread = 0.0
    for values in groups.values():
        matrix = np.asarray(values, dtype=np.float64)
        median = np.median(matrix, axis=0)
        relative = np.abs(matrix - median) / np.maximum(np.abs(median), 1e-9)
        maximum_spread = max(maximum_spread, float(np.max(relative)))
    return maximum_spread


def _cpu_match_pairs(
    image_names: list[str],
    *,
    sequential: bool,
) -> list[tuple[str, str]]:
    """Create a bounded video graph or a complete unordered-photo graph."""
    pairs: list[tuple[str, str]] = []
    if sequential:
        offsets = [*range(1, 17), 32, 64, 128]
        for left in range(len(image_names)):
            for offset in offsets:
                right = left + offset
                if right < len(image_names):
                    pairs.append((image_names[left], image_names[right]))
        return pairs
    for left in range(len(image_names)):
        pairs.extend(
            (image_names[left], image_names[right])
            for right in range(left + 1, len(image_names))
        )
    return pairs


def _learned_camera_arrays(
    proposal: LingbotGeometry | None,
    image_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return finite learned camera centers, forward axes, and confidence.

    Learned cameras are proposals only.  They may decide which expensive
    high-resolution pairs to verify, but never become observations in the SfM
    model and never bypass COLMAP's two-view geometry or bundle adjustment.
    """
    if proposal is None:
        return None
    poses = np.asarray(proposal.world_from_cameras, dtype=np.float64)
    confidence = np.asarray(proposal.frame_confidence, dtype=np.float64).reshape(-1)
    if poses.shape != (image_count, 4, 4) or confidence.shape != (image_count,):
        return None
    if not np.isfinite(poses).all() or not np.isfinite(confidence).all():
        return None
    rotations = poses[:, :3, :3]
    determinants = np.linalg.det(rotations)
    if np.any(np.abs(determinants - 1.0) > 0.08):
        return None
    centers = poses[:, :3, 3]
    forward = rotations[:, :, 2]
    forward_norm = np.linalg.norm(forward, axis=1, keepdims=True)
    if np.any(forward_norm <= 1e-8):
        return None
    forward = forward / forward_norm
    if image_count > 1 and float(np.max(np.linalg.norm(centers - centers[0], axis=1))) <= 1e-7:
        return None
    return centers, forward, np.clip(confidence, 0.0, 1.0)


def _guided_match_pairs(
    image_names: Sequence[str],
    proposal: LingbotGeometry | None,
    *,
    sequential: bool,
) -> list[tuple[str, str]]:
    """Build a bounded, connected pair graph from a learned trajectory.

    The graph combines spatial nearest neighbours, view-direction agreement,
    and temporal baselines.  Connecting every view to an earlier spatial
    neighbour prevents an otherwise plausible learned trajectory from
    producing disconnected matching islands.
    """
    names = list(image_names)
    learned = _learned_camera_arrays(proposal, len(names))
    if learned is None:
        return []
    centers, forward, confidence = learned
    pair_indices: set[tuple[int, int]] = set()
    neighbour_count = min(len(names) - 1, 12 if sequential else 10)
    for index in range(len(names)):
        distances = np.linalg.norm(centers - centers[index], axis=1)
        direction_agreement = np.clip(forward @ forward[index], -1.0, 1.0)
        confidence_penalty = 0.35 * (1.0 - np.minimum(confidence, confidence[index]))
        nonzero = distances[distances > 1e-7]
        local_scale = float(np.median(nonzero)) if len(nonzero) else 1.0
        score = distances / max(local_scale, 1e-7)
        score += 0.65 * (1.0 - direction_agreement) + confidence_penalty
        score[index] = math.inf
        nearest = np.argpartition(score, neighbour_count - 1)[:neighbour_count]
        for other in nearest:
            left, right = sorted((index, int(other)))
            pair_indices.add((left, right))
        if index:
            earlier = np.linalg.norm(centers[:index] - centers[index], axis=1)
            pair_indices.add((int(np.argmin(earlier)), index))

    if sequential:
        # Local continuity plus widening baselines supplies parallax and loop
        # evidence without the quadratic graph used by exhaustive matching.
        for offset in (*range(1, 9), 16, 32, 64, 128):
            pair_indices.update(
                (left, left + offset)
                for left in range(len(names) - offset)
            )
    return [(names[left], names[right]) for left, right in sorted(pair_indices)]


def _camera_recovery_pairs(
    image_names: Sequence[str],
    proposal: LingbotGeometry | None,
    registered_names: set[str],
    existing_pairs: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Connect missing learned cameras to nearby verified SfM cameras."""
    names = list(image_names)
    learned = _learned_camera_arrays(proposal, len(names))
    registered_indices = [
        index for index, name in enumerate(names) if name in registered_names
    ]
    if learned is None or len(registered_indices) < 2:
        return []
    centers, forward, _confidence = learned
    recovered: set[tuple[str, str]] = set()
    registered_array = np.asarray(registered_indices, dtype=np.int64)
    for index, name in enumerate(names):
        if name in registered_names:
            continue
        distances = np.linalg.norm(centers[registered_array] - centers[index], axis=1)
        angles = 1.0 - np.clip(forward[registered_array] @ forward[index], -1.0, 1.0)
        score = distances / max(float(np.median(distances)), 1e-7) + 0.75 * angles
        # Reach beyond the first-pass neighbourhood. Otherwise a single
        # missing view would only request the same pairs that already failed
        # to connect it to the reconstruction.
        count = min(20, len(registered_array))
        neighbours = registered_array[np.argpartition(score, count - 1)[:count]]
        for other in neighbours:
            pair = tuple(sorted((name, names[int(other)])))
            if pair not in existing_pairs:
                recovered.add(pair)
    return sorted(recovered)


def _feature_extraction_batch_size(
    image_count: int,
    *,
    use_cuda: bool,
    worker_threads: int,
) -> int:
    """Bound native extractor queues so long videos cannot exhaust host RAM.

    COLMAP pipelines image reads ahead of feature extraction. Passing hundreds
    of 4K frames to one call can therefore retain many decoded images even
    though SIFT itself processes them incrementally. CUDA startup is cheap
    enough to use moderately sized batches; CPU batches are capped at the
    number of useful parallel extractors.
    """
    if image_count <= 0:
        return 1
    if use_cuda:
        return min(image_count, 16)
    return min(image_count, max(1, min(worker_threads, 8)))


def _match_imported_pairs(
    pycolmap: Any,
    database: Path,
    workspace: Path,
    pairs: Sequence[tuple[str, str]],
    matching: Any,
    verification: Any,
    feature_device: Any,
    project_root: Path,
    feature_backend: str,
    *,
    progress_start: float,
    progress_end: float,
    label: str,
) -> None:
    """Match and geometrically verify an explicit pair graph in bounded batches."""
    if not pairs:
        return
    pair_batch_size = max(512, min(4096, int(matching.num_threads) * 128))
    for batch_index, start in enumerate(range(0, len(pairs), pair_batch_size)):
        batch = pairs[start : start + pair_batch_size]
        pair_path = workspace / f"{label}-pairs-{batch_index:05d}.txt"
        pair_path.write_text(
            "".join(f"{left} {right}\n" for left, right in batch),
            encoding="utf-8",
        )
        pairing = pycolmap.ImportedPairingOptions()
        pairing.block_size = min(pair_batch_size, len(batch))
        pairing.match_list_path = pair_path
        with _progress_heartbeat(
            project_root,
            "feature_matching",
            f"{label}: verifying pairs {start + 1:,}-{start + len(batch):,} of {len(pairs):,}",
            progress_start
            + (progress_end - progress_start) * start / max(len(pairs), 1),
            compute_backend=feature_backend,
            metrics={
                "pairCount": len(pairs),
                "processedPairs": start,
                "pairingMode": label,
            },
        ):
            pycolmap.match_image_pairs(
                database,
                matching_options=matching,
                pairing_options=pairing,
                verification_options=verification,
                device=feature_device,
            )
        pair_path.unlink(missing_ok=True)
        processed = start + len(batch)
        _progress(
            project_root,
            "feature_matching",
            f"{label}: verified {processed:,} of {len(pairs):,} image pairs",
            progress_start
            + (progress_end - progress_start) * processed / max(len(pairs), 1),
            compute_backend=feature_backend,
            metrics={
                "pairCount": len(pairs),
                "processedPairs": processed,
                "pairingMode": label,
            },
        )
        _check_cancelled(project_root)


def _rank_reconstructions(models: dict[int, Any]) -> list[Any]:
    return sorted(
        models.values(),
        key=lambda model: (model.num_reg_images(), model.num_points3D()),
        reverse=True,
    )


def _registered_image_names(reconstruction: Any) -> set[str]:
    return {
        str(image.name)
        for image in reconstruction.images.values()
        if image.has_pose
    }


def _run_sfm(
    images_root: Path,
    workspace: Path,
    records: Sequence[dict[str, Any]],
    options: MediaPreparationOptions,
    project_root: Path,
    sequential: bool,
    camera_proposal: LingbotGeometry | None = None,
) -> tuple[Any, dict[str, Any]]:
    import pycolmap

    image_count = len(records)
    database = workspace / "database.db"
    models_root = workspace / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    feature_runtime = pycolmap_feature_runtime()
    feature_device = pycolmap_device(pycolmap)
    feature_backend = str(feature_runtime["backend"])
    reader, extraction, matching, verification, mapping = _configure_sfm(
        image_count,
        options.maximum_features,
        bool(feature_runtime["cudaValidated"]),
        sequential,
        options.maximum_image_dimension,
        feature_runtime,
    )
    runtime_error = feature_runtime.get("error")
    backend_detail = (
        feature_backend
        if not runtime_error
        else f"{feature_backend} (CUDA unavailable: {runtime_error})"
    )
    _progress(
        project_root,
        "feature_extraction",
        f"Extracting high-detail features from {image_count:,} images with {backend_detail}",
        0.20,
        compute_backend=feature_backend,
        metrics={
            "imageCount": image_count,
            "processedImages": 0,
            "cudaValidated": bool(feature_runtime["cudaValidated"]),
            "cudaError": runtime_error,
        },
    )
    image_names = sorted(path.name for path in images_root.iterdir() if path.is_file())
    proposal_image_names = [str(record["image"]) for record in records]
    if sorted(proposal_image_names) != image_names:
        raise RuntimeError("Learned camera proposal order does not match the prepared image set")
    extraction_groups = _feature_extraction_groups(records)
    grouped_names = sorted(
        image_name
        for group_names, _shared_camera in extraction_groups
        for image_name in group_names
    )
    if grouped_names != image_names:
        raise RuntimeError("Media camera grouping does not match the prepared image set")
    batch_size = _feature_extraction_batch_size(
        len(image_names),
        use_cuda=bool(feature_runtime["cudaValidated"]),
        worker_threads=int(extraction.num_threads),
    )
    processed = 0
    shared_video_camera_ids: list[int] = []
    for group_names, shared_camera in extraction_groups:
        shared_camera_id = -1
        for start in range(0, len(group_names), batch_size):
            _check_cancelled(project_root)
            batch = group_names[start : start + batch_size]
            processed_before = processed
            processed_after = processed + len(batch)
            batch_progress = 0.20 + 0.12 * processed_before / max(image_count, 1)
            reader.existing_camera_id = shared_camera_id if shared_camera else -1
            with _progress_heartbeat(
                project_root,
                "feature_extraction",
                (
                    f"Extracting {feature_backend} features from images "
                    f"{processed_before + 1:,}-{processed_after:,} of {image_count:,}"
                ),
                batch_progress,
                compute_backend=feature_backend,
                metrics={
                    "imageCount": image_count,
                    "processedImages": processed_before,
                    "batchSize": len(batch),
                    "workerThreads": int(extraction.num_threads),
                    "cudaValidated": bool(feature_runtime["cudaValidated"]),
                    "cudaError": runtime_error,
                    "sharedVideoCamera": shared_camera,
                },
            ):
                pycolmap.extract_features(
                    database,
                    images_root,
                    image_names=batch,
                    camera_mode=(
                        pycolmap.CameraMode.SINGLE
                        if shared_camera
                        else pycolmap.CameraMode.AUTO
                    ),
                    reader_options=reader,
                    extraction_options=extraction,
                    device=feature_device,
                )
            if shared_camera and shared_camera_id < 0:
                database_handle = pycolmap.Database.open(database)
                try:
                    first_image = database_handle.read_image_with_name(batch[0])
                    if first_image is None:
                        raise RuntimeError("COLMAP did not persist the shared video camera")
                    shared_camera_id = int(first_image.camera_id)
                    shared_video_camera_ids.append(shared_camera_id)
                finally:
                    database_handle.close()
            processed = processed_after
            _progress(
                project_root,
                "feature_extraction",
                f"Extracted {feature_backend} features from {processed_after:,} of {image_count:,} images",
                0.20 + 0.12 * processed_after / max(image_count, 1),
                compute_backend=feature_backend,
                metrics={
                    "imageCount": image_count,
                    "processedImages": processed_after,
                    "batchSize": len(batch),
                    "workerThreads": int(extraction.num_threads),
                    "cudaValidated": bool(feature_runtime["cudaValidated"]),
                    "cudaError": runtime_error,
                    "sharedVideoCamera": shared_camera,
                },
            )
            _check_cancelled(project_root)
    if shared_video_camera_ids:
        database_handle = pycolmap.Database.open(database)
        try:
            for group_names, shared_camera in extraction_groups:
                if not shared_camera:
                    continue
                camera_ids = {
                    int(image.camera_id)
                    for image_name in group_names
                    for image in [database_handle.read_image_with_name(image_name)]
                    if image is not None
                }
                if len(camera_ids) != 1:
                    raise RuntimeError(
                        "COLMAP assigned multiple intrinsics to one locked-settings video; "
                        "refusing an elastic camera solve"
                    )
        finally:
            database_handle.close()
    _check_cancelled(project_root)
    _progress(
        project_root,
        "feature_matching",
        "Matching overlapping views with geometric verification",
        0.32,
        compute_backend=feature_backend,
        metrics={"imageCount": image_count},
    )
    guided_pairs = _guided_match_pairs(
        proposal_image_names,
        camera_proposal,
        sequential=sequential,
    )
    proposal_used = bool(guided_pairs)
    initial_pairs = guided_pairs or _cpu_match_pairs(
        image_names,
        sequential=sequential and image_count > 120,
    )
    _match_imported_pairs(
        pycolmap,
        database,
        workspace,
        initial_pairs,
        matching,
        verification,
        feature_device,
        project_root,
        feature_backend,
        progress_start=0.32,
        progress_end=0.43,
        label="learned guided matching" if proposal_used else "conventional matching",
    )
    _check_cancelled(project_root)

    all_models: list[Any] = []

    def solve_attempt(output_name: str, progress_value: float, detail: str) -> list[Any]:
        attempt_root = models_root / output_name
        attempt_root.mkdir(parents=True, exist_ok=True)
        camera_telemetry = _CameraSolveTelemetry(image_count)
        with _progress_heartbeat(
            project_root,
            "camera_solving",
            detail,
            progress_value,
            compute_backend="COLMAP incremental mapper / CPU",
            metrics={
                "imageCount": image_count,
                "learnedProposal": proposal_used,
                "guidedPairCount": len(guided_pairs),
            },
            progress_provider=camera_telemetry.progress,
            detail_provider=camera_telemetry.detail,
            metrics_provider=camera_telemetry.snapshot,
        ):
            models = pycolmap.incremental_mapping(
                database,
                images_root,
                attempt_root,
                options=mapping,
                initial_image_pair_callback=camera_telemetry.initial_pair_registered,
                next_image_callback=camera_telemetry.next_image_registered,
            )
        ranked_attempt = _rank_reconstructions(models)
        all_models.extend(ranked_attempt)
        return ranked_attempt

    ranked = solve_attempt(
        "guided" if proposal_used else "conventional",
        0.44,
        (
            "Solving cameras from the learned-guided verified view graph"
            if proposal_used
            else "Solving cameras and globally refining structure"
        ),
    )
    best = ranked[0] if ranked else None
    initial_registered = best.num_reg_images() if best is not None else 0
    matched_pairs = {tuple(sorted(pair)) for pair in initial_pairs}
    recovery_pairs: list[tuple[str, str]] = []
    if proposal_used and best is not None and best.num_reg_images() < image_count:
        recovery_pairs = _camera_recovery_pairs(
            proposal_image_names,
            camera_proposal,
            _registered_image_names(best),
            matched_pairs,
        )
        if recovery_pairs:
            _match_imported_pairs(
                pycolmap,
                database,
                workspace,
                recovery_pairs,
                matching,
                verification,
                feature_device,
                project_root,
                feature_backend,
                progress_start=0.52,
                progress_end=0.57,
                label="camera recovery",
            )
            ranked = solve_attempt(
                "recovery",
                0.57,
                "Recovering missing cameras through learned-neighbour geometric verification",
            )
            if ranked and (
                best is None
                or (ranked[0].num_reg_images(), ranked[0].num_points3D())
                > (best.num_reg_images(), best.num_points3D())
            ):
                best = ranked[0]

    minimum_registered = _minimum_useful_registration_count(image_count)
    preferred_registered = max(minimum_registered, math.ceil(image_count * 0.85))
    needs_fallback = best is None or best.num_reg_images() < preferred_registered
    fallback_used = False
    fallback_pairs: list[tuple[str, str]] = []
    if needs_fallback and proposal_used:
        conventional_pairs = _cpu_match_pairs(
            image_names,
            sequential=sequential and image_count > 120,
        )
        already_matched = matched_pairs | {
            tuple(sorted(pair)) for pair in recovery_pairs
        }
        fallback_pairs = [
            pair
            for pair in conventional_pairs
            if tuple(sorted(pair)) not in already_matched
        ]
        if fallback_pairs:
            fallback_used = True
            _match_imported_pairs(
                pycolmap,
                database,
                workspace,
                fallback_pairs,
                matching,
                verification,
                feature_device,
                project_root,
                feature_backend,
                progress_start=0.58,
                progress_end=0.63,
                label="quality fallback",
            )
            ranked = solve_attempt(
                "fallback",
                0.63,
                "Retrying the camera solve with the conventional verified view graph",
            )
            if ranked and (
                best is None
                or (ranked[0].num_reg_images(), ranked[0].num_points3D())
                > (best.num_reg_images(), best.num_points3D())
            ):
                best = ranked[0]

    if best is None:
        raise RuntimeError(
            "Camera solving found no consistent reconstruction. Capture more overlap, texture, and parallax."
        )
    reconstruction = best
    registered = reconstruction.num_reg_images()
    minimum_registered = _minimum_useful_registration_count(image_count)
    if registered < minimum_registered:
        raise RuntimeError(
            f"Camera solving registered only {registered} of {image_count} images "
            f"(minimum {minimum_registered}); capture a slower path with at least "
            "60-80% overlap and avoid motion blur."
        )
    _progress(
        project_root,
        "camera_refinement",
        "Running final high-resolution global bundle adjustment",
        0.67,
        compute_backend="COLMAP Ceres bundle adjustment / CPU",
        metrics={
            "registeredCameras": registered,
            "featureImageLimit": options.maximum_image_dimension,
        },
    )
    bundle_options = pycolmap.BundleAdjustmentOptions()
    bundle_options.refine_focal_length = True
    bundle_options.refine_principal_point = False
    bundle_options.refine_extra_params = True
    bundle_options.ceres.loss_function_type = pycolmap.LossFunctionType.HUBER
    bundle_options.ceres.loss_function_scale = 1.0
    bundle_options.ceres.solver_options.max_num_iterations = 100
    bundle_options.ceres.solver_options.function_tolerance = 1e-7
    bundle_options.ceres.solver_options.num_threads = int(mapping.num_threads)
    with _progress_heartbeat(
        project_root,
        "camera_refinement",
        "Final high-resolution global bundle adjustment",
        0.67,
        compute_backend="COLMAP Ceres bundle adjustment / CPU",
        metrics={"registeredCameras": registered},
    ):
        pycolmap.bundle_adjustment(reconstruction, bundle_options)
    reconstruction.update_point_3d_errors()
    database_handle = pycolmap.Database.open(database)
    try:
        verified_pair_count = int(database_handle.num_verified_image_pairs())
        inlier_match_count = int(database_handle.num_inlier_matches())
    finally:
        database_handle.close()
    statistics = {
        "inputImageCount": image_count,
        "registeredImageCount": registered,
        "registrationRatio": registered / image_count,
        "sparsePointCount": reconstruction.num_points3D(),
        "modelCount": len(all_models),
        "excludedImageCount": image_count - registered,
        "featureBackend": feature_backend,
        "matching": (
            "learned guided + geometric verification"
            if proposal_used
            else "sequential bounded" if sequential and image_count > 120 else "exhaustive"
        ),
        "cameraSolveOrder": "learned_first" if proposal_used else "conventional_fallback",
        "learnedProposalBackend": camera_proposal.backend if proposal_used else None,
        "guidedPairCount": len(guided_pairs),
        "geometricallyVerifiedPairCount": verified_pair_count,
        "geometricInlierMatchCount": inlier_match_count,
        "cameraRecoveryPairCount": len(recovery_pairs),
        "conventionalFallbackPairCount": len(fallback_pairs),
        "conventionalFallbackUsed": fallback_used,
        "preferredRegistrationCount": preferred_registered,
        "initialRegisteredImageCount": initial_registered,
        "recoveredImageCount": max(0, registered - initial_registered),
        "finalBundleAdjustment": True,
        "bundleAdjustmentFeatureImageLimit": options.maximum_image_dimension,
        "sharedVideoCameraCount": len(shared_video_camera_ids),
    }
    return reconstruction, statistics


def _camera_intrinsics(camera: Any) -> dict[str, Any]:
    calibration = camera.calibration_matrix()
    return {
        "width": int(camera.width),
        "height": int(camera.height),
        "fx": float(calibration[0, 0]),
        "fy": float(calibration[1, 1]),
        "cx": float(calibration[0, 2]),
        "cy": float(calibration[1, 2]),
        "model": "pinhole",
        "distortion": [],
    }


def _shared_video_camera(reconstruction: Any) -> tuple[Any, dict[str, Any]]:
    registered_by_name = {
        image.name: image
        for image in reconstruction.images.values()
        if image.has_pose
    }
    if not registered_by_name:
        raise RuntimeError("COLMAP supplied no registered camera for the video")
    camera_ids = {int(image.camera_id) for image in registered_by_name.values()}
    if len(camera_ids) != 1:
        raise RuntimeError("A locked-settings video unexpectedly uses multiple COLMAP cameras")
    return reconstruction.cameras[next(iter(camera_ids))], registered_by_name


def _lingbot_calibrated_rays(camera: Any, records: Sequence[dict[str, Any]]) -> np.ndarray:
    sizes = {
        (int(record["width"]), int(record["height"]))
        for record in records
    }
    if len(sizes) != 1:
        raise RuntimeError("A locked-settings video unexpectedly contains mixed frame sizes")
    width, height = next(iter(sizes))
    if int(camera.width) != width or int(camera.height) != height:
        raise RuntimeError(
            "COLMAP's calibrated video dimensions do not match the selected frames"
        )
    source_pixels = lingbot_source_pixel_grid(width, height)
    normalized = camera.cam_from_img(source_pixels.reshape(-1, 2))
    if normalized is None:
        raise RuntimeError("COLMAP could not unproject the calibrated video pixels")
    rays = np.asarray(normalized, dtype=np.float64).reshape(*source_pixels.shape)
    if not np.isfinite(rays).all():
        raise RuntimeError("COLMAP produced non-finite calibrated camera rays")
    return rays


def _world_from_camera(image: Any) -> list[float]:
    camera_from_world = np.eye(4, dtype=np.float64)
    camera_from_world[:3, :] = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
    return np.linalg.inv(camera_from_world).reshape(-1).tolist()


def _point_records(reconstruction: Any) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    errors: list[float] = []
    track_lengths: list[int] = []
    for point in reconstruction.points3D.values():
        error = float(point.error)
        track_length = int(point.track.length())
        if not np.all(np.isfinite(point.xyz)) or not math.isfinite(error):
            continue
        if error > 4.0 or track_length < 2:
            continue
        points.append(np.asarray(point.xyz, dtype=np.float32))
        colors.append(np.asarray(point.color, dtype=np.uint8))
        errors.append(error)
        track_lengths.append(track_length)
    if len(points) < 100:
        raise RuntimeError(
            f"Camera solving produced only {len(points)} reliable sparse points; the capture lacks usable parallax or texture."
        )
    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(colors, dtype=np.uint8),
        {
            "meanReprojectionErrorPx": float(np.mean(errors)),
            "medianReprojectionErrorPx": float(np.median(errors)),
            "meanTrackLength": float(np.mean(track_lengths)),
        },
    )


def _write_initialization(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
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
    vertices = np.empty(len(points), dtype=vertex_type)
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment COLMAP sparse initialization\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def _write_initialization_parameters(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    scales: np.ndarray,
    quaternions: np.ndarray,
    confidence: np.ndarray | None = None,
    opacity: np.ndarray | None = None,
    fusion_confidence: np.ndarray | None = None,
    source_frame_indices: np.ndarray | None = None,
    provenance: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = dict(
        points=np.asarray(points, dtype=np.float32),
        colors=np.asarray(colors, dtype=np.uint8),
        scales=np.asarray(scales, dtype=np.float32),
        quaternions=np.asarray(quaternions, dtype=np.float32),
    )
    if confidence is not None:
        values["confidence"] = np.asarray(confidence, dtype=np.float32)
    if opacity is not None:
        values["opacity"] = np.asarray(opacity, dtype=np.float32)
    if fusion_confidence is not None:
        values["fusion_confidence"] = np.asarray(fusion_confidence, dtype=np.float32)
    if source_frame_indices is not None:
        values["source_frame_indices"] = np.asarray(source_frame_indices, dtype=np.int32)
    if provenance is not None:
        values["provenance"] = np.asarray(provenance, dtype=np.uint8)
    np.savez_compressed(path, **values)


def _geometry_fusion_confidence(geometry: LingbotGeometry) -> np.ndarray:
    owners = np.asarray(geometry.source_frame_indices, dtype=np.int64)
    frame_confidence = np.asarray(geometry.frame_confidence, dtype=np.float32)
    if owners.shape != (len(geometry.points),):
        raise ValueError("Learned dense geometry ownership does not match its points")
    if np.any((owners < 0) | (owners >= len(frame_confidence))):
        raise ValueError("Learned dense geometry contains an invalid source-frame owner")
    values = np.clip(frame_confidence[owners], 0.0, 1.0)
    if geometry.opacities is not None:
        # Direct Gaussian opacity is visibility, not geometric confidence.
        # Validate its shape here, but keep it separate so a surface seed does
        # not disappear from point/mesh fusion merely because the renderer
        # expects optimization to raise its initial opacity.
        opacity = np.asarray(geometry.opacities, dtype=np.float32).reshape(-1)
        if opacity.shape != values.shape:
            raise ValueError("Learned opacity does not match dense geometry points")
    return values.astype(np.float32, copy=False)


def _average_rotation(rotations: np.ndarray) -> np.ndarray:
    total = np.sum(np.asarray(rotations, dtype=np.float64), axis=0)
    left, _values, right = np.linalg.svd(total)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def _rotation_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        value = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(max(1e-12, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            value = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = math.sqrt(max(1e-12, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
            value = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(max(1e-12, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
            value = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return value / max(float(np.linalg.norm(value)), 1e-12)


def _multiply_quaternions_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    values = np.asarray(right, dtype=np.float64)
    rw, rx, ry, rz = values.T
    result = np.column_stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
    return result.astype(np.float32)


def _align_lingbot_geometry(
    geometry: LingbotGeometry,
    reconstruction: Any,
    image_names: Sequence[str],
) -> tuple[LingbotGeometry, dict[str, Any]]:
    """Estimate a positive-scale LingBot-to-COLMAP similarity from camera centers."""
    references: list[tuple[int, np.ndarray]] = []
    for frame_index, image_name in enumerate(image_names):
        image = reconstruction.find_image_with_name(image_name)
        if image is None or not image.has_pose:
            continue
        references.append(
            (
                frame_index,
                np.asarray(_world_from_camera(image), dtype=np.float64).reshape(4, 4),
            )
        )
    if len(references) < 3:
        raise RuntimeError(
            "LingBot-Map geometry could not be aligned because COLMAP validated fewer than three shared cameras"
        )
    indices = np.asarray([index for index, _pose in references], dtype=np.int64)
    source_poses = geometry.world_from_cameras[indices]
    target_poses = np.asarray([pose for _index, pose in references], dtype=np.float64)
    inliers = np.ones(len(indices), dtype=bool)
    rotation = np.eye(3, dtype=np.float64)
    scale = 1.0
    translation = np.zeros(3, dtype=np.float64)
    residuals = np.zeros(len(indices), dtype=np.float64)
    for _iteration in range(5):
        source_centers = source_poses[inliers, :3, 3]
        target_centers = target_poses[inliers, :3, 3]
        source_mean = np.mean(source_centers, axis=0)
        target_mean = np.mean(target_centers, axis=0)
        source_centered = source_centers - source_mean
        target_centered = target_centers - target_mean
        left, _values, right = np.linalg.svd(source_centered.T @ target_centered)
        rotation = right.T @ left.T
        if np.linalg.det(rotation) < 0.0:
            right[-1] *= -1.0
            rotation = right.T @ left.T
        rotated = source_centered @ rotation.T
        denominator = float(np.sum(source_centered * source_centered))
        scale = float(np.sum(rotated * target_centered) / max(denominator, 1e-12))
        if not math.isfinite(scale) or scale <= 1e-8:
            raise RuntimeError("LingBot-Map and COLMAP camera scales could not be reconciled")
        translation = target_mean - scale * (rotation @ source_mean)
        predicted = scale * (source_poses[:, :3, 3] @ rotation.T) + translation
        residuals = np.linalg.norm(predicted - target_poses[:, :3, 3], axis=1)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        next_inliers = residuals <= median + max(3.5 * mad, median * 0.35, 1e-5)
        if int(np.sum(next_inliers)) < max(3, len(indices) // 2):
            break
        if np.array_equal(next_inliers, inliers):
            break
        inliers = next_inliers

    poses = np.asarray(geometry.world_from_cameras, dtype=np.float64).copy()
    poses[:, :3, :3] = rotation[None] @ poses[:, :3, :3]
    poses[:, :3, 3] = scale * (poses[:, :3, 3] @ rotation.T) + translation
    points = scale * (geometry.points.astype(np.float64) @ rotation.T) + translation
    rotation_quaternion = _rotation_quaternion_wxyz(rotation)
    quaternions = _multiply_quaternions_wxyz(
        rotation_quaternion,
        geometry.quaternions,
    )
    aligned = LingbotGeometry(
        world_from_cameras=poses,
        intrinsics=geometry.intrinsics,
        points=points.astype(np.float32),
        colors=geometry.colors,
        scales=(geometry.scales * scale).astype(np.float32),
        quaternions=quaternions,
        source_frame_indices=geometry.source_frame_indices,
        frame_confidence=geometry.frame_confidence,
        backend=geometry.backend,
        model_path=geometry.model_path,
        processed_size=geometry.processed_size,
        opacities=geometry.opacities,
    )
    rotation_residuals = []
    for source, target in zip(poses[indices, :3, :3], target_poses[:, :3, :3], strict=True):
        delta = target @ source.T
        angle = math.degrees(math.acos(float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))))
        rotation_residuals.append(angle)
    target_centers = target_poses[:, :3, 3]
    target_scale = max(
        float(
            np.percentile(
                np.linalg.norm(
                    target_centers - np.median(target_centers, axis=0, keepdims=True),
                    axis=1,
                ),
                90.0,
            )
        ),
        1e-6,
    )
    return aligned, {
        "alignmentCameraCount": len(indices),
        "alignmentInlierCount": int(np.sum(inliers)),
        "alignmentScale": scale,
        "medianCameraCenterResidual": float(np.median(residuals[inliers])),
        "maximumCameraCenterResidual": float(np.max(residuals[inliers])),
        "medianAllCameraCenterResidual": float(np.median(residuals)),
        "normalizedMedianCameraCenterResidual": float(np.median(residuals) / target_scale),
        "medianCameraRotationResidualDegrees": float(np.median(rotation_residuals)),
    }


def _accept_lingbot_alignment(quality: dict[str, Any]) -> bool:
    camera_count = int(quality.get("alignmentCameraCount", 0))
    inlier_count = int(quality.get("alignmentInlierCount", 0))
    return bool(
        camera_count >= 6
        and inlier_count >= max(6, math.ceil(camera_count * 0.6))
        and float(quality.get("normalizedMedianCameraCenterResidual", math.inf)) <= 0.08
        and float(quality.get("medianCameraRotationResidualDegrees", math.inf)) <= 12.0
    )


def _anchor_lingbot_geometry(
    geometry: LingbotGeometry,
    reconstruction: Any,
    image_names: Sequence[str],
) -> tuple[LingbotGeometry, dict[str, Any]]:
    """Warp LingBot locally so every validated COLMAP camera remains exact."""
    from scipy.spatial.transform import Rotation, Slerp

    references: list[tuple[int, np.ndarray]] = []
    for frame_index, image_name in enumerate(image_names):
        image = reconstruction.find_image_with_name(image_name)
        if image is not None and image.has_pose:
            references.append(
                (
                    frame_index,
                    np.asarray(_world_from_camera(image), dtype=np.float64).reshape(4, 4),
                )
            )
    if len(references) < 2:
        raise RuntimeError("Hybrid LingBot anchoring requires at least two COLMAP cameras")
    anchor_indices = np.asarray([index for index, _pose in references], dtype=np.int64)
    source_poses = np.asarray(geometry.world_from_cameras, dtype=np.float64)
    target_poses = np.asarray([pose for _index, pose in references], dtype=np.float64)
    source_centers = source_poses[anchor_indices, :3, 3]
    target_centers = target_poses[:, :3, 3]
    correction_rotations = (
        target_poses[:, :3, :3]
        @ np.transpose(source_poses[anchor_indices, :3, :3], (0, 2, 1))
    )
    pair_source = np.linalg.norm(np.diff(source_centers, axis=0), axis=1)
    pair_target = np.linalg.norm(np.diff(target_centers, axis=0), axis=1)
    valid_pairs = (pair_source > 1e-6) & (pair_target > 1e-6)
    pair_scales = pair_target[valid_pairs] / pair_source[valid_pairs]
    global_scale = float(np.median(pair_scales)) if len(pair_scales) else 1.0
    # One global scale keeps dense depths from different anchors in the same
    # world metric. Per-anchor ratios become unstable when two selected frames
    # have almost no translation and previously produced giant splats.
    anchor_scales = np.full(len(anchor_indices), global_scale, dtype=np.float64)
    anchor_translations = target_centers - anchor_scales[:, None] * np.einsum(
        "nij,nj->ni",
        correction_rotations,
        source_centers,
    )

    frame_count = len(source_poses)
    frame_rotations = np.empty((frame_count, 3, 3), dtype=np.float64)
    frame_scales = np.empty(frame_count, dtype=np.float64)
    frame_translations = np.empty((frame_count, 3), dtype=np.float64)
    for frame_index in range(frame_count):
        upper = int(np.searchsorted(anchor_indices, frame_index, side="left"))
        if upper <= 0:
            lower = upper = 0
            alpha = 0.0
        elif upper >= len(anchor_indices):
            lower = upper = len(anchor_indices) - 1
            alpha = 0.0
        else:
            lower = upper - 1
            span = max(1, int(anchor_indices[upper] - anchor_indices[lower]))
            alpha = float((frame_index - anchor_indices[lower]) / span)
        if lower == upper:
            frame_rotations[frame_index] = correction_rotations[lower]
            frame_scales[frame_index] = anchor_scales[lower]
            frame_translations[frame_index] = anchor_translations[lower]
        else:
            interpolator = Slerp(
                [0.0, 1.0],
                Rotation.from_matrix(correction_rotations[[lower, upper]]),
            )
            frame_rotations[frame_index] = interpolator([alpha]).as_matrix()[0]
            frame_scales[frame_index] = math.exp(
                (1.0 - alpha) * math.log(anchor_scales[lower])
                + alpha * math.log(anchor_scales[upper])
            )
            frame_translations[frame_index] = (
                (1.0 - alpha) * anchor_translations[lower]
                + alpha * anchor_translations[upper]
            )

    poses = source_poses.copy()
    poses[:, :3, :3] = frame_rotations @ poses[:, :3, :3]
    poses[:, :3, 3] = frame_scales[:, None] * np.einsum(
        "nij,nj->ni",
        frame_rotations,
        poses[:, :3, 3],
    ) + frame_translations
    # Remove interpolation round-off at validated anchors.
    poses[anchor_indices] = target_poses
    source_indices = np.asarray(geometry.source_frame_indices, dtype=np.int64)
    if (
        source_indices.shape != (len(geometry.points),)
        or np.any(source_indices < 0)
        or np.any(source_indices >= frame_count)
    ):
        raise RuntimeError("LingBot dense seeds lost their source-frame ownership")
    point_rotations = frame_rotations[source_indices]
    point_scales = frame_scales[source_indices]
    points = point_scales[:, None] * np.einsum(
        "nij,nj->ni",
        point_rotations,
        np.asarray(geometry.points, dtype=np.float64),
    ) + frame_translations[source_indices]
    correction_quaternions = np.asarray(
        [_rotation_quaternion_wxyz(rotation) for rotation in frame_rotations],
        dtype=np.float64,
    )
    quaternions = np.empty_like(geometry.quaternions, dtype=np.float32)
    for frame_index in np.unique(source_indices):
        mask = source_indices == frame_index
        quaternions[mask] = _multiply_quaternions_wxyz(
            correction_quaternions[frame_index],
            geometry.quaternions[mask],
        )
    anchored = LingbotGeometry(
        world_from_cameras=poses,
        intrinsics=geometry.intrinsics,
        points=points.astype(np.float32),
        colors=geometry.colors,
        scales=(geometry.scales * point_scales[:, None]).astype(np.float32),
        quaternions=quaternions,
        source_frame_indices=geometry.source_frame_indices,
        frame_confidence=geometry.frame_confidence,
        backend=geometry.backend,
        model_path=geometry.model_path,
        processed_size=geometry.processed_size,
        opacities=geometry.opacities,
    )
    center_residuals = np.linalg.norm(
        poses[anchor_indices, :3, 3] - target_centers,
        axis=1,
    )
    rotation_residuals = []
    for source, target in zip(
        poses[anchor_indices, :3, :3],
        target_poses[:, :3, :3],
        strict=True,
    ):
        delta = target @ source.T
        rotation_residuals.append(
            math.degrees(
                math.acos(float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)))
            )
        )
    return anchored, {
        "anchorCameraCount": len(anchor_indices),
        "recoveredCameraCount": frame_count - len(anchor_indices),
        "medianLocalScale": float(np.median(frame_scales)),
        "minimumLocalScale": float(np.min(frame_scales)),
        "maximumLocalScale": float(np.max(frame_scales)),
        "maximumAnchorCenterResidual": float(np.max(center_residuals)),
        "maximumAnchorRotationResidualDegrees": float(np.max(rotation_residuals)),
    }


def _restrict_lingbot_seeds_to_frames(
    geometry: LingbotGeometry,
    frame_indices: Sequence[int],
) -> LingbotGeometry:
    allowed = np.asarray(sorted(set(int(index) for index in frame_indices)), dtype=np.int64)
    if not len(allowed):
        raise RuntimeError("No validated cameras are available for LingBot depth seeding")
    keep = np.isin(geometry.source_frame_indices, allowed)
    if not np.any(keep):
        raise RuntimeError("LingBot produced no dense seeds owned by validated cameras")
    return LingbotGeometry(
        world_from_cameras=geometry.world_from_cameras,
        intrinsics=geometry.intrinsics,
        points=geometry.points[keep],
        colors=geometry.colors[keep],
        scales=geometry.scales[keep],
        quaternions=geometry.quaternions[keep],
        source_frame_indices=geometry.source_frame_indices[keep],
        frame_confidence=geometry.frame_confidence,
        backend=geometry.backend,
        model_path=geometry.model_path,
        processed_size=geometry.processed_size,
        opacities=(geometry.opacities[keep] if geometry.opacities is not None else None),
    )


def _undistort_complete_video(
    pycolmap: Any,
    input_images: Path,
    output_images: Path,
    records: Sequence[dict[str, Any]],
    reconstruction: Any,
    geometry: LingbotGeometry,
    *,
    use_colmap_poses: bool,
    learned_pose_source: str,
) -> tuple[list[dict[str, Any]], int]:
    """Undistort every selected video frame and keep validated COLMAP poses."""
    distorted_camera, registered_by_name = _shared_video_camera(reconstruction)
    options = pycolmap.UndistortCameraOptions()
    options.max_image_size = -1
    output_images.mkdir(parents=True, exist_ok=True)
    source_by_name = {str(record["image"]): record for record in records}
    frames: list[dict[str, Any]] = []
    recovered = 0
    canonical_intrinsics: dict[str, Any] | None = None
    for frame_index, record in enumerate(records):
        image_name = str(record["image"])
        # Resample in linear light, then let Bitmap.write convert back to sRGB.
        # This avoids both gamma-darkened interpolation and accidental double
        # delinearization in PyCOLMAP's default write path.
        bitmap = pycolmap.Bitmap.read(input_images / image_name, True, True)
        if bitmap is None or bitmap.is_empty:
            raise RuntimeError(f"Could not read video frame for undistortion: {image_name}")
        undistorted_bitmap, output_camera = pycolmap.undistort_image(
            options,
            bitmap,
            distorted_camera,
        )
        undistorted_bitmap.set_jpeg_quality(100)
        if not undistorted_bitmap.write(output_images / image_name):
            raise RuntimeError(f"Could not write undistorted video frame: {image_name}")
        intrinsics = _camera_intrinsics(output_camera)
        if canonical_intrinsics is None:
            canonical_intrinsics = intrinsics
        else:
            normalized_delta = max(
                abs(float(intrinsics[key]) - float(canonical_intrinsics[key]))
                / max(abs(float(canonical_intrinsics[key])), 1e-9)
                for key in ("fx", "fy", "cx", "cy")
            )
            if normalized_delta > 1e-6:
                raise RuntimeError("Video undistortion produced inconsistent shared intrinsics")
        validated = registered_by_name.get(image_name)
        if validated is not None and use_colmap_poses:
            world_from_camera = np.asarray(_world_from_camera(validated), dtype=np.float64)
            pose_confidence = 1.0
            pose_source = "colmap_bundle_adjusted"
        else:
            world_from_camera = geometry.world_from_cameras[frame_index].reshape(-1)
            pose_confidence = float(geometry.frame_confidence[frame_index])
            pose_source = learned_pose_source
            if validated is None:
                recovered += 1
        source = source_by_name[image_name]
        frames.append(
            {
                "phaseId": "media",
                "frameIndex": frame_index,
                "sourceFrameIndex": frame_index,
                "timestampUs": (
                    round(float(source["timestampSeconds"]) * 1_000_000)
                    if source.get("timestampSeconds") is not None
                    else None
                ),
                "intrinsics": intrinsics,
                "worldFromRgbCamera": world_from_camera.tolist(),
                "image": f"images/{image_name}",
                "sourcePath": source.get("source"),
                "sharpness": source.get("sharpness"),
                "poseConfidence": pose_confidence,
                "poseSource": pose_source,
                "poseAnchor": (
                    (validated is not None if use_colmap_poses else frame_index == 0)
                    and not any(frame.get("poseAnchor", False) for frame in frames)
                ),
            }
        )
    return frames, recovered


def _publish_pointer(project_root: Path, dataset_root: Path) -> Path:
    datasets = project_root / "outputs" / "cache" / "datasets"
    relative = os.path.relpath(dataset_root, datasets).replace("\\", "/")
    pointer = datasets / "current.json"
    payload = {"schemaVersion": 1, "path": relative}
    _write_json_atomic(pointer, payload)
    # Hybrid reconstruction subsequently publishes its metric canonical
    # dataset to current.json. Keep the independently solved media geometry
    # addressable so dense fusion can align it through localized cameras.
    _write_json_atomic(datasets / "media-current.json", payload)
    return pointer


def prepare_media_observations(
    project_root: Path,
    sources: Sequence[Path] = (),
    options: MediaPreparationOptions | None = None,
) -> dict[str, Any]:
    """Decode immutable media views for metric RGB-D localization.

    Unlike ``prepare_media_dataset``, this path deliberately stops before
    structure-from-motion: an RGB-D reconstruction already supplies metric 3D
    landmarks and camera anchors. The reconstruction worker localizes these
    observations against that map in the next pipeline stage.
    """
    project_root = project_root.resolve()
    options = options or MediaPreparationOptions()
    sources = _media_sources(project_root, sources)
    fingerprint = _source_fingerprint(sources, options)
    observations_root = project_root / "outputs" / "cache" / "media-observations"
    observations_root.mkdir(parents=True, exist_ok=True)
    published_root = observations_root / f"media-{fingerprint[:20]}"
    manifest_path = published_root / "observations.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _write_json_atomic(
            observations_root / "current.json",
            {"schemaVersion": 1, "path": published_root.name},
        )
        _progress(
            project_root,
            "media_ready",
            f"Reusing {len(manifest.get('frames', [])):,} decoded media observations",
            1.0,
            compute_backend="Cached media observations",
            metrics={"selectedFrames": len(manifest.get("frames", []))},
        )
        return manifest

    staging = observations_root / f".media-observations-{uuid.uuid4().hex}"
    images_root = staging / "images"
    try:
        images_root.mkdir(parents=True, exist_ok=False)
        records, videos = _collect_images(
            sources,
            images_root,
            options,
            project_root,
        )
        frames = [
            {
                "image": f"images/{record['image']}",
                "sourcePath": record["source"],
                "sourceType": (
                    "video"
                    if Path(str(record["source"])).suffix.lower() in VIDEO_EXTENSIONS
                    else "photo"
                ),
                "timestampSeconds": record.get("timestampSeconds"),
                "width": int(record["width"]),
                "height": int(record["height"]),
                "sharpness": float(record["sharpness"]),
            }
            for record in records
        ]
        manifest = {
            "schemaVersion": 1,
            "fingerprint": fingerprint,
            "frames": frames,
            "videoSources": videos,
        }
        _write_json_atomic(staging / "observations.json", manifest)
        os.replace(staging, published_root)
        _write_json_atomic(
            observations_root / "current.json",
            {"schemaVersion": 1, "path": published_root.name},
        )
        _progress(
            project_root,
            "media_ready",
            f"Prepared {len(frames):,} high-resolution observations for metric localization",
            1.0,
            compute_backend="PyAV / Pillow media preparation",
            metrics={"selectedFrames": len(frames)},
        )
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _materialize_observation_inputs(
    observations_root: Path,
    observations: dict[str, Any],
    input_images: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Link immutable decoded views into a disposable camera-analysis workspace."""
    records: list[dict[str, Any]] = []
    for frame in observations.get("frames", []):
        relative = Path(str(frame.get("image", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Media observation manifest contains an unsafe image path")
        source_image = (observations_root / relative).resolve(strict=True)
        if not source_image.is_relative_to(observations_root.resolve()):
            raise ValueError("Media observation image escapes its immutable cache")
        image_name = source_image.name
        destination = input_images / image_name
        try:
            os.link(source_image, destination)
        except OSError:
            shutil.copy2(source_image, destination)
        records.append(
            {
                "source": str(frame.get("sourcePath", "")),
                "image": image_name,
                "width": int(frame["width"]),
                "height": int(frame["height"]),
                "orientation": 1,
                "sharpness": float(frame.get("sharpness", 0.0)),
                "timestampSeconds": frame.get("timestampSeconds"),
            }
        )
    if len(records) < 3:
        raise ValueError("At least three usable media observations are required")
    return records, list(observations.get("videoSources", []))


def prepare_media_dataset(
    project_root: Path,
    sources: Sequence[Path] = (),
    options: MediaPreparationOptions | None = None,
    *,
    geometry_worker: Path | None = None,
    progressive_rgb_preview: bool = False,
) -> dict[str, Any]:
    """Create an immutable, undistorted COLMAP dataset for Gaussian training."""
    import pycolmap

    project_root = project_root.resolve()
    options = options or MediaPreparationOptions()
    sources = _media_sources(project_root, sources)
    observation_fingerprint = _source_fingerprint(sources, options)
    fingerprint = _media_dataset_fingerprint(observation_fingerprint, options)
    datasets_root = project_root / "outputs" / "cache" / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)
    for stale_build in datasets_root.glob(".media-build-*"):
        if stale_build.is_dir():
            shutil.rmtree(stale_build, ignore_errors=True)
    published_root = datasets_root / f"media-{fingerprint[:20]}"
    if (published_root / "dataset.json").is_file():
        _publish_pointer(project_root, published_root)
        dataset = json.loads((published_root / "dataset.json").read_text(encoding="utf-8"))
        _progress(project_root, "media_ready", "Reusing the validated photo/video camera solution", 1.0)
        return dataset

    staging = datasets_root / f".media-build-{uuid.uuid4().hex}"
    input_images = staging / "input-images"
    sfm_workspace = staging / "sfm"
    undistorted = staging / "undistorted"
    started = time.perf_counter()
    try:
        input_images.mkdir(parents=True, exist_ok=False)
        observations = prepare_media_observations(
            project_root,
            sources,
            options,
        )
        observations_root = (
            project_root
            / "outputs"
            / "cache"
            / "media-observations"
            / f"media-{observation_fingerprint[:20]}"
        )
        records, video_statistics = _materialize_observation_inputs(
            observations_root,
            observations,
            input_images,
        )
        single_video = (
            len(video_statistics) == 1
            and not any(path.suffix.lower() in PHOTO_EXTENSIONS for path in sources)
        )
        lingbot_geometry: LingbotGeometry | None = None
        lingbot_context: dict[str, Any] | None = None
        mapanything_proposal: LingbotGeometry | None = None
        mapanything_geometry: LingbotGeometry | None = None
        mapanything_quality: dict[str, Any] = {}
        mapanything_error: str | None = None
        mapanything_evaluated = False
        da3_proposal: LingbotGeometry | None = None
        da3_geometry: LingbotGeometry | None = None
        da3_quality: dict[str, Any] = {}
        da3_error: str | None = None
        da3_evaluated = False
        da3_streaming: dict[str, Any] | None = None
        da3_direct_gaussians = False
        da3_direct_gaussians_used = False
        # P9 reverses the old authority order: a bounded learned model proposes
        # cameras first, then high-resolution local features decide which of
        # those relationships survive geometric verification and bundle
        # adjustment.  A failed proposal remains a transparent COLMAP fallback.
        if geometry_worker is not None and 3 <= len(records) <= 32:
            mapanything_evaluated = True

            def report_mapanything(
                stage: str,
                detail: str,
                progress_value: float,
                backend: str,
                metrics: dict[str, Any],
            ) -> None:
                _progress(
                    project_root,
                    stage,
                    detail,
                    0.09 + 0.03 * min(max(progress_value, 0.0), 1.0),
                    compute_backend=backend,
                    metrics={**metrics, "cameraProposal": True},
                )

            try:
                mapanything_proposal = infer_mapanything_geometry_isolated(
                    geometry_worker,
                    [input_images / str(record["image"]) for record in records],
                    work_root=staging / "geometry-ipc",
                    cancel_path=project_root / "outputs" / "cancel.flag",
                    progress=report_mapanything,
                )
            except Exception as error:
                mapanything_error = str(error)
            _check_cancelled(project_root)
        if geometry_worker is not None and len(records) >= 3:
            da3_evaluated = True
            da3_direct_gaussians = True

            def report_da3(
                stage: str,
                detail: str,
                progress_value: float,
                backend: str,
                metrics: dict[str, Any],
            ) -> None:
                _progress(
                    project_root,
                    stage,
                    detail,
                    0.12 + 0.07 * min(max(progress_value, 0.0), 1.0),
                    compute_backend=backend,
                    metrics={
                        **metrics,
                        "cameraProposal": True,
                        "noncommercial": True,
                    },
                )

            try:
                da3_proposal, da3_streaming = infer_da3_geometry_isolated(
                    geometry_worker,
                    [input_images / str(record["image"]) for record in records],
                    work_root=staging / "geometry-ipc",
                    cancel_path=project_root / "outputs" / "cancel.flag",
                    direct_gaussians=da3_direct_gaussians,
                    progress=report_da3,
                )
                da3_direct_gaussians_used = bool(
                    da3_streaming.get("directGaussiansUsed", False)
                )
            except Exception as error:
                da3_error = str(error)
            _check_cancelled(project_root)
        camera_proposal = (
            da3_proposal
            if _learned_camera_arrays(da3_proposal, len(records)) is not None
            else mapanything_proposal
            if _learned_camera_arrays(mapanything_proposal, len(records)) is not None
            else None
        )
        reconstruction, solve = _run_sfm(
            input_images,
            sfm_workspace,
            records,
            options,
            project_root,
            sequential=bool(video_statistics),
            camera_proposal=camera_proposal,
        )
        _check_cancelled(project_root)
        if single_video:
            if geometry_worker is None:
                raise RuntimeError(
                    "Video reconstruction requires the isolated ScanLan geometry worker"
                )
            calibrated_rays = _lingbot_calibrated_rays(
                _shared_video_camera(reconstruction)[0],
                records,
            )
            context_paths, lingbot_output_indices, lingbot_context = _extract_lingbot_context(
                sources[0],
                records,
                input_images,
                staging / "lingbot-context",
                project_root,
            )

            def report_lingbot(
                stage: str,
                detail: str,
                progress_value: float,
                backend: str,
                metrics: dict[str, Any],
            ) -> None:
                # LingBot's standalone callback uses 0.09-0.20. Camera solving
                # already occupies the first 68% of this combined pipeline.
                mapped_progress = 0.69 + 0.05 * min(
                    1.0,
                    max(0.0, (progress_value - 0.09) / 0.11),
                )
                _progress(
                    project_root,
                    stage,
                    detail,
                    mapped_progress,
                    compute_backend=backend,
                    metrics={**metrics, "calibratedCameraRays": True},
                )

            def publish_rgb_preview(
                points: np.ndarray,
                colors: np.ndarray,
                status: dict[str, Any],
            ) -> None:
                _check_cancelled(project_root)
                preview = [
                    {
                        "position": [float(point[0]), float(point[1]), float(point[2])],
                        "color": [int(color[0]), int(color[1]), int(color[2])],
                    }
                    for point, color in zip(points, colors, strict=True)
                ]
                _write_json_atomic(project_root / "outputs" / "build-preview.json", preview)
                _write_json_atomic(
                    project_root / "outputs" / "rgb-preview-status.json",
                    status,
                )

            lingbot_geometry = infer_lingbot_geometry_isolated(
                geometry_worker,
                context_paths,
                work_root=staging / "geometry-ipc",
                cancel_path=project_root / "outputs" / "cancel.flag",
                normalized_rays=calibrated_rays,
                output_indices=lingbot_output_indices,
                progress=report_lingbot,
                preview=publish_rgb_preview if progressive_rgb_preview else None,
            )
            _check_cancelled(project_root)
        if mapanything_proposal is not None:
            try:
                mapanything_geometry, mapanything_quality = _align_lingbot_geometry(
                    mapanything_proposal,
                    reconstruction,
                    [str(record["image"]) for record in records],
                )
                if not _accept_lingbot_alignment(mapanything_quality):
                    mapanything_geometry = None
                    mapanything_error = "camera proposal failed the COLMAP agreement gate"
            except Exception as error:
                mapanything_geometry = None
                mapanything_error = str(error)
            _check_cancelled(project_root)
        if da3_proposal is not None:
            try:
                da3_geometry, da3_quality = _align_lingbot_geometry(
                    da3_proposal,
                    reconstruction,
                    [str(record["image"]) for record in records],
                )
                if not _accept_lingbot_alignment(da3_quality):
                    da3_geometry = None
                    da3_error = "camera proposal failed the COLMAP agreement gate"
            except Exception as error:
                da3_geometry = None
                da3_error = str(error)
            _check_cancelled(project_root)
        chosen_model = staging / "chosen-model"
        chosen_model.mkdir(parents=True, exist_ok=True)
        reconstruction.write(chosen_model)
        alignment_quality: dict[str, Any] = {}
        alignment_accepted = False
        alignment_mode: str | None = None
        lingbot_alignment_quality: dict[str, Any] = {}
        lingbot_alignment_accepted = False
        lingbot_alignment_mode: str | None = None
        recovered_camera_count = 0
        dense_seed_count = 0
        if lingbot_geometry is not None:
            _progress(
                project_root,
                "camera_refinement",
                "Aligning LingBot dense geometry to the bundle-adjusted COLMAP cameras",
                0.75,
                compute_backend="LingBot-Map + COLMAP similarity alignment",
                metrics={
                    "lingbotCameraCount": len(lingbot_geometry.world_from_cameras),
                    "colmapCameraCount": reconstruction.num_reg_images(),
                },
            )
            aligned_geometry, alignment_quality = _align_lingbot_geometry(
                lingbot_geometry,
                reconstruction,
                [str(record["image"]) for record in records],
            )
            alignment_accepted = _accept_lingbot_alignment(alignment_quality)
            if alignment_accepted:
                lingbot_geometry = aligned_geometry
                alignment_mode = "global_similarity"
            else:
                alignment_mode = "rejected_to_colmap"
                _progress(
                    project_root,
                    "camera_refinement",
                    "LingBot geometry failed the COLMAP agreement gate; using validated SfM only",
                    0.76,
                    compute_backend="COLMAP quality fallback",
                    metrics=alignment_quality,
                )
                lingbot_geometry = None
            lingbot_alignment_quality = dict(alignment_quality)
            lingbot_alignment_accepted = alignment_accepted
            lingbot_alignment_mode = alignment_mode
        if single_video and mapanything_geometry is not None:
            map_residual = float(
                mapanything_quality.get("normalizedMedianCameraCenterResidual", math.inf)
            )
            selected_residual = float(
                alignment_quality.get("normalizedMedianCameraCenterResidual", math.inf)
            )
            if lingbot_geometry is None or map_residual < selected_residual:
                lingbot_geometry = mapanything_geometry
                alignment_quality = mapanything_quality
                alignment_accepted = True
                alignment_mode = "mapanything_global_similarity"
        if single_video and da3_geometry is not None:
            da3_residual = float(
                da3_quality.get("normalizedMedianCameraCenterResidual", math.inf)
            )
            selected_residual = float(
                alignment_quality.get("normalizedMedianCameraCenterResidual", math.inf)
            )
            if lingbot_geometry is None or da3_residual < selected_residual:
                lingbot_geometry = da3_geometry
                alignment_quality = da3_quality
                alignment_accepted = True
                alignment_mode = "da3_global_similarity"
        photo_learned_prior: LingbotGeometry | None = None
        if not single_video:
            candidates = [
                (geometry, quality)
                for geometry, quality in (
                    (mapanything_geometry, mapanything_quality),
                    (da3_geometry, da3_quality),
                )
                if geometry is not None
            ]
            if candidates:
                photo_learned_prior = min(
                    candidates,
                    key=lambda value: float(
                        value[1].get("normalizedMedianCameraCenterResidual", math.inf)
                    ),
                )[0]
        if lingbot_geometry is not None:
            _progress(
                project_root,
                "image_undistortion",
                "Undistorting every selected video view with shared calibrated intrinsics",
                0.77,
                compute_backend="COLMAP calibrated image resampling",
                metrics={"imageCount": len(records)},
            )
            frames, recovered_camera_count = _undistort_complete_video(
                pycolmap,
                input_images,
                undistorted / "images",
                records,
                reconstruction,
                lingbot_geometry,
                use_colmap_poses=alignment_accepted,
                learned_pose_source={
                    "global_similarity": "lingbot_map_aligned",
                    "mapanything_global_similarity": "mapanything_aligned",
                    "da3_global_similarity": "da3_nested_aligned",
                }.get(alignment_mode, "learned_geometry"),
            )
            points = lingbot_geometry.points
            colors = lingbot_geometry.colors
            dense_seed_count = len(points)
            _write_initialization(undistorted / "initialization.ply", points, colors)
            _write_initialization_parameters(
                undistorted / "initialization-parameters.npz",
                points,
                colors,
                lingbot_geometry.scales,
                lingbot_geometry.quaternions,
                confidence=_geometry_fusion_confidence(lingbot_geometry),
                opacity=lingbot_geometry.opacities,
                fusion_confidence=_geometry_fusion_confidence(lingbot_geometry),
                source_frame_indices=lingbot_geometry.source_frame_indices,
                provenance=np.full(len(points), 2, dtype=np.uint8),
            )
            _sparse_points, _sparse_colors, point_quality = _point_records(reconstruction)
        else:
            _progress(project_root, "image_undistortion", "Undistorting registered views at source resolution", 0.70)
            pycolmap.undistort_images(
                undistorted,
                chosen_model,
                input_images,
                output_type="COLMAP",
                copy_policy=pycolmap.FileCopyType.copy,
                jpeg_quality=100,
                num_threads=-1,
            )
            _check_cancelled(project_root)
            undistorted_model_root = undistorted / "sparse"
            if (undistorted_model_root / "cameras.bin").is_file():
                undistorted_reconstruction = pycolmap.Reconstruction(undistorted_model_root)
            elif (undistorted_model_root / "0" / "cameras.bin").is_file():
                undistorted_reconstruction = pycolmap.Reconstruction(undistorted_model_root / "0")
            else:
                raise RuntimeError("COLMAP did not publish an undistorted sparse model")
            points, colors, point_quality = _point_records(undistorted_reconstruction)
            if photo_learned_prior is not None:
                points = photo_learned_prior.points
                colors = photo_learned_prior.colors
                dense_seed_count = len(points)
            _write_initialization(undistorted / "initialization.ply", points, colors)
            if photo_learned_prior is not None:
                _write_initialization_parameters(
                    undistorted / "initialization-parameters.npz",
                    points,
                    colors,
                    photo_learned_prior.scales,
                    photo_learned_prior.quaternions,
                    confidence=_geometry_fusion_confidence(photo_learned_prior),
                    opacity=photo_learned_prior.opacities,
                    fusion_confidence=_geometry_fusion_confidence(photo_learned_prior),
                    source_frame_indices=photo_learned_prior.source_frame_indices,
                    provenance=np.full(len(points), 2, dtype=np.uint8),
                )

            source_by_name = {record["image"]: record for record in records}
            source_index_by_name = {
                str(record["image"]): index for index, record in enumerate(records)
            }
            frames = []
            for frame_index, image in enumerate(
                sorted(undistorted_reconstruction.images.values(), key=lambda value: value.name)
            ):
                camera = undistorted_reconstruction.cameras[image.camera_id]
                image_path = undistorted / "images" / image.name
                if not image_path.is_file():
                    continue
                source = source_by_name.get(image.name, {})
                frames.append(
                    {
                        "phaseId": "media",
                        "frameIndex": frame_index,
                        "sourceFrameIndex": source_index_by_name[image.name],
                        "timestampUs": (
                            round(float(source["timestampSeconds"]) * 1_000_000)
                            if source.get("timestampSeconds") is not None
                            else None
                        ),
                        "intrinsics": _camera_intrinsics(camera),
                        "worldFromRgbCamera": _world_from_camera(image),
                        "image": f"images/{image.name}",
                        "sourcePath": source.get("source"),
                        "sharpness": source.get("sharpness"),
                        "poseConfidence": 1.0,
                        "poseSource": "colmap_bundle_adjusted",
                        "poseAnchor": frame_index == 0,
                    }
                )
            if len(frames) != undistorted_reconstruction.num_reg_images():
                raise RuntimeError("One or more registered images were missing after undistortion")
        _check_cancelled(project_root)
        luminances: list[float] = []
        for frame in frames:
            image_path = undistorted / str(frame["image"])
            with Image.open(image_path) as opened:
                sample = ImageOps.grayscale(opened)
                sample.thumbnail((128, 128), Image.Resampling.BILINEAR)
                luminances.append(float(np.asarray(sample, dtype=np.float32).mean() / 255.0))
        video_intrinsic_spread = _video_intrinsic_spread(frames)
        if video_intrinsic_spread > 0.005:
            raise RuntimeError(
                "Undistortion produced inconsistent intrinsics for one locked-settings "
                f"video ({video_intrinsic_spread * 100.0:.2f}% normalized spread)"
            )
        appearance_anchor = (
            int(np.argmin(np.abs(np.asarray(luminances) - np.median(luminances))))
            if luminances
            else 0
        )
        warnings: list[str] = []
        if mapanything_evaluated and mapanything_geometry is None:
            warnings.append(
                "MapAnything camera/depth challenger was excluded; "
                f"{mapanything_error or 'it did not pass COLMAP camera agreement'}."
            )
        if da3_evaluated and da3_geometry is None:
            warnings.append(
                "DA3 Nested Giant-Large challenger was excluded; "
                f"{da3_error or 'it did not pass COLMAP camera agreement'}."
            )
        if da3_evaluated:
            warnings.append(
                "DA3 Nested Giant-Large 1.1 model output is restricted to noncommercial use (CC BY-NC 4.0)."
            )
        if alignment_mode == "rejected_to_colmap":
            warnings.append(
                "LingBot geometry failed the COLMAP camera-agreement gate and was excluded; "
                f"ScanLan retained {len(frames)} validated cameras rather than publishing "
                "an incoherent reconstruction."
            )
        elif solve["registrationRatio"] < 0.85 and lingbot_geometry is None:
            warnings.append(
                f"Only {solve['registeredImageCount']} of {solve['inputImageCount']} views formed one consistent model."
            )
        elif lingbot_geometry is not None and not alignment_accepted:
            warnings.append(
                "COLMAP's trajectory disagreed with the complete LingBot-Map path; "
                "ScanLan retained LingBot poses and used COLMAP only for calibrated camera rays and lens correction."
            )
        elif recovered_camera_count:
            warnings.append(
                f"COLMAP validated {solve['registeredImageCount']} views; "
                f"{lingbot_geometry.backend if lingbot_geometry is not None else 'learned geometry'} recovered "
                f"the remaining {recovered_camera_count} trajectory poses for joint optimization."
            )
        if point_quality["medianReprojectionErrorPx"] > 1.5:
            warnings.append("Camera reprojection error is higher than the preferred 1.5 px quality gate.")
        selected_gaussian_prior = (
            lingbot_geometry
            if lingbot_geometry is not None
            else photo_learned_prior
        )
        direct_gaussian_initialization = bool(
            selected_gaussian_prior is not None
            and selected_gaussian_prior.opacities is not None
        )
        initialization_kind = (
            InitializationKind.DIRECT_GAUSSIAN
            if direct_gaussian_initialization
            else InitializationKind.DENSE_SURFACE
            if selected_gaussian_prior is not None
            else InitializationKind.SPARSE_SFM
        )
        initialization_representation = (
            GaussianRepresentation.PREDICTED_ANISOTROPIC_3D
            if direct_gaussian_initialization
            else GaussianRepresentation.VOLUMETRIC_3D
        )
        dataset = {
            "schemaVersion": 3,
            "fingerprint": fingerprint,
            "metric": False,
            "sourceType": "video" if video_statistics and not any(path.suffix.lower() in PHOTO_EXTENSIONS for path in sources) else "photos",
            "initialization": "initialization.ply",
            "initializationParameters": (
                "initialization-parameters.npz"
                if lingbot_geometry is not None or photo_learned_prior is not None
                else None
            ),
            "gaussianInitialization": initialization_manifest(
                initialization_kind,
                initialization_representation,
                parameters=(
                    "initialization-parameters.npz"
                    if selected_gaussian_prior is not None
                    else None
                ),
                adaptive_densification=True,
            ),
            "denseGeometryPrior": selected_gaussian_prior is not None,
            "directGaussianPrior": direct_gaussian_initialization,
            "poseRefinement": True,
            # A single video is captured with locked focal length, exposure,
            # white balance, shutter, and ISO. Per-frame color transforms would
            # give the optimizer a way to explain away geometric disagreement.
            "appearanceOptimization": not (
                len(video_statistics) == 1
                and not any(path.suffix.lower() in PHOTO_EXTENSIONS for path in sources)
            ),
            "appearanceAnchorIndex": appearance_anchor,
            "coordinateConvention": {
                "handedness": "right",
                "cameraAxes": "opencv_x_right_y_down_z_forward",
                "pose": "worldFromCamera",
                "matrixStorage": "row-major",
            },
            "quality": {
                **solve,
                **point_quality,
                "geometryBackend": (
                    lingbot_geometry.backend
                    if lingbot_geometry is not None
                    else photo_learned_prior.backend
                    if photo_learned_prior is not None
                    else "COLMAP sparse structure-from-motion"
                ),
                "trainingViewCount": len(frames),
                "lingbotRecoveredCameraCount": recovered_camera_count,
                "lingbotExcludedCameraCount": (
                    solve["inputImageCount"] - len(frames)
                    if alignment_mode == "rejected_to_colmap"
                    else 0
                ),
                "denseSeedCount": dense_seed_count,
                "selectedGeometryAlignment": alignment_quality or None,
                "selectedGeometryAlignmentAccepted": alignment_accepted,
                "selectedGeometryAlignmentMode": alignment_mode,
                "lingbotAlignment": lingbot_alignment_quality or None,
                "lingbotAlignmentAccepted": lingbot_alignment_accepted,
                "lingbotAlignmentMode": lingbot_alignment_mode,
                "lingbotCalibratedCameraRays": lingbot_context is not None,
                "lingbotEvaluated": lingbot_context is not None,
                "lingbotContext": lingbot_context,
                "mapAnythingEvaluated": mapanything_evaluated,
                "mapAnythingAlignment": mapanything_quality or None,
                "mapAnythingAlignmentAccepted": mapanything_geometry is not None,
                "mapAnythingError": mapanything_error,
                "da3Evaluated": da3_evaluated,
                "da3Alignment": da3_quality or None,
                "da3AlignmentAccepted": da3_geometry is not None,
                "da3Error": da3_error,
                "da3Streaming": da3_streaming,
                "da3DirectGaussiansRequested": da3_direct_gaussians,
                "da3DirectGaussians": da3_direct_gaussians_used and da3_geometry is not None,
                "da3License": "CC-BY-NC-4.0" if da3_evaluated else None,
                "maximumVideoIntrinsicSpread": video_intrinsic_spread,
                "videoSources": video_statistics,
                "warnings": warnings,
                "preparationSeconds": round(time.perf_counter() - started, 3),
            },
            "frames": frames,
        }
        _write_json_atomic(undistorted / "dataset.json", dataset)
        shutil.rmtree(undistorted / "stereo", ignore_errors=True)
        shutil.rmtree(undistorted / "sparse", ignore_errors=True)
        shutil.rmtree(input_images, ignore_errors=True)
        shutil.rmtree(sfm_workspace, ignore_errors=True)
        shutil.rmtree(chosen_model, ignore_errors=True)
        published_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(undistorted, published_root)
        _publish_pointer(project_root, published_root)
        _progress(
            project_root,
            "media_ready",
            (
                f"Prepared {len(frames):,} cameras and {len(points):,} confidence-gated dense learned seeds"
                if lingbot_geometry is not None or photo_learned_prior is not None
                else f"Solved {len(frames):,} cameras and {len(points):,} sparse seed points"
            ),
            1.0,
            compute_backend=(
                (
                    f"{lingbot_geometry.backend} + COLMAP bundle-adjusted poses/intrinsics"
                    if alignment_accepted
                    else f"{lingbot_geometry.backend} + COLMAP calibrated intrinsics"
                )
                if lingbot_geometry is not None
                else f"{photo_learned_prior.backend} + COLMAP bundle-adjusted poses/intrinsics"
                if photo_learned_prior is not None
                else "COLMAP structure-from-motion"
            ),
            metrics={
                "trainingViewCount": len(frames),
                "seedCount": len(points),
                "recoveredCameraCount": recovered_camera_count,
            },
        )
        return dataset
    finally:
        shutil.rmtree(staging, ignore_errors=True)
