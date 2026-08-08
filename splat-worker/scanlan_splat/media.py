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
    video_fps: float = 1.0
    maximum_video_frames: int = 240
    maximum_image_dimension: int = 2560
    minimum_image_dimension: int = 480
    maximum_features: int = 8_192


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
    digest.update(b"scanlan-media-observations-v1-streaming-jpeg95\0")
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
    # v8 also fingerprints the bounded multi-model and bundle-adjustment
    # schedule so an older rejected camera analysis is never silently reused.
    digest.update(b"scanlan-media-dataset-v8-bounded-shared-video-camera\0")
    digest.update(observation_fingerprint.encode("ascii"))
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
    image = ImageOps.exif_transpose(image).convert("RGB")
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


def _video_candidates(
    source: Path,
    target_fps: float,
) -> tuple[list[tuple[float, Image.Image, float, np.ndarray]], dict[str, Any]]:
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
        evaluation_interval = 1.0 / max(target_fps * 3.0, 1.0)
        next_evaluation = 0.0
        candidates: list[tuple[float, Image.Image, float, np.ndarray]] = []
        decoded = 0
        for frame in container.decode(stream):
            decoded += 1
            timestamp = float(frame.time) if frame.time is not None else (
                decoded / source_rate if source_rate else float(decoded)
            )
            if timestamp + 1e-6 < next_evaluation:
                continue
            next_evaluation = timestamp + evaluation_interval
            image = frame.to_image().convert("RGB")
            sharpness, descriptor = _sharpness_and_descriptor(image)
            candidates.append((timestamp, image, sharpness, descriptor))
        return candidates, {
            "path": str(source),
            "durationSeconds": duration,
            "sourceFps": source_rate,
            "decodedFrameCount": decoded,
            "candidateFrameCount": len(candidates),
        }
    finally:
        container.close()


def _select_video_candidates(
    candidates: Sequence[tuple[float, Image.Image, float, np.ndarray]],
    target_fps: float,
    maximum_frames: int,
) -> list[tuple[float, Image.Image, float, np.ndarray]]:
    if not candidates:
        return []
    bucket_width = 1.0 / max(target_fps, 0.1)
    best_by_bucket: dict[int, tuple[float, Image.Image, float, np.ndarray]] = {}
    for candidate in candidates:
        bucket = int(candidate[0] / bucket_width)
        previous = best_by_bucket.get(bucket)
        if previous is None or candidate[2] > previous[2]:
            best_by_bucket[bucket] = candidate
    selected: list[tuple[float, Image.Image, float, np.ndarray]] = []
    for candidate in sorted(best_by_bucket.values(), key=lambda value: value[0]):
        if selected and _descriptor_distance(selected[-1][3], candidate[3]) < 0.012:
            if candidate[2] > selected[-1][2] * 1.15:
                selected[-1] = candidate
            continue
        selected.append(candidate)
    if len(selected) > maximum_frames:
        indices = np.linspace(0, len(selected) - 1, maximum_frames, dtype=np.int64)
        selected = [selected[int(index)] for index in indices]
    return selected


def _extract_video_streaming(
    source: Path,
    images_root: Path,
    first_frame_index: int,
    options: MediaPreparationOptions,
    maximum_frames: int,
    project_root: Path,
    progress: Callable[[float, str, int | None, dict[str, Any]], None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select and write sharp video views with constant full-frame memory.

    The previous implementation retained every evaluated RGB frame until the
    entire video had decoded. A few minutes of 4K video could therefore retain
    many gigabytes and turn normal decoding into paging. This version holds one
    best decoded frame for the active time bucket plus one pending selected RGB
    image used for duplicate suppression.
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
        effective_fps = max(options.video_fps, 1e-4)
        if duration and duration > 0.0 and maximum_frames > 1:
            effective_fps = min(
                effective_fps,
                (maximum_frames - 1) / duration,
            )
        bucket_width = 1.0 / max(effective_fps, 1e-6)
        evaluation_interval = 1.0 / max(effective_fps * 3.0, 0.03)
        next_evaluation = 0.0
        active_bucket: int | None = None
        best: tuple[float, Any, float, np.ndarray] | None = None
        pending: tuple[float, Image.Image, float, np.ndarray] | None = None
        records: list[dict[str, Any]] = []
        decoded = 0
        evaluated = 0
        duplicate_count = 0
        started = time.perf_counter()
        last_progress_at = 0.0

        def write_selected(candidate: tuple[float, Image.Image, float, np.ndarray]) -> None:
            if len(records) >= maximum_frames:
                return
            timestamp, image, sharpness, descriptor = candidate
            destination = images_root / f"video-{first_frame_index + len(records):06d}.jpg"
            width, height = _save_canonical_image(
                image,
                destination,
                options.maximum_image_dimension,
            )
            if min(width, height) < options.minimum_image_dimension:
                destination.unlink(missing_ok=True)
                return
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
                }
            )

        def accept_bucket(candidate: tuple[float, Any, float, np.ndarray] | None) -> None:
            nonlocal pending, duplicate_count
            if candidate is None:
                return
            timestamp, frame, sharpness, descriptor = candidate
            output_size = _limited_size(
                int(frame.width),
                int(frame.height),
                options.maximum_image_dimension,
            )
            image = frame.to_image(width=output_size[0], height=output_size[1]).convert("RGB")
            current = (timestamp, image, sharpness, descriptor)
            if pending is None:
                pending = current
                return
            if _descriptor_distance(pending[3], descriptor) < 0.012:
                duplicate_count += 1
                if sharpness > pending[2] * 1.15:
                    pending = current
                return
            write_selected(pending)
            pending = current

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
                        "selectedFrames": len(records) + (1 if pending is not None else 0),
                        "sourceTimestampSeconds": round(timestamp, 3),
                        "sourceDurationSeconds": duration,
                    },
                )
                last_progress_at = now
            if timestamp + 1e-6 < next_evaluation:
                continue
            next_evaluation = timestamp + evaluation_interval
            bucket = int(timestamp / bucket_width)
            if active_bucket is not None and bucket != active_bucket:
                accept_bucket(best)
                best = None
            active_bucket = bucket
            sample_size = _limited_size(int(frame.width), int(frame.height), 320)
            sample = frame.to_image(width=sample_size[0], height=sample_size[1]).convert("RGB")
            sharpness, descriptor = _sharpness_and_descriptor(sample)
            evaluated += 1
            candidate = (timestamp, frame, sharpness, descriptor)
            if best is None or sharpness > best[2]:
                best = candidate

        accept_bucket(best)
        if pending is not None:
            write_selected(pending)
        progress(
            1.0,
            f"Selected {len(records):,} sharp keyframes from {source.name}",
            0,
            {
                "decodedFrames": decoded,
                "evaluatedFrames": evaluated,
                "selectedFrames": len(records),
                "duplicateFrames": duplicate_count,
                "sourceDurationSeconds": duration,
            },
        )
        return records, {
            "path": str(source),
            "durationSeconds": duration,
            "sourceFps": source_rate,
            "effectiveSelectionFps": effective_fps,
            "decodedFrameCount": decoded,
            "candidateFrameCount": evaluated,
            "selectedFrameCount": len(records),
            "duplicateFrameCount": duplicate_count,
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
    extraction.sift.max_num_features = maximum_features
    extraction.sift.peak_threshold = 0.004
    extraction.sift.edge_threshold = 12.0
    extraction.sift.max_num_orientations = 1
    # Covariant affine/DSP SIFT is prohibitively slow in the CPU-only Windows
    # PyCOLMAP wheel (roughly a minute per 4K view). Dense standard SIFT plus
    # guided geometric matching has materially better end-to-end throughput
    # and retains the source-resolution camera solve.
    extraction.sift.estimate_affine_shape = False
    extraction.sift.domain_size_pooling = False

    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = worker_threads
    matching.use_gpu = use_cuda
    matching.max_num_matches = 32_768
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


def _run_sfm(
    images_root: Path,
    workspace: Path,
    records: Sequence[dict[str, Any]],
    options: MediaPreparationOptions,
    project_root: Path,
    sequential: bool,
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
        0.10,
        compute_backend=feature_backend,
        metrics={
            "imageCount": image_count,
            "processedImages": 0,
            "cudaValidated": bool(feature_runtime["cudaValidated"]),
            "cudaError": runtime_error,
        },
    )
    image_names = sorted(path.name for path in images_root.iterdir() if path.is_file())
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
            batch_progress = 0.10 + 0.15 * processed_before / max(image_count, 1)
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
                0.10 + 0.15 * processed_after / max(image_count, 1),
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
        0.25,
        compute_backend=feature_backend,
        metrics={"imageCount": image_count},
    )
    if not feature_runtime["cudaValidated"] and hasattr(pycolmap, "match_image_pairs"):
        pairs = _cpu_match_pairs(
            image_names,
            sequential=sequential and image_count > 120,
        )
        pair_batch_size = max(512, int(matching.num_threads) * 64)
        for batch_index, start in enumerate(range(0, len(pairs), pair_batch_size)):
            batch = pairs[start : start + pair_batch_size]
            pair_path = workspace / f"match-pairs-{batch_index:05d}.txt"
            pair_path.write_text(
                "".join(f"{left} {right}\n" for left, right in batch),
                encoding="utf-8",
            )
            pairing = pycolmap.ImportedPairingOptions()
            pairing.block_size = min(pair_batch_size, len(batch))
            pairing.match_list_path = pair_path
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
                f"Matched {processed:,} of {len(pairs):,} CPU image pairs",
                0.25 + 0.18 * processed / max(len(pairs), 1),
                compute_backend=feature_backend,
                metrics={
                    "pairCount": len(pairs),
                    "processedPairs": processed,
                    "workerThreads": int(matching.num_threads),
                },
            )
            _check_cancelled(project_root)
    else:
        with _progress_heartbeat(
            project_root,
            "feature_matching",
            "Matching overlapping views with geometric verification",
            0.25,
            compute_backend=feature_backend,
            metrics={"imageCount": image_count},
        ):
            if sequential and image_count > 120:
                pairing = pycolmap.SequentialPairingOptions()
                pairing.overlap = 16
                pairing.quadratic_overlap = True
                pycolmap.match_sequential(
                    database,
                    matching_options=matching,
                    pairing_options=pairing,
                    verification_options=verification,
                    device=feature_device,
                )
            else:
                pairing = pycolmap.ExhaustivePairingOptions()
                pairing.block_size = 50
                pycolmap.match_exhaustive(
                    database,
                    matching_options=matching,
                    pairing_options=pairing,
                    verification_options=verification,
                    device=feature_device,
                )
    _check_cancelled(project_root)
    camera_telemetry = _CameraSolveTelemetry(image_count)
    with _progress_heartbeat(
        project_root,
        "camera_solving",
        "Solving cameras and globally refining structure",
        0.45,
        compute_backend="COLMAP incremental mapper / CPU",
        metrics={"imageCount": image_count},
        progress_provider=camera_telemetry.progress,
        detail_provider=camera_telemetry.detail,
        metrics_provider=camera_telemetry.snapshot,
    ):
        models = pycolmap.incremental_mapping(
            database,
            images_root,
            models_root,
            options=mapping,
            initial_image_pair_callback=camera_telemetry.initial_pair_registered,
            next_image_callback=camera_telemetry.next_image_registered,
        )
    if not models:
        raise RuntimeError(
            "Camera solving found no consistent reconstruction. Capture more overlap, texture, and parallax."
        )
    ranked = sorted(
        models.values(),
        key=lambda model: (model.num_reg_images(), model.num_points3D()),
        reverse=True,
    )
    reconstruction = ranked[0]
    registered = reconstruction.num_reg_images()
    minimum_registered = _minimum_useful_registration_count(image_count)
    if registered < minimum_registered:
        raise RuntimeError(
            f"Camera solving registered only {registered} of {image_count} images "
            f"(minimum {minimum_registered}); capture a slower path with at least "
            "60-80% overlap and avoid motion blur."
        )
    statistics = {
        "inputImageCount": image_count,
        "registeredImageCount": registered,
        "registrationRatio": registered / image_count,
        "sparsePointCount": reconstruction.num_points3D(),
        "modelCount": len(ranked),
        "excludedImageCount": image_count - registered,
        "featureBackend": feature_backend,
        "matching": "sequential quadratic overlap" if sequential and image_count > 120 else "exhaustive",
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


def _publish_pointer(project_root: Path, dataset_root: Path) -> Path:
    datasets = project_root / "outputs" / "cache" / "datasets"
    relative = os.path.relpath(dataset_root, datasets).replace("\\", "/")
    pointer = datasets / "current.json"
    _write_json_atomic(pointer, {"schemaVersion": 1, "path": relative})
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
        reconstruction, solve = _run_sfm(
            input_images,
            sfm_workspace,
            records,
            options,
            project_root,
            sequential=bool(video_statistics),
        )
        _check_cancelled(project_root)
        chosen_model = staging / "chosen-model"
        chosen_model.mkdir(parents=True, exist_ok=True)
        reconstruction.write(chosen_model)
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
        _write_initialization(undistorted / "initialization.ply", points, colors)

        source_by_name = {record["image"]: record for record in records}
        frames: list[dict[str, Any]] = []
        luminances: list[float] = []
        for frame_index, image in enumerate(
            sorted(undistorted_reconstruction.images.values(), key=lambda value: value.name)
        ):
            camera = undistorted_reconstruction.cameras[image.camera_id]
            image_path = undistorted / "images" / image.name
            if not image_path.is_file():
                continue
            with Image.open(image_path) as opened:
                sample = ImageOps.grayscale(opened)
                sample.thumbnail((128, 128), Image.Resampling.BILINEAR)
                luminances.append(float(np.asarray(sample, dtype=np.float32).mean() / 255.0))
            source = source_by_name.get(image.name, {})
            frames.append(
                {
                    "phaseId": "media",
                    "frameIndex": frame_index,
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
                }
            )
        if len(frames) != undistorted_reconstruction.num_reg_images():
            raise RuntimeError("One or more registered images were missing after undistortion")
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
        if solve["registrationRatio"] < 0.85:
            warnings.append(
                f"Only {solve['registeredImageCount']} of {solve['inputImageCount']} views formed one consistent model."
            )
        if point_quality["medianReprojectionErrorPx"] > 1.5:
            warnings.append("Camera reprojection error is higher than the preferred 1.5 px quality gate.")
        dataset = {
            "schemaVersion": 3,
            "fingerprint": fingerprint,
            "metric": False,
            "sourceType": "video" if video_statistics and not any(path.suffix.lower() in PHOTO_EXTENSIONS for path in sources) else "photos",
            "initialization": "initialization.ply",
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
            f"Solved {len(frames):,} cameras and {len(points):,} sparse seed points",
            1.0,
        )
        return dataset
    finally:
        shutil.rmtree(staging, ignore_errors=True)
