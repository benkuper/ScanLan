from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps


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
    video_fps: float = 2.0
    maximum_video_frames: int = 600
    maximum_image_dimension: int = 4096
    minimum_image_dimension: int = 480
    maximum_features: int = 12_000


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _progress(project_root: Path, stage: str, detail: str, progress: float) -> None:
    _write_json_atomic(
        project_root / "outputs" / "splat-progress.json",
        {
            "stage": stage,
            "detail": detail,
            "progress": float(np.clip(progress, 0.0, 1.0)),
            "stageProgress": float(np.clip(progress, 0.0, 1.0)),
            "iteration": None,
            "totalIterations": None,
            "etaSeconds": None,
            "stageEtaSeconds": None,
            "computeBackend": "COLMAP structure-from-motion",
        },
    )


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
    digest = hashlib.sha256()
    digest.update(b"scanlan-media-dataset-v3\0")
    digest.update(json.dumps(options.__dict__, sort_keys=True).encode("utf-8"))
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
        "quality": 98,
        "subsampling": 0,
        "optimize": True,
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
            0.08 * (source_index + 1) / max(len(photo_sources), 1),
        )
    video_frame_index = 0
    for video_index, source in enumerate(video_sources):
        _check_cancelled(project_root)
        candidates, video = _video_candidates(source, options.video_fps)
        selected = _select_video_candidates(
            candidates,
            options.video_fps,
            max(2, options.maximum_video_frames // max(1, len(video_sources))),
        )
        video["selectedFrameCount"] = len(selected)
        videos.append(video)
        for timestamp, image, sharpness, descriptor in selected:
            destination = images_root / f"video-{video_frame_index:06d}.jpg"
            width, height = _save_canonical_image(
                image,
                destination,
                options.maximum_image_dimension,
            )
            if min(width, height) < options.minimum_image_dimension:
                destination.unlink(missing_ok=True)
                continue
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
            video_frame_index += 1
        _progress(
            project_root,
            "media_decode",
            f"Selected {len(selected):,} sharp keyframes from video {video_index + 1:,} of {len(video_sources):,}",
            0.08 * (video_index + 1) / max(len(video_sources), 1),
        )
    if len(records) < 3:
        raise ValueError("At least three usable, overlapping images are required")
    return records, videos


def _configure_sfm(image_count: int, maximum_features: int) -> tuple[Any, ...]:
    import pycolmap

    worker_threads = max(1, min(os.cpu_count() or 1, 8))

    reader = pycolmap.ImageReaderOptions()
    # A single radial parameter is substantially better conditioned than the
    # full OPENCV model for casual phone sweeps with only a few dozen views.
    # COLMAP still refines it, then the canonicalizer removes it completely.
    reader.camera_model = "SIMPLE_RADIAL"
    reader.default_focal_length_factor = 1.2

    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = 4096
    extraction.num_threads = worker_threads
    extraction.use_gpu = bool(pycolmap.has_cuda)
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
    matching.use_gpu = bool(pycolmap.has_cuda)
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
    mapping.max_num_models = min(10, max(2, image_count // 10))
    mapping.min_model_size = min(8, max(3, image_count // 4))
    mapping.ba_global_max_num_iterations = 100
    mapping.ba_global_function_tolerance = 1e-7
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


def _run_sfm(
    images_root: Path,
    workspace: Path,
    image_count: int,
    options: MediaPreparationOptions,
    project_root: Path,
    sequential: bool,
) -> tuple[Any, dict[str, Any]]:
    import pycolmap

    database = workspace / "database.db"
    models_root = workspace / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    reader, extraction, matching, verification, mapping = _configure_sfm(
        image_count,
        options.maximum_features,
    )
    _progress(project_root, "feature_extraction", f"Extracting high-detail features from {image_count:,} images", 0.10)
    pycolmap.extract_features(
        database,
        images_root,
        camera_mode=pycolmap.CameraMode.AUTO,
        reader_options=reader,
        extraction_options=extraction,
        device=pycolmap.Device.auto,
    )
    _check_cancelled(project_root)
    _progress(project_root, "feature_matching", "Matching overlapping views with geometric verification", 0.25)
    if sequential and image_count > 120:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = 16
        pairing.quadratic_overlap = True
        pycolmap.match_sequential(
            database,
            matching_options=matching,
            pairing_options=pairing,
            verification_options=verification,
            device=pycolmap.Device.auto,
        )
    else:
        pairing = pycolmap.ExhaustivePairingOptions()
        pairing.block_size = 50
        pycolmap.match_exhaustive(
            database,
            matching_options=matching,
            pairing_options=pairing,
            verification_options=verification,
            device=pycolmap.Device.auto,
        )
    _check_cancelled(project_root)
    _progress(project_root, "camera_solving", "Solving cameras and globally refining structure", 0.45)
    models = pycolmap.incremental_mapping(database, images_root, models_root, options=mapping)
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
    minimum_registered = max(3, math.ceil(image_count * 0.5))
    if registered < minimum_registered:
        raise RuntimeError(
            f"Camera solving registered only {registered} of {image_count} images; "
            "capture a slower path with at least 60-80% overlap and avoid motion blur."
        )
    statistics = {
        "inputImageCount": image_count,
        "registeredImageCount": registered,
        "registrationRatio": registered / image_count,
        "sparsePointCount": reconstruction.num_points3D(),
        "modelCount": len(ranked),
        "excludedImageCount": image_count - registered,
        "featureBackend": "COLMAP SIFT GPU" if pycolmap.has_cuda else "COLMAP SIFT CPU",
        "matching": "sequential quadratic overlap" if sequential and image_count > 120 else "exhaustive",
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
    fingerprint = _source_fingerprint(sources, options)
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
        _progress(project_root, "media_decode", "Preparing full-quality input images", 0.0)
        records, video_statistics = _collect_images(
            sources,
            input_images,
            options,
            project_root,
        )
        reconstruction, solve = _run_sfm(
            input_images,
            sfm_workspace,
            len(records),
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
            "appearanceOptimization": True,
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
