from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def resolve_dataset(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        pointer = json.loads(path.read_text(encoding="utf-8"))
        return (path.parent / pointer["path"]).resolve()
    return path


def load_dataset(path: Path) -> tuple[Path, dict[str, Any]]:
    root = resolve_dataset(path)
    return root, json.loads((root / "dataset.json").read_text(encoding="utf-8"))


class _MediaProgress:
    def __init__(self, project_root: Path) -> None:
        self.path = project_root / "outputs" / "splat-progress.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()

    def update(
        self,
        stage: str,
        detail: str,
        progress: float,
        stage_progress: float,
        stage_eta_seconds: int | None = None,
    ) -> None:
        _write_json_atomic(
            self.path,
            {
                "stage": stage,
                "detail": detail,
                "progress": round(progress, 4),
                "stageProgress": round(stage_progress, 4),
                "stageEtaSeconds": stage_eta_seconds,
                "elapsedSeconds": round(time.perf_counter() - self.started),
                "computeBackend": "COLMAP CUDA + CPU mapper",
            },
        )


def _run(
    command: list[str],
    cwd: Path | None = None,
    on_wait: Callable[[float], None] | None = None,
    cancel_path: Path | None = None,
) -> None:
    process = subprocess.Popen(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    started = time.perf_counter()
    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - started
            if on_wait:
                on_wait(elapsed)
            if cancel_path is not None and cancel_path.exists():
                process.terminate()
                process.communicate()
                raise RuntimeError("Media registration cancelled")
    if process.returncode:
        detail = (stderr or stdout).strip()
        raise RuntimeError(f"{' '.join(command[:2])} failed: {detail[-3000:]}")


def _filter_images(
    source: Path,
    destination: Path,
    progress: Callable[[int, int, int], None] | None = None,
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    accepted: list[Path] = []
    previous: np.ndarray | None = None
    candidates = sorted(path for path in source.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"})
    for index, path in enumerate(candidates, start=1):
        with Image.open(path) as raw:
            image = raw.convert("RGB")
            thumbnail = np.asarray(image.resize((96, 64), Image.Resampling.BILINEAR), dtype=np.float32).mean(axis=2)
            gradient = np.diff(thumbnail, axis=0).var() + np.diff(thumbnail, axis=1).var()
            if gradient < 5.0:
                if progress:
                    progress(index, len(candidates), len(accepted))
                continue
            if previous is not None and float(np.mean(np.abs(thumbnail - previous))) < 2.2:
                if progress:
                    progress(index, len(candidates), len(accepted))
                continue
            output = destination / f"{len(accepted):06}.jpg"
            image.save(output, quality=95, optimize=True)
            accepted.append(output)
            previous = thumbnail
        if progress:
            progress(index, len(candidates), len(accepted))
    return accepted


def _run_phase(
    reporter: _MediaProgress,
    command: list[str],
    stage: str,
    detail: str,
    progress_start: float,
    progress_end: float,
    expected_seconds: int,
    cancel_path: Path,
) -> None:
    reporter.update(stage, detail, progress_start, 0.0)

    def waiting(elapsed: float) -> None:
        fraction = min(0.92, elapsed / max(expected_seconds, 1) * 0.8)
        eta = max(1, round(expected_seconds - elapsed)) if elapsed < expected_seconds else None
        reporter.update(
            stage,
            f"{detail} · {round(elapsed)}s elapsed",
            progress_start + (progress_end - progress_start) * fraction,
            fraction,
            eta,
        )

    _run(command, on_wait=waiting, cancel_path=cancel_path)
    reporter.update(stage, detail, progress_end, 1.0, 0)


def _qvec_rotation(values: list[float]) -> np.ndarray:
    w, x, y, z = values
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _read_colmap_model(model: Path) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray, float]:
    cameras: dict[int, dict[str, Any]] = {}
    for line in (model / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        values = line.split()
        camera_id, camera_model = int(values[0]), values[1]
        width, height = int(values[2]), int(values[3])
        parameters = [float(value) for value in values[4:]]
        if camera_model == "SIMPLE_PINHOLE":
            fx = fy = parameters[0]; cx, cy = parameters[1:3]; distortion = []
        elif camera_model == "PINHOLE":
            fx, fy, cx, cy = parameters[:4]; distortion = []
        elif camera_model in {"SIMPLE_RADIAL", "RADIAL"}:
            fx = fy = parameters[0]; cx, cy = parameters[1:3]; distortion = parameters[3:]
        else:
            fx, fy, cx, cy = parameters[:4]; distortion = parameters[4:]
        cameras[camera_id] = {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy, "model": camera_model.lower(), "distortion": distortion}

    lines = [line for line in (model / "images.txt").read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    images: list[dict[str, Any]] = []
    for line in lines[::2]:
        values = line.split()
        rotation = _qvec_rotation([float(value) for value in values[1:5]])
        translation = np.asarray([float(value) for value in values[5:8]])
        camera_from_world = np.eye(4)
        camera_from_world[:3, :3] = rotation
        camera_from_world[:3, 3] = translation
        images.append({"name": values[9], "camera": int(values[8]), "worldFromCamera": np.linalg.inv(camera_from_world)})

    points: list[list[float]] = []
    colors: list[list[int]] = []
    errors: list[float] = []
    for line in (model / "points3D.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        values = line.split()
        points.append([float(value) for value in values[1:4]])
        colors.append([int(value) for value in values[4:7]])
        errors.append(float(values[7]))
    return cameras, images, np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.uint8), float(np.mean(errors)) if errors else 0.0


def _write_initialization(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertices = np.empty(len(points), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = ("ply\nformat binary_little_endian 1.0\n" + f"element vertex {len(points)}\n" + "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode("ascii")
    with path.open("wb") as handle:
        handle.write(header); vertices.tofile(handle)


def prepare_media_dataset(project_root: Path, source_ids: list[str]) -> Path:
    reporter = _MediaProgress(project_root)
    reporter.update("media_preparing", "Reading imported media sources", 0.0, 0.0)
    cancel_path = project_root / "outputs" / "cancel.flag"
    project_path = project_root / "project.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    selected = [source for source in project.get("mediaSources", []) if not source_ids or source["id"] in source_ids]
    if not selected:
        raise ValueError("Select at least one imported photo or video source")
    digest = hashlib.sha256(json.dumps([(source["id"], source.get("originals", [])) for source in selected], sort_keys=True).encode()).hexdigest()[:24]
    root = project_root / "outputs" / "cache" / "media" / digest
    if (root / "dataset.json").is_file():
        reporter.update("registration_quality", "Reused registered camera dataset", 0.25, 1.0, 0)
        return root
    raw = root / "raw"; images_root = root / "images"; database = root / "colmap.db"; sparse = root / "sparse"
    raw.mkdir(parents=True, exist_ok=True)
    ordered = any(source["kind"] == "video" for source in selected)
    for source_index, source in enumerate(selected, start=1):
        originals = [project_root / source["path"] / value for value in source.get("originals", [])]
        if source["kind"] == "video":
            _run_phase(
                reporter,
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-hwaccel", "auto", "-i", str(originals[0]), "-vf", "fps=2", "-q:v", "2", str(raw / f"{source['id']}-%06d.jpg")],
                "extracting_media",
                f"Extracting video {source_index} of {len(selected)} with FFmpeg hardware decode",
                0.01,
                0.04,
                30,
                cancel_path,
            )
        else:
            for index, original in enumerate(originals):
                shutil.copy2(original, raw / f"{source['id']}-{index:06}{original.suffix.lower()}")
            reporter.update(
                "extracting_media",
                f"Copied photo source {source_index} of {len(selected)}",
                0.01 + 0.03 * source_index / len(selected),
                source_index / len(selected),
            )
    reporter.update("filtering_media", "Scoring sharpness and removing near-duplicate frames", 0.04, 0.0)
    accepted = _filter_images(
        raw,
        images_root,
        lambda done, total, kept: reporter.update(
            "filtering_media",
            f"Reviewed {done} of {total} frames · kept {kept}",
            0.04 + 0.03 * done / max(total, 1),
            done / max(total, 1),
        ),
    )
    if len(accepted) < 8:
        raise RuntimeError(f"Only {len(accepted)} sharp, distinct images remain; at least 8 are required")
    _run_phase(reporter, ["colmap", "feature_extractor", "--database_path", str(database), "--image_path", str(images_root), "--ImageReader.single_camera", "0", "--SiftExtraction.use_gpu", "1"], "feature_extraction", f"Extracting CUDA SIFT features from {len(accepted)} frames", 0.07, 0.11, 45, cancel_path)
    if ordered:
        _run_phase(reporter, ["colmap", "sequential_matcher", "--database_path", str(database), "--SequentialMatching.loop_detection", "1", "--SiftMatching.use_gpu", "1"], "matching_views", "Matching ordered video views on CUDA", 0.11, 0.16, 60, cancel_path)
    elif len(accepted) > 500 and os.environ.get("COLMAP_VOCAB_TREE"):
        _run_phase(reporter, ["colmap", "vocab_tree_matcher", "--database_path", str(database), "--VocabTreeMatching.vocab_tree_path", os.environ["COLMAP_VOCAB_TREE"], "--SiftMatching.use_gpu", "1"], "matching_views", "Matching large photo collection on CUDA", 0.11, 0.16, 90, cancel_path)
    else:
        _run_phase(reporter, ["colmap", "exhaustive_matcher", "--database_path", str(database), "--SiftMatching.use_gpu", "1"], "matching_views", "Matching overlapping photo views on CUDA", 0.11, 0.16, 90, cancel_path)
    sparse.mkdir(parents=True, exist_ok=True)
    _run_phase(reporter, ["colmap", "mapper", "--database_path", str(database), "--image_path", str(images_root), "--output_path", str(sparse)], "mapping_cameras", "Solving connected camera poses and sparse geometry", 0.16, 0.23, 120, cancel_path)
    components = [path for path in sparse.iterdir() if path.is_dir()]
    if not components:
        raise RuntimeError("COLMAP did not register a connected camera component")
    text_models: list[Path] = []
    for component in components:
        text = component.with_name(component.name + "-txt"); text.mkdir(exist_ok=True)
        _run(["colmap", "model_converter", "--input_path", str(component), "--output_path", str(text), "--output_type", "TXT"], cancel_path=cancel_path)
        text_models.append(text)
    parsed = [_read_colmap_model(model) for model in text_models]
    cameras, registered, points, colors, error = max(parsed, key=lambda value: len(value[1]))
    ratio = len(registered) / len(accepted)
    if len(registered) < 8 or ratio < 0.35:
        raise RuntimeError(f"COLMAP registered {len(registered)}/{len(accepted)} images ({ratio:.0%}); capture more overlap")
    frames = []
    registered_root = root / "registered"; registered_root.mkdir(exist_ok=True)
    for index, record in enumerate(registered):
        destination = registered_root / f"{index:06}.jpg"
        shutil.copy2(images_root / record["name"], destination)
        frames.append({"image": f"registered/{destination.name}", "worldFromRgbCamera": record["worldFromCamera"].reshape(-1).tolist(), "intrinsics": cameras[record["camera"]], "timestampUs": 0, "phaseId": "colmap", "metric": False})
    _write_initialization(root / "initialization.ply", points, colors)
    dataset = {"schemaVersion": 1, "fingerprint": digest, "coordinateConvention": {"handedness": "right", "units": "arbitrary", "cameraAxes": "opencv_x_right_y_down_z_forward", "pose": "worldFromCamera", "matrixStorage": "row-major"}, "metric": False, "frames": frames, "initialization": "initialization.ply", "quality": {"registeredImages": len(registered), "totalImages": len(accepted), "reprojectionError": error, "disconnectedComponents": max(0, len(components) - 1)}}
    _write_json_atomic(root / "dataset.json", dataset)
    for source in project.get("mediaSources", []):
        if source in selected:
            source["status"] = "registered"; source["imageCount"] = len(accepted); source["quality"] = dataset["quality"] | {"detail": f"{len(registered)}/{len(accepted)} images registered"}
    _write_json_atomic(project_path, project)
    reporter.update("registration_quality", f"Registered {len(registered)} of {len(accepted)} images", 0.25, 1.0, 0)
    return root
