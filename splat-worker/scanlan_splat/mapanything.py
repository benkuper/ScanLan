from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .lingbot import LingbotGeometry


MAPANYTHING_CODE_REVISION = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"
MAPANYTHING_MODEL_REPOSITORY = "facebook/map-anything-apache"
MAPANYTHING_MODEL_REVISION = "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a"
MAPANYTHING_MODEL_FILENAME = "model.safetensors"
MAPANYTHING_CONFIG_FILENAME = "config.json"
MAPANYTHING_MODEL_SHA256 = "fa06c0fdccefc5048e072c85935d5789b1e36b307f3859033c17f9dcb9fd5201"
MAPANYTHING_MAX_SEEDS = 750_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_mapanything_model(*, verify: bool = True) -> Path:
    configured = os.environ.get("SCANLAN_MAPANYTHING_MODEL")
    executable_root = Path(sys.executable).resolve().parent
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        (
            executable_root / "models" / "map-anything-apache",
            executable_root.parent / "models" / "map-anything-apache",
            Path(__file__).resolve().parent / "models" / "map-anything-apache",
            Path(__file__).resolve().parent.parent / "models" / "map-anything-apache",
        )
    )
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        candidates.append(Path(frozen_root) / "models" / "map-anything-apache")
    for candidate in candidates:
        root = candidate.parent if candidate.is_file() else candidate
        checkpoint = root / MAPANYTHING_MODEL_FILENAME
        config = root / MAPANYTHING_CONFIG_FILENAME
        if not checkpoint.is_file() or not config.is_file():
            continue
        if verify and _sha256(checkpoint) != MAPANYTHING_MODEL_SHA256:
            raise RuntimeError(
                "The installed MapAnything Apache checkpoint does not match ScanLan's pinned digest"
            )
        return root.resolve()
    raise FileNotFoundError(
        "MapAnything Apache is not installed; run npm run prepare:splat or set "
        "SCANLAN_MAPANYTHING_MODEL"
    )


@contextmanager
def _bundled_dinov2_loader(torch: Any) -> Iterator[None]:
    """Make MapAnything construction offline without changing its checkpoint.

    UniCeption calls torch.hub even when pretrained=False. MapAnything ships the
    matching DINOv2 implementation and its full checkpoint contains the encoder
    weights, so constructing that bundled backbone is sufficient and avoids a
    runtime source-code download.
    """

    original = torch.hub.load

    def load(repo_or_dir: str, model: str, *args: Any, **kwargs: Any) -> Any:
        if repo_or_dir != "facebookresearch/dinov2":
            return original(repo_or_dir, model, *args, **kwargs)
        from mapanything.models.external.dinov2.hub import backbones

        factory = getattr(backbones, model, None)
        if factory is None:
            raise RuntimeError(f"Unsupported bundled DINOv2 backbone: {model}")
        kwargs.pop("force_reload", None)
        # The full MapAnything safetensors file is loaded immediately after
        # construction and is the sole source of encoder weights.
        kwargs["pretrained"] = False
        return factory(*args, **kwargs)

    torch.hub.load = load
    try:
        yield
    finally:
        torch.hub.load = original


def _confidence_probability(value: np.ndarray) -> np.ndarray:
    confidence = np.asarray(value, dtype=np.float32)
    confidence = np.where(np.isfinite(confidence), confidence, 1.0)
    # MapAnything's exponential confidence head is bounded below by one.
    return np.clip(1.0 - np.reciprocal(np.maximum(confidence, 1.0)), 0.0, 1.0)


def _as_numpy(value: Any, *, dtype: np.dtype[Any] = np.float32) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    result = np.asarray(value)
    while result.ndim and result.shape[0] == 1:
        result = result[0]
    return result.astype(dtype, copy=False)


def _as_raster(value: Any, *, dtype: np.dtype[Any] = np.float32) -> np.ndarray:
    result = _as_numpy(value, dtype=dtype)
    if result.ndim == 3 and result.shape[-1] == 1:
        result = result[..., 0]
    if result.ndim != 2:
        raise RuntimeError(f"MapAnything returned an incompatible raster shape: {result.shape}")
    return result


def _resize_transform(
    source_size: tuple[int, int], target_size: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    scale = max(target_width / source_width, target_height / source_height) + 1e-8
    resized_width = int(np.floor(source_width * scale))
    resized_height = int(np.floor(source_height * scale))
    left = int(np.rint((resized_width - target_width) * 0.5))
    top = int(np.rint((resized_height - target_height) * 0.5))
    return (resized_width, resized_height), (left, top)


def restore_processed_raster(
    value: np.ndarray,
    valid: np.ndarray,
    source_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Undo MapAnything's cover-resize and centered crop on a dense raster."""

    import cv2

    raster = np.asarray(value, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    if raster.shape != mask.shape or raster.ndim != 2:
        raise ValueError("MapAnything raster and validity mask must be aligned 2D arrays")
    target_height, target_width = raster.shape
    resized_size, (left, top) = _resize_transform(
        source_size, (target_width, target_height)
    )
    resized_width, resized_height = resized_size
    if left < 0 or top < 0 or left + target_width > resized_width or top + target_height > resized_height:
        raise RuntimeError("MapAnything preprocessing transform is inconsistent")
    canvas = np.zeros((resized_height, resized_width), dtype=np.float32)
    canvas_mask = np.zeros((resized_height, resized_width), dtype=np.uint8)
    canvas[top : top + target_height, left : left + target_width] = np.where(
        mask, raster, 0.0
    )
    canvas_mask[top : top + target_height, left : left + target_width] = mask
    restored = cv2.resize(canvas, source_size, interpolation=cv2.INTER_LINEAR)
    restored_mask = cv2.resize(
        canvas_mask, source_size, interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    restored[~restored_mask] = 0.0
    return restored.astype(np.float32, copy=False), restored_mask


def _rotation_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
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
        other = [value for value in range(3) if value != axis]
        j, k = other
        scale = np.sqrt(max(1e-12, 1.0 + matrix[axis, axis] - matrix[j, j] - matrix[k, k])) * 2.0
        xyz = np.zeros(3, dtype=np.float64)
        xyz[axis] = 0.25 * scale
        xyz[j] = (matrix[j, axis] + matrix[axis, j]) / scale
        xyz[k] = (matrix[k, axis] + matrix[axis, k]) / scale
        value = np.asarray(
            [(matrix[k, j] - matrix[j, k]) / scale, xyz[0], xyz[1], xyz[2]]
        )
    return (value / max(float(np.linalg.norm(value)), 1e-12)).astype(np.float32)


@dataclass
class MapAnythingPredictor:
    model: Any
    torch: Any
    device: Any
    use_bfloat16: bool
    model_root: Path
    backend: str

    @classmethod
    def load(cls) -> "MapAnythingPredictor":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from mapanything.models import MapAnything

        if not torch.cuda.is_available():
            raise RuntimeError("MapAnything inference requires CUDA")
        model_root = resolve_mapanything_model(verify=True)
        with _bundled_dinov2_loader(torch):
            model = MapAnything.from_pretrained(str(model_root), local_files_only=True)
        device = torch.device("cuda")
        model = model.to(device).eval()
        use_bfloat16 = bool(torch.cuda.is_bf16_supported())
        precision = "BF16" if use_bfloat16 else "FP16"
        return cls(
            model=model,
            torch=torch,
            device=device,
            use_bfloat16=use_bfloat16,
            model_root=model_root,
            backend=(
                f"MapAnything Apache / PyTorch {precision} / "
                f"{torch.cuda.get_device_name(device)}"
            ),
        )

    def infer(self, views: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=True,
            amp_dtype="bf16" if self.use_bfloat16 else "fp16",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
            # The multi-view consistency score is useful for image-only
            # geometry, but with metric RGB-D inputs it collapses valid
            # confidence to the head's lower bound on real Femto captures.
            # ScanLan applies its stronger metric/free-space gates downstream.
            use_multiview_confidence=False,
        )

    def infer_rgbd(
        self,
        colors: Sequence[np.ndarray],
        depths_m: Sequence[np.ndarray],
        intrinsics: Sequence[np.ndarray],
        camera_poses: Sequence[np.ndarray] | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        from mapanything.utils.image import preprocess_inputs

        if not (len(colors) == len(depths_m) == len(intrinsics)) or not colors:
            raise ValueError("MapAnything RGB-D inputs must be non-empty and aligned")
        if camera_poses is not None and len(camera_poses) != len(colors):
            raise ValueError("MapAnything RGB-D camera poses are not aligned")
        views: list[dict[str, Any]] = []
        source_sizes: list[tuple[int, int]] = []
        for index, (color, depth, calibration) in enumerate(
            zip(colors, depths_m, intrinsics, strict=True)
        ):
            image = np.asarray(color, dtype=np.uint8)
            source_sizes.append((image.shape[1], image.shape[0]))
            view: dict[str, Any] = {
                "img": image,
                "depth_z": np.asarray(depth, dtype=np.float32),
                "intrinsics": np.asarray(calibration, dtype=np.float32),
                "is_metric_scale": self.torch.tensor([True]),
            }
            if camera_poses is not None:
                view["camera_poses"] = np.asarray(camera_poses[index], dtype=np.float32)
            views.append(view)
        processed = preprocess_inputs(views)
        predictions = self.infer(processed)
        outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for prediction, source_size in zip(predictions, source_sizes, strict=True):
            depth = _as_raster(prediction["depth_z"])
            mask = _as_raster(prediction["mask"], dtype=np.float32) > 0.5
            confidence = _confidence_probability(_as_raster(prediction["conf"]))
            restored_depth, restored_mask = restore_processed_raster(
                depth, mask, source_size
            )
            restored_confidence, confidence_mask = restore_processed_raster(
                confidence, mask, source_size
            )
            outputs.append(
                (
                    restored_depth,
                    restored_mask & confidence_mask,
                    restored_confidence,
                )
            )
        return outputs

    def infer_geometry(
        self,
        image_paths: Sequence[Path],
        *,
        maximum_seeds: int = MAPANYTHING_MAX_SEEDS,
    ) -> LingbotGeometry:
        from mapanything.utils.image import load_images
        from scanlan_validation import (
            CameraValidationConfig,
            GeometryValidationConfig,
            validate_camera_trajectory,
            validate_geometry,
        )

        paths = [Path(path).resolve(strict=True) for path in image_paths]
        if len(paths) < 3:
            raise ValueError("MapAnything geometry inference requires at least three images")
        views = load_images([str(path) for path in paths], verbose=False)
        predictions = self.infer(views)
        processed_height, processed_width = _as_raster(
            predictions[0]["depth_z"]
        ).shape
        poses = np.asarray(
            [_as_numpy(value["camera_poses"], dtype=np.float64) for value in predictions]
        )
        intrinsics = np.asarray(
            [_as_numpy(value["intrinsics"], dtype=np.float64) for value in predictions]
        )
        frame_confidence = np.asarray(
            [
                float(np.median(_confidence_probability(_as_raster(value["conf"]))))
                for value in predictions
            ],
            dtype=np.float32,
        )
        camera_validation = validate_camera_trajectory(
            poses,
            frame_confidence,
            CameraValidationConfig(maximum_translation_step=2.0),
        )
        if int(np.count_nonzero(camera_validation.frame_mask)) < 3:
            raise RuntimeError("MapAnything camera proposal failed the trajectory gate")
        point_chunks: list[np.ndarray] = []
        color_chunks: list[np.ndarray] = []
        confidence_chunks: list[np.ndarray] = []
        owner_chunks: list[np.ndarray] = []
        for index, prediction in enumerate(predictions):
            if not camera_validation.frame_mask[index]:
                continue
            points = _as_numpy(prediction["pts3d"])
            mask = _as_raster(prediction["mask"], dtype=np.float32) > 0.5
            confidence = _confidence_probability(_as_raster(prediction["conf"]))
            image = _as_numpy(prediction["img_no_norm"])
            if image.max(initial=0.0) <= 1.0:
                image = image * 255.0
            valid = mask & np.isfinite(points).all(axis=-1) & (confidence >= 0.35)
            selected_points = points[valid]
            selected_confidence = confidence[valid]
            geometry_validation = validate_geometry(
                selected_points,
                selected_confidence,
                config=GeometryValidationConfig(minimum_confidence=0.35),
            )
            selected_points = selected_points[geometry_validation.point_mask]
            selected_confidence = selected_confidence[geometry_validation.point_mask]
            selected_colors = np.clip(np.rint(image[valid]), 0, 255).astype(np.uint8)
            selected_colors = selected_colors[geometry_validation.point_mask]
            point_chunks.append(selected_points.astype(np.float32, copy=False))
            color_chunks.append(selected_colors)
            confidence_chunks.append(selected_confidence.astype(np.float32, copy=False))
            owner_chunks.append(np.full(len(selected_points), index, dtype=np.int32))
        if not point_chunks or not sum(map(len, point_chunks)):
            raise RuntimeError("MapAnything produced no confidence-gated geometry")
        points = np.concatenate(point_chunks)
        colors = np.concatenate(color_chunks)
        point_confidence = np.concatenate(confidence_chunks)
        owners = np.concatenate(owner_chunks)
        if len(points) > maximum_seeds:
            # Pick the strongest response from each spatially ordered bucket.
            # This preserves image coverage without retaining weak seeds merely
            # because they happened to fall on a fixed sampling stride.
            edges = np.linspace(0, len(points), maximum_seeds + 1, dtype=np.int64)
            selected = np.asarray(
                [
                    start + int(np.argmax(point_confidence[start:stop]))
                    for start, stop in zip(edges[:-1], edges[1:], strict=True)
                ],
                dtype=np.int64,
            )
            points, colors, owners = points[selected], colors[selected], owners[selected]
        focal = np.sqrt(intrinsics[owners, 0, 0] * intrinsics[owners, 1, 1])
        camera_from_world = np.linalg.inv(poses[owners])
        camera_points = np.einsum(
            "nij,nj->ni",
            camera_from_world[:, :3, :3],
            points - poses[owners, :3, 3],
        )
        footprint = np.maximum(camera_points[:, 2] / np.maximum(focal, 1.0), 1e-5)
        scales = np.repeat((footprint * 0.55)[:, None], 3, axis=1).astype(np.float32)
        camera_quaternions = np.asarray(
            [_rotation_quaternion_wxyz(pose[:3, :3]) for pose in poses],
            dtype=np.float32,
        )
        quaternions = camera_quaternions[owners]
        return LingbotGeometry(
            world_from_cameras=poses,
            intrinsics=intrinsics,
            points=points,
            colors=colors,
            scales=scales,
            quaternions=quaternions,
            source_frame_indices=owners,
            frame_confidence=np.where(
                camera_validation.frame_mask, frame_confidence, 0.0
            ).astype(np.float32),
            backend=self.backend,
            model_path=str(self.model_root),
            processed_size=(processed_width, processed_height),
        )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_rgbd_frame(
    frame: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    width = int(frame["width"])
    height = int(frame["height"])
    expected_pixels = width * height
    color_path = Path(frame["colorPath"])
    depth_path = Path(frame["depthPath"])
    color = np.fromfile(color_path, dtype=np.uint8)
    depth = np.fromfile(depth_path, dtype="<u2")
    if color.size != expected_pixels * 3:
        raise ValueError(f"Aligned RGB frame {color_path} has an unexpected size")
    if depth.size != expected_pixels:
        raise ValueError(f"Depth frame {depth_path} has an unexpected size")
    intrinsics = np.asarray(
        [
            [float(frame["fx"]), 0.0, float(frame["cx"])],
            [0.0, float(frame["fy"]), float(frame["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    pose_value = frame.get("cameraPose")
    pose = None
    if pose_value is not None:
        pose = np.asarray(pose_value, dtype=np.float32).reshape(4, 4)
        if not np.isfinite(pose).all():
            raise ValueError("MapAnything RGB-D request contains a non-finite camera pose")
    return (
        color.reshape(height, width, 3),
        depth.reshape(height, width).astype(np.float32) / float(frame["depthScale"]),
        intrinsics,
        pose,
    )


def refine_mapanything_depth_request(
    request_path: Path,
    progress_path: Path,
    *,
    predictor: MapAnythingPredictor | None = None,
) -> dict[str, Any]:
    request = json.loads(request_path.resolve().read_text(encoding="utf-8"))
    if int(request.get("schemaVersion", 0)) != 1:
        raise ValueError("Unsupported MapAnything depth request schema")
    frames = request.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("MapAnything depth request contains no frames")
    cancel_value = str(request.get("cancelPath", "")).strip()
    cancel_path = Path(cancel_value) if cancel_value else None
    if predictor is None:
        _write_json(
            progress_path,
            {
                "stage": "mapanything_loading",
                "detail": "Loading pinned MapAnything Apache checkpoint offline",
                "progress": 0.0,
            },
        )
        predictor = MapAnythingPredictor.load()
    outputs: list[dict[str, Any]] = []
    # Bounded windows retain multi-view context while preventing attention and
    # dense-output memory from growing with an entire long RGB-D capture.
    chunk_size = min(max(int(request.get("chunkSize", 8)), 2), 16)
    for first in range(0, len(frames), chunk_size):
        if cancel_path is not None and cancel_path.is_file():
            raise RuntimeError("MapAnything depth refinement cancelled")
        chunk = frames[first : first + chunk_size]
        loaded = [_load_rgbd_frame(frame) for frame in chunk]
        poses = [value[3] for value in loaded]
        pose_inputs = poses if all(value is not None for value in poses) else None
        predictions = predictor.infer_rgbd(
            [value[0] for value in loaded],
            [value[1] for value in loaded],
            [value[2] for value in loaded],
            pose_inputs,  # type: ignore[arg-type]
        )
        for offset, (frame, prediction) in enumerate(
            zip(chunk, predictions, strict=True)
        ):
            depth, mask, confidence = prediction
            prediction_path = Path(frame["predictionPath"])
            mask_path = Path(frame["modelMaskPath"])
            confidence_value = str(frame.get("confidencePath", "")).strip()
            confidence_path = Path(confidence_value) if confidence_value else None
            _save_array(prediction_path, depth.astype(np.float32, copy=False))
            _save_array(mask_path, mask.astype(np.uint8, copy=False))
            if confidence_path is not None:
                _save_array(
                    confidence_path, confidence.astype(np.float32, copy=False)
                )
            outputs.append(
                {
                    "key": str(frame["key"]),
                    "predictionPath": str(prediction_path),
                    "modelMaskPath": str(mask_path),
                    "confidencePath": (
                        str(confidence_path) if confidence_path is not None else None
                    ),
                }
            )
        completed = first + len(chunk)
        _write_json(
            progress_path,
            {
                "stage": "mapanything_depth_inference",
                "detail": f"Proposed metric depth for {completed} of {len(frames)} RGB-D keyframes",
                "progress": completed / len(frames),
                "frameCount": len(frames),
                "computeBackend": predictor.backend,
            },
        )
    result = {
        "schemaVersion": 1,
        "status": "complete",
        "backend": predictor.backend,
        "codeRevision": MAPANYTHING_CODE_REVISION,
        "modelRevision": MAPANYTHING_MODEL_REVISION,
        "modelSha256": MAPANYTHING_MODEL_SHA256,
        "frames": outputs,
    }
    _write_json(Path(request["resultPath"]), result)
    return result


def mapanything_runtime_status(
    *, verify_model: bool = False, smoke_test: bool = False
) -> dict[str, Any]:
    import importlib.util

    package_available = importlib.util.find_spec("mapanything") is not None
    model_path: str | None = None
    error: str | None = None
    runtime_validated = False
    backend: str | None = None
    try:
        model_path = str(resolve_mapanything_model(verify=verify_model))
    except (FileNotFoundError, RuntimeError) as caught:
        error = str(caught)
    if smoke_test and package_available and model_path is not None:
        try:
            predictor = MapAnythingPredictor.load()
            height, width = 56, 70
            colors = [np.full((height, width, 3), 127, dtype=np.uint8) for _ in range(2)]
            depths = [np.full((height, width), 2.0, dtype=np.float32) for _ in range(2)]
            depths[0][18:38, 24:46] = 0.0
            calibration = np.asarray(
                [[50.0, 0.0, 35.0], [0.0, 50.0, 28.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            outputs = predictor.infer_rgbd(colors, depths, [calibration, calibration])
            if len(outputs) != 2 or any(value[0].shape != (height, width) for value in outputs):
                raise RuntimeError("MapAnything smoke test returned a misaligned raster")
            if not any(np.any(mask & np.isfinite(depth) & (depth > 0.0)) for depth, mask, _ in outputs):
                raise RuntimeError("MapAnything smoke test returned no valid metric depth")
            runtime_validated = True
            backend = predictor.backend
        except Exception as caught:  # Structured diagnostics boundary.
            error = str(caught)
    return {
        "available": package_available
        and model_path is not None
        and (runtime_validated or not smoke_test),
        "packageAvailable": package_available,
        "modelPath": model_path,
        "runtimeValidated": runtime_validated,
        "backend": backend,
        "codeRevision": MAPANYTHING_CODE_REVISION,
        "modelRevision": MAPANYTHING_MODEL_REVISION,
        "modelSha256": MAPANYTHING_MODEL_SHA256,
        "license": "Apache-2.0",
        "offline": True,
        "error": error,
    }
