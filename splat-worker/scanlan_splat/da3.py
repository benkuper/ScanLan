from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scanlan_validation import (
    CameraValidationConfig,
    GeometryValidationConfig,
    validate_camera_trajectory,
    validate_geometry,
)

from .lingbot import LingbotGeometry


DA3_CODE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DA3_MODEL_REPOSITORY = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DA3_MODEL_REVISION = "b2359bdf726fb44ef62acca04d629dcf158053e7"
DA3_MODEL_SHA256 = "8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c"
DA3_CONFIG_SHA256 = "09adf89474017e717bc05aa86fd3a378708ba8914b036d61874eced328069468"
DA3_MODEL_FILENAME = "model.safetensors"
DA3_CONFIG_FILENAME = "config.json"
DA3_MAX_SEEDS = 750_000
DA3_STREAM_WINDOW = 24
DA3_STREAM_OVERLAP = 6

# The strongest refreshed Nested checkpoint supplies any-view cameras, metric
# scaling, pose conditioning, and the direct Gaussian head in one model.
DA3_GAUSSIAN_MODEL_REPOSITORY = DA3_MODEL_REPOSITORY
DA3_GAUSSIAN_MODEL_REVISION = DA3_MODEL_REVISION
DA3_GAUSSIAN_MODEL_SHA256 = DA3_MODEL_SHA256

ProgressCallback = Callable[[str, str, float, str, dict[str, Any]], None]
SH_C0 = 0.28209479177387814


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_candidates(configured: str | None, directory_name: str) -> list[Path]:
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    executable_root = Path(sys.executable).resolve().parent
    candidates.extend(
        (
            executable_root / "models" / directory_name,
            Path(__file__).resolve().parents[1] / "models" / directory_name,
        )
    )
    return candidates


def resolve_da3_model(
    *, verify: bool = True, direct_gaussians: bool = False
) -> Path:
    configured = os.environ.get("SCANLAN_DA3_MODEL")
    if direct_gaussians and not configured:
        configured = os.environ.get("SCANLAN_DA3_GAUSSIAN_MODEL")
    directory_name = "da3nested-giant-large-1.1-noncommercial"
    expected_digest = DA3_MODEL_SHA256
    for candidate in _model_candidates(configured, directory_name):
        root = candidate.expanduser().resolve()
        checkpoint = root / DA3_MODEL_FILENAME
        config = root / DA3_CONFIG_FILENAME
        if not checkpoint.is_file() or not config.is_file():
            continue
        if verify and _sha256(checkpoint) != expected_digest:
            raise RuntimeError(f"DA3 checkpoint at {checkpoint} does not match its pinned digest")
        if verify and _sha256(config) != DA3_CONFIG_SHA256:
            raise RuntimeError(f"DA3 configuration at {config} does not match its pinned digest")
        return root
    raise FileNotFoundError("Pinned DA3 Nested model assets were not found; set SCANLAN_DA3_MODEL")


def _confidence_probability(value: np.ndarray) -> np.ndarray:
    confidence = np.asarray(value, dtype=np.float32)
    confidence = np.nan_to_num(confidence, nan=1.0, posinf=1.0, neginf=1.0)
    return np.clip(1.0 - 1.0 / np.maximum(confidence, 1.0), 0.0, 1.0)


def _metric_scaled_gaussians(
    means: np.ndarray, scales: np.ndarray, scale_factor: float | None
) -> tuple[np.ndarray, np.ndarray]:
    metric_scale = 1.0 if scale_factor is None else float(scale_factor)
    if not np.isfinite(metric_scale) or metric_scale <= 0.0:
        raise RuntimeError("DA3 direct Gaussian metric scale is invalid")
    return (
        np.asarray(means, dtype=np.float32) * metric_scale,
        np.asarray(scales, dtype=np.float32) * metric_scale,
    )


def _direct_gaussian_mask(
    confidence_maps: np.ndarray,
    opacities: np.ndarray,
) -> np.ndarray:
    """Apply only representation-safe filtering to DA3's direct splats.

    Learned opacity is part of the direct renderer, not a quality score: on
    real outdoor captures, high-opacity background splats can be farther than
    foreground surfaces. Keep that signal intact and mirror upstream's
    resolution-relative border crop, while rejecting malformed values.
    """
    confidence = np.asarray(confidence_maps, dtype=np.float32)
    opacity = np.asarray(opacities, dtype=np.float32).reshape(-1)
    if confidence.ndim != 3 or opacity.shape != (confidence.size,):
        raise RuntimeError("DA3 direct Gaussian maps have inconsistent shapes")
    views, height, width = confidence.shape
    border_y = max(1, int(round(8.0 * height / 256.0)))
    border_x = max(1, int(round(8.0 * width / 256.0)))
    interior = np.ones((views, height, width), dtype=bool)
    interior[:, :border_y, :] = False
    interior[:, -border_y:, :] = False
    interior[:, :, :border_x] = False
    interior[:, :, -border_x:] = False
    return (
        interior.reshape(-1)
        & np.isfinite(confidence).reshape(-1)
        & np.isfinite(opacity)
        & (opacity >= 0.0)
        & (opacity <= 1.0)
    )


def _as_homogeneous_extrinsics(value: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(value, dtype=np.float64)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] not in ((3, 4), (4, 4)):
        raise RuntimeError("DA3 returned an invalid camera extrinsic array")
    if extrinsics.shape[1:] == (3, 4):
        padded = np.repeat(np.eye(4, dtype=np.float64)[None], len(extrinsics), axis=0)
        padded[:, :3, :] = extrinsics
        extrinsics = padded
    return extrinsics


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
        j, k = [index for index in range(3) if index != axis]
        scale = np.sqrt(max(1e-12, 1.0 + matrix[axis, axis] - matrix[j, j] - matrix[k, k])) * 2.0
        xyz = np.zeros(3, dtype=np.float64)
        xyz[axis] = 0.25 * scale
        xyz[j] = (matrix[j, axis] + matrix[axis, j]) / scale
        xyz[k] = (matrix[k, axis] + matrix[axis, k]) / scale
        value = np.asarray([(matrix[k, j] - matrix[j, k]) / scale, *xyz])
    return (value / max(float(np.linalg.norm(value)), 1e-12)).astype(np.float32)


def _multiply_quaternions_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    result = np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )
    return (result / np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-12)).astype(np.float32)


def _unproject(
    depth: np.ndarray, intrinsics: np.ndarray, world_from_camera: np.ndarray
) -> np.ndarray:
    height, width = depth.shape
    rows, columns = np.indices((height, width), dtype=np.float32)
    x = (columns - float(intrinsics[0, 2])) * depth / float(intrinsics[0, 0])
    y = (rows - float(intrinsics[1, 2])) * depth / float(intrinsics[1, 1])
    camera = np.stack((x, y, depth), axis=-1)
    return (
        camera @ np.asarray(world_from_camera[:3, :3], dtype=np.float32).T
        + np.asarray(world_from_camera[:3, 3], dtype=np.float32)
    )


def _bounded_indices(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    owners: np.ndarray,
    maximum_seeds: int,
) -> np.ndarray:
    if len(points) <= maximum_seeds:
        return np.arange(len(points), dtype=np.int64)
    edges = np.linspace(0, len(points), maximum_seeds + 1, dtype=np.int64)
    selected = np.asarray(
        [start + int(np.argmax(confidence[start:stop])) for start, stop in zip(edges[:-1], edges[1:], strict=True)],
        dtype=np.int64,
    )
    return selected


@dataclass
class Da3Predictor:
    model: Any
    torch: Any
    device: Any
    dtype: Any
    model_root: Path
    backend: str
    direct_gaussians: bool = False

    @classmethod
    def load(cls, *, direct_gaussians: bool = False) -> "Da3Predictor":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch

        # DA3 has one eager-compatible helper decorated with torch.jit.script.
        # PyInstaller stores Python modules in a compressed archive, so
        # TorchScript cannot retrieve that helper's source during import. Keep
        # JIT in development, but import the frozen package with that decorator
        # acting as identity and restore Torch immediately afterward.
        original_script = torch.jit.script
        if getattr(sys, "frozen", False):
            torch.jit.script = lambda function, *args, **kwargs: function
        try:
            from depth_anything_3.api import DepthAnything3
        finally:
            torch.jit.script = original_script

        if not torch.cuda.is_available():
            raise RuntimeError("DA3 inference requires CUDA")
        model_root = resolve_da3_model(verify=True, direct_gaussians=direct_gaussians)
        model = DepthAnything3.from_pretrained(str(model_root), local_files_only=True)
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # Keep checkpoint weights in FP32. Upstream intentionally disables
        # autocast around its camera, depth, and GS heads, while applying mixed
        # precision only to the regions it has marked safe.
        model = model.to(device=device).eval()
        precision = "BF16" if dtype == torch.bfloat16 else "FP16"
        model_name = "DA3NESTED-GIANT-LARGE-1.1"
        return cls(
            model=model,
            torch=torch,
            device=device,
            dtype=dtype,
            model_root=model_root,
            backend=(
                f"{model_name} / FP32 weights + {precision} safe autocast / "
                f"{torch.cuda.get_device_name(device)}"
            ),
            direct_gaussians=direct_gaussians,
        )

    def _predict(
        self,
        image_paths: Sequence[Path] | Sequence[np.ndarray],
        *,
        world_from_cameras: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        infer_gaussians: bool = False,
    ) -> Any:
        extrinsics = None
        if world_from_cameras is not None:
            extrinsics = np.linalg.inv(np.asarray(world_from_cameras, dtype=np.float64)).astype(np.float32)
        with self.torch.inference_mode(), self.torch.autocast(
            device_type="cuda", dtype=self.dtype
        ):
            return self.model.inference(
                [str(path) if isinstance(path, Path) else path for path in image_paths],
                extrinsics=extrinsics,
                intrinsics=(
                    None
                    if intrinsics is None
                    else np.asarray(intrinsics, dtype=np.float32)
                ),
                align_to_input_ext_scale=True,
                infer_gs=infer_gaussians,
                use_ray_pose=False,
                ref_view_strategy=(
                    "middle" if len(image_paths) > 8 else "saddle_balanced"
                ),
                process_res=504,
                process_res_method="upper_bound_resize",
            )

    def infer_pose_conditioned_depth(
        self,
        colors: Sequence[np.ndarray],
        intrinsics: Sequence[np.ndarray],
        world_from_cameras: Sequence[np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if not (len(colors) == len(intrinsics) == len(world_from_cameras)) or not colors:
            raise ValueError("DA3 pose-conditioned inputs must be non-empty and aligned")
        prediction = self._predict(
            colors,
            world_from_cameras=np.asarray(world_from_cameras),
            intrinsics=np.asarray(intrinsics),
        )
        depths = np.asarray(prediction.depth, dtype=np.float32)
        confidences = _confidence_probability(np.asarray(prediction.conf, dtype=np.float32))
        outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        import cv2

        for color, depth, confidence in zip(colors, depths, confidences, strict=True):
            height, width = np.asarray(color).shape[:2]
            if depth.shape != (height, width):
                depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
                confidence = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)
            valid = np.isfinite(depth) & (depth > 0.0) & np.isfinite(confidence)
            outputs.append((depth.astype(np.float32), valid, confidence.astype(np.float32)))
        return outputs

    def infer_geometry_window(
        self,
        image_paths: Sequence[Path],
        *,
        maximum_seeds: int = DA3_MAX_SEEDS,
        world_from_cameras: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        infer_gaussians: bool = False,
    ) -> LingbotGeometry:
        paths = [Path(path).resolve(strict=True) for path in image_paths]
        if len(paths) < 3:
            raise ValueError("DA3 geometry inference requires at least three images")
        if infer_gaussians and not self.direct_gaussians:
            raise RuntimeError("DA3 direct Gaussian inference was not enabled for this worker instance")
        prediction = self._predict(
            paths,
            world_from_cameras=world_from_cameras,
            intrinsics=intrinsics,
            infer_gaussians=infer_gaussians,
        )
        if prediction.extrinsics is None or prediction.intrinsics is None or prediction.conf is None:
            raise RuntimeError("DA3 omitted camera or confidence outputs")
        camera_from_world = _as_homogeneous_extrinsics(prediction.extrinsics)
        poses = np.linalg.inv(camera_from_world)
        calibrations = np.asarray(prediction.intrinsics, dtype=np.float64)
        frame_confidence = np.asarray(
            [float(np.median(value)) for value in _confidence_probability(prediction.conf)],
            dtype=np.float32,
        )
        camera_validation = validate_camera_trajectory(
            poses,
            frame_confidence,
            # A DA3 window is an any-view set: adjacent indices may be
            # unordered photos or widely spaced video keyframes. Validate each
            # rigid/confident camera here; bounded-window overlap and COLMAP
            # agreement provide the cross-camera coherence gates downstream.
            CameraValidationConfig(
                maximum_rotation_step_degrees=180.0,
                maximum_translation_step=None,
                adaptive_translation_limit=False,
            ),
        )
        if int(np.count_nonzero(camera_validation.frame_mask)) < 3:
            raise RuntimeError(
                "DA3 camera proposal failed the any-view camera gate: "
                + json.dumps(camera_validation.to_dict(), allow_nan=False)
            )

        opacities: np.ndarray | None = None
        if infer_gaussians and prediction.gaussians is not None:
            gaussian = prediction.gaussians
            # Nested DA3 1.1 applies its metric scale to depth and camera
            # translations after the GS adapter has already emitted world-space
            # Gaussians. Apply the published scale factor here so means/scales
            # remain in the same coordinate system as the returned cameras.
            points, scales = _metric_scaled_gaussians(
                gaussian.means.squeeze(0).detach().cpu(),
                gaussian.scales.squeeze(0).detach().cpu(),
                prediction.scale_factor,
            )
            quaternions = np.asarray(gaussian.rotations.squeeze(0).detach().cpu(), dtype=np.float32)
            harmonics = np.asarray(gaussian.harmonics.squeeze(0).detach().cpu(), dtype=np.float32)
            opacities = np.asarray(
                gaussian.opacities.squeeze(0).detach().cpu(), dtype=np.float32
            )
            colors = np.clip(np.rint((harmonics[:, :, 0] * SH_C0 + 0.5) * 255.0), 0, 255).astype(np.uint8)
            owners = np.repeat(np.arange(len(paths), dtype=np.int32), len(points) // len(paths))
            if len(owners) != len(points):
                raise RuntimeError("DA3 Gaussian ownership does not match its input views")
            point_confidence = np.repeat(frame_confidence, len(points) // len(paths))
            geometry_validation = validate_geometry(
                points,
                point_confidence,
                config=GeometryValidationConfig(minimum_confidence=0.35),
            )
            keep = geometry_validation.point_mask & _direct_gaussian_mask(
                _confidence_probability(prediction.conf), opacities
            )
            if (
                not np.any(keep)
                or not np.isfinite(scales).all()
                or not np.isfinite(quaternions).all()
                or np.any(scales <= 0.0)
            ):
                raise RuntimeError("DA3 direct Gaussian proposal failed the geometry contract")
            points, colors = points[keep], colors[keep]
            scales, quaternions = scales[keep], quaternions[keep]
            opacities = opacities[keep]
            point_confidence, owners = point_confidence[keep], owners[keep]
        else:
            images = np.asarray(prediction.processed_images, dtype=np.uint8)
            depths = np.asarray(prediction.depth, dtype=np.float32)
            confidence_maps = _confidence_probability(prediction.conf)
            point_chunks: list[np.ndarray] = []
            color_chunks: list[np.ndarray] = []
            confidence_chunks: list[np.ndarray] = []
            owner_chunks: list[np.ndarray] = []
            for index, (image, depth, confidence) in enumerate(zip(images, depths, confidence_maps, strict=True)):
                if not camera_validation.frame_mask[index]:
                    continue
                points_map = _unproject(depth, calibrations[index], poses[index])
                valid = np.isfinite(points_map).all(axis=-1) & np.isfinite(depth) & (depth > 0.0) & (confidence >= 0.35)
                selected_points = points_map[valid]
                selected_confidence = confidence[valid]
                geometry_validation = validate_geometry(
                    selected_points,
                    selected_confidence,
                    config=GeometryValidationConfig(minimum_confidence=0.35),
                )
                keep = geometry_validation.point_mask
                point_chunks.append(selected_points[keep].astype(np.float32, copy=False))
                color_chunks.append(image[valid][keep].astype(np.uint8, copy=False))
                confidence_chunks.append(selected_confidence[keep].astype(np.float32, copy=False))
                owner_chunks.append(np.full(int(np.count_nonzero(keep)), index, dtype=np.int32))
            if not point_chunks or not sum(map(len, point_chunks)):
                raise RuntimeError("DA3 produced no confidence-gated geometry")
            points = np.concatenate(point_chunks)
            colors = np.concatenate(color_chunks)
            point_confidence = np.concatenate(confidence_chunks)
            owners = np.concatenate(owner_chunks)
            focal = np.sqrt(calibrations[owners, 0, 0] * calibrations[owners, 1, 1])
            camera_points = np.einsum(
                "nij,nj->ni",
                camera_from_world[owners, :3, :3],
                points - poses[owners, :3, 3],
            )
            footprint = np.maximum(camera_points[:, 2] / np.maximum(focal, 1.0), 1e-5)
            scales = np.repeat((footprint * 0.55)[:, None], 3, axis=1).astype(np.float32)
            camera_quaternions = np.asarray(
                [_rotation_quaternion_wxyz(pose[:3, :3]) for pose in poses], dtype=np.float32
            )
            quaternions = camera_quaternions[owners]
        selected = _bounded_indices(points, colors, point_confidence, owners, maximum_seeds)
        points, colors = points[selected], colors[selected]
        point_confidence, owners = point_confidence[selected], owners[selected]
        scales, quaternions = scales[selected], quaternions[selected]
        if opacities is not None:
            opacities = opacities[selected]
        return LingbotGeometry(
            world_from_cameras=poses,
            intrinsics=calibrations,
            points=points,
            colors=colors,
            scales=scales,
            quaternions=quaternions,
            source_frame_indices=owners,
            frame_confidence=np.where(camera_validation.frame_mask, frame_confidence, 0.0).astype(np.float32),
            backend=self.backend + (" / direct Gaussian proposal" if infer_gaussians else ""),
            model_path=str(self.model_root),
            processed_size=(int(prediction.depth.shape[2]), int(prediction.depth.shape[1])),
            opacities=opacities,
        )


def _similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    left, singular, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right))
    rotation = left @ correction @ right
    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    scale = float(np.sum(singular * np.diag(correction)) / max(variance, 1e-12))
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _pose_similarity(
    source_poses: np.ndarray, target_poses: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit Sim(3) from camera centers plus oriented basis points.

    Camera centers alone become ill-conditioned on the nearly straight paths
    common in handheld video. Small basis points retain the overlapping
    cameras' measured orientation without allowing them to dominate scale.
    """
    source_poses = np.asarray(source_poses, dtype=np.float64)
    target_poses = np.asarray(target_poses, dtype=np.float64)
    source_centers = source_poses[:, :3, 3]
    target_centers = target_poses[:, :3, 3]
    source_steps = np.linalg.norm(np.diff(source_centers, axis=0), axis=1)
    target_steps = np.linalg.norm(np.diff(target_centers, axis=0), axis=1)
    source_extent = max(float(np.median(source_steps)), 0.05)
    target_extent = max(float(np.median(target_steps)), 0.05)
    source_samples = [source_centers]
    target_samples = [target_centers]
    for axis in range(3):
        source_samples.append(
            source_centers + source_extent * source_poses[:, :3, axis]
        )
        target_samples.append(
            target_centers + target_extent * target_poses[:, :3, axis]
        )
    return _similarity(np.concatenate(source_samples), np.concatenate(target_samples))


def _transform_geometry(
    geometry: LingbotGeometry, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> LingbotGeometry:
    poses = np.asarray(geometry.world_from_cameras, dtype=np.float64).copy()
    poses[:, :3, :3] = np.einsum("ij,njk->nik", rotation, poses[:, :3, :3])
    poses[:, :3, 3] = scale * (poses[:, :3, 3] @ rotation.T) + translation
    points = scale * (np.asarray(geometry.points, dtype=np.float64) @ rotation.T) + translation
    transform_quaternion = _rotation_quaternion_wxyz(rotation)
    return LingbotGeometry(
        world_from_cameras=poses,
        intrinsics=geometry.intrinsics,
        points=points.astype(np.float32),
        colors=geometry.colors,
        scales=(geometry.scales * scale).astype(np.float32),
        quaternions=_multiply_quaternions_wxyz(transform_quaternion, geometry.quaternions),
        source_frame_indices=geometry.source_frame_indices,
        frame_confidence=geometry.frame_confidence,
        backend=geometry.backend,
        model_path=geometry.model_path,
        processed_size=geometry.processed_size,
        opacities=geometry.opacities,
    )


def _restrict_geometry(geometry: LingbotGeometry, indices: Sequence[int]) -> LingbotGeometry:
    selected = np.asarray(indices, dtype=np.int64)
    remap = np.full(len(geometry.world_from_cameras), -1, dtype=np.int64)
    remap[selected] = np.arange(len(selected))
    keep = remap[geometry.source_frame_indices] >= 0
    return LingbotGeometry(
        world_from_cameras=geometry.world_from_cameras[selected],
        intrinsics=geometry.intrinsics[selected],
        points=geometry.points[keep],
        colors=geometry.colors[keep],
        scales=geometry.scales[keep],
        quaternions=geometry.quaternions[keep],
        source_frame_indices=remap[geometry.source_frame_indices[keep]].astype(np.int32),
        frame_confidence=geometry.frame_confidence[selected],
        backend=geometry.backend,
        model_path=geometry.model_path,
        processed_size=geometry.processed_size,
        opacities=(geometry.opacities[keep] if geometry.opacities is not None else None),
    )


def infer_da3_geometry_streaming(
    predictor: Da3Predictor,
    image_paths: Sequence[Path],
    *,
    maximum_seeds: int = DA3_MAX_SEEDS,
    window_size: int = DA3_STREAM_WINDOW,
    overlap: int = DA3_STREAM_OVERLAP,
    output_indices: Sequence[int] | None = None,
    progress: ProgressCallback | None = None,
    cancelled: Callable[[], bool] | None = None,
    infer_gaussians: bool = False,
) -> tuple[LingbotGeometry, dict[str, Any]]:
    paths = [Path(path).resolve(strict=True) for path in image_paths]
    if len(paths) < 3:
        raise ValueError("DA3 streaming geometry requires at least three images")
    window_size = min(max(int(window_size), 8), 32)
    overlap = min(max(int(overlap), 3), window_size // 2)
    starts = list(range(0, max(1, len(paths) - overlap), window_size - overlap))
    if starts[-1] + overlap >= len(paths) and len(starts) > 1:
        starts.pop()
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(paths), axis=0)
    calibrations = np.repeat(np.eye(3, dtype=np.float64)[None], len(paths), axis=0)
    frame_confidence = np.zeros(len(paths), dtype=np.float32)
    point_groups: list[np.ndarray] = []
    color_groups: list[np.ndarray] = []
    scale_groups: list[np.ndarray] = []
    quaternion_groups: list[np.ndarray] = []
    owner_groups: list[np.ndarray] = []
    opacity_groups: list[np.ndarray] = []
    overlap_residuals: list[float] = []
    overlap_rotation_degrees: list[float] = []
    completed = 0
    for window_index, start in enumerate(starts):
        if cancelled is not None and cancelled():
            raise RuntimeError("DA3 geometry inference cancelled")
        stop = min(start + window_size, len(paths))
        if stop - start < 3:
            continue
        if progress:
            progress(
                "da3_streaming",
                f"Evaluating DA3 window {window_index + 1} of {len(starts)}",
                window_index / len(starts),
                predictor.backend,
                {"window": window_index + 1, "windowCount": len(starts), "boundedFrames": stop - start},
            )
        local = predictor.infer_geometry_window(
            paths[start:stop],
            maximum_seeds=max(50_000, maximum_seeds // max(1, len(starts))),
            infer_gaussians=infer_gaussians,
        )
        if window_index:
            shared = min(overlap, stop - start, completed - start)
            if shared < 3:
                raise RuntimeError("DA3 streaming window lost its required overlap")
            scale, rotation, translation = _pose_similarity(
                local.world_from_cameras[:shared],
                poses[start : start + shared],
            )
            local = _transform_geometry(local, scale, rotation, translation)
            residual = np.linalg.norm(
                local.world_from_cameras[:shared, :3, 3] - poses[start : start + shared, :3, 3],
                axis=1,
            )
            baseline = np.linalg.norm(
                poses[start : start + shared, :3, 3] - poses[start, :3, 3], axis=1
            )
            normalized = float(np.median(residual) / max(float(np.median(baseline)), 1e-3))
            overlap_residuals.append(normalized)
            relative_rotations = np.einsum(
                "nij,nkj->nik",
                local.world_from_cameras[:shared, :3, :3],
                poses[start : start + shared, :3, :3],
            )
            cosines = np.clip(
                (np.trace(relative_rotations, axis1=1, axis2=2) - 1.0) * 0.5,
                -1.0,
                1.0,
            )
            rotation_degrees = float(np.degrees(np.median(np.arccos(cosines))))
            overlap_rotation_degrees.append(rotation_degrees)
            if (
                not np.isfinite(normalized)
                or normalized > 0.12
                or not np.isfinite(rotation_degrees)
                or rotation_degrees > 8.0
            ):
                raise RuntimeError(
                    "DA3 streaming window failed overlap alignment "
                    f"({normalized:.3f} normalized residual, {rotation_degrees:.2f} degrees)"
                )
            new_camera_start = shared
        else:
            new_camera_start = 0
        global_indices = np.arange(start, stop)
        poses[global_indices[new_camera_start:]] = local.world_from_cameras[new_camera_start:]
        calibrations[global_indices[new_camera_start:]] = local.intrinsics[new_camera_start:]
        frame_confidence[global_indices] = np.maximum(
            frame_confidence[global_indices], local.frame_confidence
        )
        keep = local.source_frame_indices >= new_camera_start
        point_groups.append(local.points[keep])
        color_groups.append(local.colors[keep])
        scale_groups.append(local.scales[keep])
        quaternion_groups.append(local.quaternions[keep])
        owner_groups.append((local.source_frame_indices[keep] + start).astype(np.int32))
        if local.opacities is not None:
            opacity_groups.append(local.opacities[keep])
        completed = max(completed, stop)
        if stop == len(paths):
            break
    if completed != len(paths):
        raise RuntimeError("DA3 streaming inference did not cover every input frame")
    geometry = LingbotGeometry(
        world_from_cameras=poses,
        intrinsics=calibrations,
        points=np.concatenate(point_groups),
        colors=np.concatenate(color_groups),
        scales=np.concatenate(scale_groups),
        quaternions=np.concatenate(quaternion_groups),
        source_frame_indices=np.concatenate(owner_groups),
        frame_confidence=frame_confidence,
        backend=predictor.backend + " / bounded overlap streaming",
        model_path=str(predictor.model_root),
        processed_size=local.processed_size,
        opacities=(np.concatenate(opacity_groups) if opacity_groups else None),
    )
    if len(geometry.points) > maximum_seeds:
        selected = np.linspace(0, len(geometry.points) - 1, maximum_seeds, dtype=np.int64)
        geometry = LingbotGeometry(
            world_from_cameras=geometry.world_from_cameras,
            intrinsics=geometry.intrinsics,
            points=geometry.points[selected],
            colors=geometry.colors[selected],
            scales=geometry.scales[selected],
            quaternions=geometry.quaternions[selected],
            source_frame_indices=geometry.source_frame_indices[selected],
            frame_confidence=geometry.frame_confidence,
            backend=geometry.backend,
            model_path=geometry.model_path,
            processed_size=geometry.processed_size,
            opacities=(
                geometry.opacities[selected]
                if geometry.opacities is not None
                else None
            ),
        )
    if output_indices is not None:
        geometry = _restrict_geometry(geometry, output_indices)
    telemetry = {
        "mode": "bounded_overlap_streaming" if len(starts) > 1 else "single_window",
        "windowSize": window_size,
        "overlap": overlap,
        "windowCount": len(starts),
        "maximumOverlapResidual": max(overlap_residuals, default=0.0),
        "maximumOverlapRotationDegrees": max(overlap_rotation_degrees, default=0.0),
        "directGaussians": infer_gaussians,
    }
    return geometry, telemetry


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def _load_rgbd_frame(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width, height = int(frame["width"]), int(frame["height"])
    color = np.fromfile(Path(frame["colorPath"]), dtype=np.uint8)
    if color.size != width * height * 3:
        raise ValueError("DA3 aligned RGB frame has an unexpected size")
    calibration = np.asarray(
        [[float(frame["fx"]), 0.0, float(frame["cx"])], [0.0, float(frame["fy"]), float(frame["cy"])], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pose = np.asarray(frame["cameraPose"], dtype=np.float32).reshape(4, 4)
    if not np.isfinite(pose).all():
        raise ValueError("DA3 depth request contains a non-finite camera pose")
    return color.reshape(height, width, 3), calibration, pose


def refine_da3_depth_request(
    request_path: Path,
    progress_path: Path,
    *,
    predictor: Da3Predictor | None = None,
) -> dict[str, Any]:
    request = json.loads(request_path.resolve().read_text(encoding="utf-8"))
    if int(request.get("schemaVersion", 0)) != 1:
        raise ValueError("Unsupported DA3 depth request schema")
    frames = request.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("DA3 depth request requires at least two posed frames")
    cancel_value = str(request.get("cancelPath", "")).strip()
    cancel_path = Path(cancel_value) if cancel_value else None
    predictor = predictor or Da3Predictor.load()
    outputs: list[dict[str, Any]] = []
    chunk_size = min(max(int(request.get("chunkSize", 8)), 2), 16)
    for first in range(0, len(frames), chunk_size):
        if cancel_path is not None and cancel_path.is_file():
            raise RuntimeError("DA3 pose-conditioned depth inference cancelled")
        chunk = frames[first : first + chunk_size]
        loaded = [_load_rgbd_frame(frame) for frame in chunk]
        predictions = predictor.infer_pose_conditioned_depth(
            [value[0] for value in loaded],
            [value[1] for value in loaded],
            [value[2] for value in loaded],
        )
        for frame, (depth, mask, confidence) in zip(chunk, predictions, strict=True):
            prediction_path = Path(frame["predictionPath"])
            mask_path = Path(frame["modelMaskPath"])
            _save_array(prediction_path, depth.astype(np.float32, copy=False))
            _save_array(mask_path, mask.astype(np.uint8, copy=False))
            outputs.append(
                {"key": str(frame["key"]), "predictionPath": str(prediction_path), "modelMaskPath": str(mask_path)}
            )
        completed = first + len(chunk)
        _write_json(
            progress_path,
            {
                "schemaVersion": 1,
                "stage": "da3_pose_conditioned_depth",
                "detail": f"Proposed pose-conditioned depth for {completed} of {len(frames)} RGB-D keyframes",
                "progress": completed / len(frames),
                "computeBackend": predictor.backend,
            },
        )
    result = {
        "schemaVersion": 1,
        "status": "complete",
        "backend": predictor.backend,
        "codeRevision": DA3_CODE_REVISION,
        "modelRevision": DA3_MODEL_REVISION,
        "modelSha256": DA3_MODEL_SHA256,
        "configSha256": DA3_CONFIG_SHA256,
        "frames": outputs,
    }
    _write_json(Path(request["resultPath"]), result)
    return result


def da3_runtime_status(*, verify_model: bool = False, smoke_test: bool = False) -> dict[str, Any]:
    import importlib.util

    package_available = importlib.util.find_spec("depth_anything_3") is not None
    model_path: str | None = None
    error: str | None = None
    runtime_validated = False
    backend: str | None = None
    peak_memory_mib: float | None = None
    direct_gaussians_validated = False
    try:
        model_path = str(resolve_da3_model(verify=verify_model))
    except (FileNotFoundError, RuntimeError) as caught:
        error = str(caught)
    if smoke_test and package_available and model_path is not None:
        try:
            predictor = Da3Predictor.load()
            predictor.torch.cuda.empty_cache()
            predictor.torch.cuda.reset_peak_memory_stats(predictor.device)
            height, width = 56, 70
            colors = [np.full((height, width, 3), 127 + index, dtype=np.uint8) for index in range(3)]
            intrinsics = [np.asarray([[50.0, 0.0, 35.0], [0.0, 50.0, 28.0], [0.0, 0.0, 1.0]], dtype=np.float32)] * 3
            poses = [np.eye(4, dtype=np.float32) for _ in range(3)]
            poses[1][0, 3] = 0.05
            poses[2][1, 3] = 0.05
            outputs = predictor.infer_pose_conditioned_depth(colors, intrinsics, poses)
            if len(outputs) != 3 or any(value[0].shape != (height, width) for value in outputs):
                raise RuntimeError("DA3 smoke test returned a misaligned raster")
            if not all(np.any(mask & (depth > 0.0)) for depth, mask, _ in outputs):
                raise RuntimeError("DA3 smoke test returned no valid depth")
            gaussian_prediction = predictor._predict(colors, infer_gaussians=True)
            gaussian = gaussian_prediction.gaussians
            if gaussian is None:
                raise RuntimeError("DA3 Nested smoke test omitted direct Gaussians")
            for name in ("means", "scales", "rotations", "harmonics", "opacities"):
                value = getattr(gaussian, name, None)
                if value is None or not bool(predictor.torch.isfinite(value).all()):
                    raise RuntimeError(f"DA3 direct Gaussian smoke test returned invalid {name}")
            direct_gaussians_validated = True
            peak_memory_mib = float(
                predictor.torch.cuda.max_memory_allocated(predictor.device) / (1024 * 1024)
            )
            if peak_memory_mib > 11_264.0:
                raise RuntimeError(
                    f"DA3 peak CUDA allocation {peak_memory_mib:.0f} MiB leaves unsafe headroom on 12 GB"
                )
            runtime_validated = True
            backend = predictor.backend
        except Exception as caught:  # Structured diagnostics boundary.
            error = str(caught)
    gaussian_available = False
    gaussian_error: str | None = None
    try:
        resolve_da3_model(verify=verify_model, direct_gaussians=True)
        gaussian_available = package_available
    except (FileNotFoundError, RuntimeError) as caught:
        gaussian_error = str(caught)
    return {
        "available": package_available and model_path is not None and (runtime_validated or not smoke_test),
        "packageAvailable": package_available,
        "modelPath": model_path,
        "runtimeValidated": runtime_validated,
        "backend": backend,
        "codeRevision": DA3_CODE_REVISION,
        "modelRevision": DA3_MODEL_REVISION,
        "modelSha256": DA3_MODEL_SHA256,
        "configSha256": DA3_CONFIG_SHA256,
        "license": "CC-BY-NC-4.0",
        "offline": True,
        "streaming": {"bounded": True, "windowSize": DA3_STREAM_WINDOW, "overlap": DA3_STREAM_OVERLAP},
        "peakCudaMemoryMiB": peak_memory_mib,
        "referenceMemoryLimitMiB": 11_264,
        "directGaussians": {
            "available": gaussian_available and (direct_gaussians_validated or not smoke_test),
            "runtimeValidated": direct_gaussians_validated,
            "bundled": model_path is not None,
            "license": "CC-BY-NC-4.0",
            "modelRevision": DA3_GAUSSIAN_MODEL_REVISION,
            "modelSha256": DA3_GAUSSIAN_MODEL_SHA256,
            "error": gaussian_error,
        },
        "error": error,
    }
