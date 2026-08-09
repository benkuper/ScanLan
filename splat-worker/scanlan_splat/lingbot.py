from __future__ import annotations

import gc
import math
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


LINGBOT_CODE_REVISION = "1f480aeb8a47a24656090d46d053115b7fe60435"
LINGBOT_MODEL_REPOSITORY = "robbyant/lingbot-map"
LINGBOT_MODEL_REVISION = "204754b72bb24f561f8d7e7e1e4e4cd9e809adf9"
LINGBOT_MODEL_FILENAME = "lingbot-map-long.pt"
LINGBOT_SKY_MODEL_FILENAME = "skyseg_batch.onnx"
LINGBOT_IMAGE_SIZE = 518
LINGBOT_PATCH_SIZE = 14
LINGBOT_SCALE_FRAMES = 8
LINGBOT_MAX_SEEDS = 750_000
FLASHINFER_CACHE_FILENAME = "flashinfer-cache.zip"


@dataclass(frozen=True)
class LingbotGeometry:
    world_from_cameras: np.ndarray
    intrinsics: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    scales: np.ndarray
    quaternions: np.ndarray
    source_frame_indices: np.ndarray
    frame_confidence: np.ndarray
    backend: str
    model_path: str
    processed_size: tuple[int, int]


ProgressCallback = Callable[[str, str, float, str, dict[str, Any]], None]


def lingbot_processed_size(width: int, height: int) -> tuple[int, int]:
    """Mirror LingBot's crop-mode resize without loading an image."""
    if width <= 0 or height <= 0:
        raise ValueError("LingBot input dimensions must be positive")
    processed_width = LINGBOT_IMAGE_SIZE
    resized_height = (
        round(height * processed_width / width / LINGBOT_PATCH_SIZE)
        * LINGBOT_PATCH_SIZE
    )
    return processed_width, min(max(LINGBOT_PATCH_SIZE, resized_height), LINGBOT_IMAGE_SIZE)


def lingbot_source_pixel_grid(width: int, height: int) -> np.ndarray:
    """Map processed pixel centers back to the source image's pixel plane."""
    processed_width, processed_height = lingbot_processed_size(width, height)
    resized_height = (
        round(height * processed_width / width / LINGBOT_PATCH_SIZE)
        * LINGBOT_PATCH_SIZE
    )
    resized_height = max(LINGBOT_PATCH_SIZE, resized_height)
    crop_top = max(0, (resized_height - processed_height) // 2)
    u = (np.arange(processed_width, dtype=np.float64) + 0.5) * (
        width / processed_width
    ) - 0.5
    v = (
        (np.arange(processed_height, dtype=np.float64) + crop_top + 0.5)
        * (height / resized_height)
        - 0.5
    )
    grid_u, grid_v = np.meshgrid(u, v)
    return np.stack((grid_u, grid_v), axis=-1)


def _asset_candidates(filename: str, environment_name: str) -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get(environment_name)
    if configured:
        candidates.append(Path(configured).expanduser())
    executable_root = Path(sys.executable).resolve().parent
    candidates.extend(
        (
            executable_root / "models" / filename,
            executable_root.parent / "models" / filename,
            Path(__file__).resolve().parent / "models" / filename,
            Path(__file__).resolve().parent.parent / "models" / filename,
        )
    )
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        candidates.append(Path(frozen_root) / "models" / filename)
    return candidates


def resolve_lingbot_asset(
    filename: str,
    environment_name: str,
    *,
    allow_download: bool = True,
) -> Path:
    for candidate in _asset_candidates(filename, environment_name):
        if candidate.is_file():
            return candidate.resolve()
    if not allow_download:
        raise FileNotFoundError(
            f"{filename} is not installed; run npm run prepare:splat or set {environment_name}"
        )
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "LingBot-Map model download support is not installed; run npm run prepare:splat"
        ) from error
    return Path(
        hf_hub_download(
            repo_id=LINGBOT_MODEL_REPOSITORY,
            filename=filename,
            revision=LINGBOT_MODEL_REVISION,
        )
    ).resolve()


def _restore_bundled_flashinfer_cache() -> None:
    """Restore ScanLan's prewarmed Windows kernels to FlashInfer's short path."""
    archive = next(
        (
            candidate
            for candidate in _asset_candidates(
                FLASHINFER_CACHE_FILENAME,
                "SCANLAN_FLASHINFER_CACHE_ARCHIVE",
            )
            if candidate.is_file()
        ),
        None,
    )
    if archive is None:
        return
    system_drive = os.environ.get("SystemDrive", "C:")
    cache_root = Path(system_drive) / "_fij"
    marker = cache_root / f".scanlan-{archive.stat().st_size}-{int(archive.stat().st_mtime)}"
    if marker.is_file():
        return
    cache_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if any(
            Path(member.filename).is_absolute()
            or ".." in Path(member.filename).parts
            for member in members
        ):
            raise RuntimeError("Bundled FlashInfer cache contains an unsafe path")
        bundle.extractall(cache_root)
    marker.touch()


def _configure_flashinfer_runtime(torch: Any) -> None:
    """Expose the bundled Ninja runner and the exact GPU architecture."""
    _restore_bundled_flashinfer_cache()
    os.environ.setdefault(
        "FLASHINFER_CUDA_ARCH_LIST",
        ".".join(map(str, torch.cuda.get_device_capability(0))),
    )
    if shutil.which("ninja") is None:
        try:
            import ninja

            ninja_directory = str(ninja.BIN_DIR)
            os.environ["PATH"] = (
                ninja_directory
                + os.pathsep
                + os.environ.get("PATH", "")
            )
        except (ImportError, AttributeError):
            pass
    bundled_aot = Path(os.environ.get("SystemDrive", "C:")) / "_fij" / "flashinfer-aot"
    if bundled_aot.is_dir():
        try:
            from flashinfer.jit import env as flashinfer_jit_environment

            flashinfer_jit_environment.FLASHINFER_AOT_DIR = bundled_aot
        except (ImportError, OSError):
            pass


def warm_lingbot_flashinfer() -> dict[str, Any]:
    """Compile and execute the exact paged-attention shape used by 16:9 video."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("FlashInfer warmup requires CUDA")
    _configure_flashinfer_runtime(torch)
    from lingbot_map.layers.flashinfer_cache import FlashInferKVCacheManager

    tokens_per_frame = (518 // LINGBOT_PATCH_SIZE) * (294 // LINGBOT_PATCH_SIZE) + 6
    manager = FlashInferKVCacheManager(
        num_blocks=1,
        max_num_frames=88,
        tokens_per_frame=tokens_per_frame,
        num_heads=16,
        head_dim=64,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        scale_frames=LINGBOT_SCALE_FRAMES,
        sliding_window=64,
        max_total_frames=420,
    )
    query = torch.randn(
        tokens_per_frame,
        16,
        64,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    manager.append_frame(0, key, value)
    output = manager.compute_attention(0, query)
    torch.cuda.synchronize()
    if output.shape != query.shape or not bool(torch.isfinite(output).all()):
        raise RuntimeError("Windows FlashInfer warmup produced invalid attention output")
    return {
        "available": True,
        "tokensPerFrame": tokens_per_frame,
        "shape": list(output.shape),
        "cudaCapability": ".".join(map(str, torch.cuda.get_device_capability(0))),
    }


def _streaming_configuration(
    torch: Any,
    frame_count: int,
    *,
    use_sdpa: bool,
) -> tuple[int, int, int]:
    """Choose bounded context without exhausting a laptop GPU."""
    vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram_gib <= 8.5:
        sliding_window = 16
    elif vram_gib <= 12.5:
        sliding_window = 32
    elif vram_gib <= 16.5:
        sliding_window = 64
    else:
        sliding_window = 96
    # FlashInfer's paged cache and sliding-window eviction keep the heavy patch
    # KV bounded, so preserve the full 320-view context the checkpoint was
    # trained with. Native SDPA retains substantially more memory and needs a
    # stricter laptop cap. Every frame still receives depth and a pose either
    # way; the interval controls only which frames persist as context.
    maximum_cached_views = 96 if use_sdpa and vram_gib <= 16.5 else 320
    keyframe_interval = max(
        1,
        math.ceil(max(0, frame_count - LINGBOT_SCALE_FRAMES) / maximum_cached_views),
    )
    max_frame_num = max(320, frame_count + 16)
    return sliding_window, keyframe_interval, max_frame_num


def lingbot_runtime_status(
    *,
    allow_download: bool = False,
    validate_flashinfer: bool = False,
) -> dict[str, Any]:
    import importlib.util

    package_available = importlib.util.find_spec("lingbot_map") is not None
    flashinfer_available = importlib.util.find_spec("flashinfer") is not None
    model_path: str | None = None
    model_error: str | None = None
    flashinfer_validated = False
    flashinfer_error: str | None = None
    flashinfer_smoke: dict[str, Any] | None = None
    if package_available:
        try:
            model_path = str(
                resolve_lingbot_asset(
                    LINGBOT_MODEL_FILENAME,
                    "SCANLAN_LINGBOT_MODEL",
                    allow_download=allow_download,
                )
            )
        except Exception as error:  # diagnostics must remain structured
            model_error = str(error)
    if validate_flashinfer:
        if not flashinfer_available:
            flashinfer_error = "Windows FlashInfer is not installed"
        else:
            try:
                flashinfer_smoke = warm_lingbot_flashinfer()
                flashinfer_validated = bool(flashinfer_smoke.get("available"))
            except Exception as error:  # diagnostics must remain structured
                flashinfer_error = str(error)
    return {
        "available": package_available and model_path is not None,
        "package": package_available,
        "model": model_path,
        "modelError": model_error,
        "flashinfer": flashinfer_available,
        "flashinferValidated": flashinfer_validated,
        "flashinferError": flashinfer_error,
        "flashinferSmoke": flashinfer_smoke,
        "attentionBackend": "Windows FlashInfer" if flashinfer_available else "PyTorch SDPA",
        "codeRevision": LINGBOT_CODE_REVISION,
        "modelRevision": LINGBOT_MODEL_REVISION,
    }


def _rotation_matrices_to_quaternions(rotations: np.ndarray) -> np.ndarray:
    """Convert N x 3 x 3 rotations to normalized wxyz quaternions."""
    matrices = np.asarray(rotations, dtype=np.float64)
    count = len(matrices)
    result = np.empty((count, 4), dtype=np.float64)
    trace = np.trace(matrices, axis1=1, axis2=2)
    positive = trace > 0.0
    if np.any(positive):
        scale = np.sqrt(trace[positive] + 1.0) * 2.0
        selected = matrices[positive]
        result[positive, 0] = 0.25 * scale
        result[positive, 1] = (selected[:, 2, 1] - selected[:, 1, 2]) / scale
        result[positive, 2] = (selected[:, 0, 2] - selected[:, 2, 0]) / scale
        result[positive, 3] = (selected[:, 1, 0] - selected[:, 0, 1]) / scale
    remaining = np.flatnonzero(~positive)
    for index in remaining:
        matrix = matrices[index]
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(max(1e-12, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            result[index] = (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            )
        elif axis == 1:
            scale = math.sqrt(max(1e-12, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
            result[index] = (
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(max(1e-12, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
            result[index] = (
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
    return result.astype(np.float32)


def _sky_confidence(images: np.ndarray) -> np.ndarray | None:
    """Return a soft non-sky mask, using CUDA ONNX when the runtime exposes it."""
    try:
        import cv2
        import onnxruntime

        model_path = resolve_lingbot_asset(
            LINGBOT_SKY_MODEL_FILENAME,
            "SCANLAN_LINGBOT_SKY_MODEL",
        )
    except Exception:
        return None
    available = set(onnxruntime.get_available_providers())
    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available
    ]
    session = onnxruntime.InferenceSession(str(model_path), providers=providers or None)
    input_meta = session.get_inputs()[0]
    output_name = session.get_outputs()[0].name
    height = int(input_meta.shape[-2]) if isinstance(input_meta.shape[-2], int) else 320
    width = int(input_meta.shape[-1]) if isinstance(input_meta.shape[-1], int) else 320
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    masks: list[np.ndarray] = []
    for start in range(0, len(images), 16):
        batch_images = images[start : start + 16]
        batch = np.stack(
            [
                ((cv2.resize(image, (width, height)).astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)
                for image in batch_images
            ]
        ).astype(np.float32)
        raw = np.asarray(session.run([output_name], {input_meta.name: batch})[0])
        while raw.ndim > 3 and raw.shape[0] == 1:
            raw = raw[0]
        if raw.ndim == 4:
            raw = raw[:, -1]
        if raw.ndim == 2:
            raw = raw[None]
        for score, image in zip(raw, batch_images, strict=True):
            low = float(np.min(score))
            high = float(np.max(score))
            normalized = (score - low) / max(high - low, 1e-8)
            sky = cv2.resize(
                normalized.astype(np.float32),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            masks.append(np.clip(1.0 - sky, 0.0, 1.0))
    return np.asarray(masks, dtype=np.float32)


def _surface_seeds(
    depths: np.ndarray,
    confidences: np.ndarray,
    images: np.ndarray,
    intrinsics: np.ndarray,
    world_from_cameras: np.ndarray,
    maximum_seeds: int,
    normalized_rays: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_count, height, width = depths.shape
    rays = None if normalized_rays is None else np.asarray(normalized_rays, dtype=np.float64)
    if rays is not None and rays.shape != (height, width, 2):
        raise ValueError(
            "Calibrated camera rays must match LingBot's processed image dimensions"
        )
    stride = max(1, int(math.ceil(math.sqrt(frame_count * height * width / max(maximum_seeds * 1.7, 1)))))
    sky = _sky_confidence(images)
    seed_points: list[np.ndarray] = []
    seed_colors: list[np.ndarray] = []
    seed_scales: list[np.ndarray] = []
    seed_rotations: list[np.ndarray] = []
    seed_frame_indices: list[np.ndarray] = []
    frame_scores = np.zeros(frame_count, dtype=np.float32)
    camera_from_world = np.linalg.inv(np.asarray(world_from_cameras, dtype=np.float64))
    confidence_thresholds = np.full(frame_count, 1.5, dtype=np.float64)
    for confidence_index in range(frame_count):
        candidate_confidence = np.asarray(confidences[confidence_index], dtype=np.float64)
        finite_confidence = candidate_confidence[np.isfinite(candidate_confidence)]
        if finite_confidence.size:
            confidence_thresholds[confidence_index] = max(
                1.5,
                float(np.percentile(finite_confidence, 50.0)),
            )

    for frame_index in range(frame_count):
        depth = np.asarray(depths[frame_index], dtype=np.float64)
        confidence = np.asarray(confidences[frame_index], dtype=np.float64)
        finite = np.isfinite(depth) & np.isfinite(confidence) & (depth > 1e-6)
        if not np.any(finite):
            continue
        positive_confidence = confidence[finite]
        # LingBot's own viewer defaults to discarding the lower half of depth
        # confidence. Dense 3DGS is even less tolerant of bad translucent
        # layers, so use the same conservative floor before surface seeding.
        threshold = confidence_thresholds[frame_index]
        valid = finite & (confidence >= threshold)
        # Depth discontinuities are exactly where monocular unprojection creates
        # long, camera-facing shards. Require a locally continuous 4-neighbour
        # surface before estimating an oriented Gaussian footprint.
        neighbour_finite = np.zeros_like(finite)
        neighbour_finite[1:-1, 1:-1] = (
            finite[1:-1, :-2]
            & finite[1:-1, 2:]
            & finite[:-2, 1:-1]
            & finite[2:, 1:-1]
        )
        neighbour_delta = np.full_like(depth, np.inf, dtype=np.float64)
        neighbour_delta[1:-1, 1:-1] = np.maximum.reduce(
            (
                np.abs(depth[1:-1, 1:-1] - depth[1:-1, :-2]),
                np.abs(depth[1:-1, 1:-1] - depth[1:-1, 2:]),
                np.abs(depth[1:-1, 1:-1] - depth[:-2, 1:-1]),
                np.abs(depth[1:-1, 1:-1] - depth[2:, 1:-1]),
            )
        )
        valid &= neighbour_finite
        valid &= neighbour_delta <= np.maximum(np.abs(depth) * 0.12, 1e-4)
        finite_depth = depth[valid]
        if finite_depth.size:
            depth_high = float(np.percentile(finite_depth, 99.7))
            valid &= depth <= depth_high
        if sky is not None:
            valid &= sky[frame_index] >= 0.20
        sampled = np.zeros_like(valid)
        sampled[stride // 2 :: stride, stride // 2 :: stride] = True
        valid &= sampled
        valid[[0, -1], :] = False
        valid[:, [0, -1]] = False
        v, u = np.nonzero(valid)
        if not len(u):
            continue

        k = intrinsics[frame_index]
        z = depth[v, u]
        if rays is None:
            x = (u - k[0, 2]) * z / k[0, 0]
            y = (v - k[1, 2]) * z / k[1, 1]
        else:
            x = rays[v, u, 0] * z
            y = rays[v, u, 1] * z
        camera_points = np.column_stack((x, y, z))

        def unproject(sample_u: np.ndarray, sample_v: np.ndarray) -> np.ndarray:
            sample_z = depth[sample_v, sample_u]
            if rays is not None:
                sample_rays = rays[sample_v, sample_u]
                return np.column_stack(
                    (
                        sample_rays[:, 0] * sample_z,
                        sample_rays[:, 1] * sample_z,
                        sample_z,
                    )
                )
            return np.column_stack(
                (
                    (sample_u - k[0, 2]) * sample_z / k[0, 0],
                    (sample_v - k[1, 2]) * sample_z / k[1, 1],
                    sample_z,
                )
            )

        left = unproject(u - 1, v)
        right = unproject(u + 1, v)
        up = unproject(u, v - 1)
        down = unproject(u, v + 1)
        tangent_x = right - left
        tangent_y = down - up
        tangent_x /= np.maximum(np.linalg.norm(tangent_x, axis=1, keepdims=True), 1e-9)
        normal = np.cross(tangent_x, tangent_y)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        facing_away = np.sum(normal * camera_points, axis=1) > 0.0
        normal[facing_away] *= -1.0
        tangent_y = np.cross(normal, tangent_x)
        tangent_y /= np.maximum(np.linalg.norm(tangent_y, axis=1, keepdims=True), 1e-9)

        pose = world_from_cameras[frame_index]
        rotation = pose[:3, :3]
        world_points = camera_points @ rotation.T + pose[:3, 3]

        # A single monocular depth map can contain a plausible-looking but
        # geometrically false surface. Those layers are catastrophic as 3DGS
        # seeds: training stretches them into translucent needles. Keep a seed
        # only when at least one nearby view predicts the same world surface.
        # This is deliberately done before concatenation so unsupported layers
        # never reach the optimizer.
        if frame_count > 1:
            supported = np.zeros(len(world_points), dtype=bool)
            for offset in (1, -1, 2, -2, 4, -4):
                neighbour_index = frame_index + offset
                if neighbour_index < 0 or neighbour_index >= frame_count:
                    continue
                neighbour_pose = camera_from_world[neighbour_index]
                neighbour_points = (
                    world_points @ neighbour_pose[:3, :3].T
                    + neighbour_pose[:3, 3]
                )
                projected_z = neighbour_points[:, 2]
                neighbour_k = intrinsics[neighbour_index]
                projected_u = np.rint(
                    neighbour_k[0, 0] * neighbour_points[:, 0]
                    / np.maximum(projected_z, 1e-9)
                    + neighbour_k[0, 2]
                ).astype(np.int64)
                projected_v = np.rint(
                    neighbour_k[1, 1] * neighbour_points[:, 1]
                    / np.maximum(projected_z, 1e-9)
                    + neighbour_k[1, 2]
                ).astype(np.int64)
                visible = (
                    (projected_z > 1e-6)
                    & (projected_u >= 1)
                    & (projected_u < width - 1)
                    & (projected_v >= 1)
                    & (projected_v < height - 1)
                )
                if not np.any(visible):
                    continue
                visible_indices = np.flatnonzero(visible)
                sample_u = projected_u[visible]
                sample_v = projected_v[visible]
                neighbour_depth = depths[neighbour_index, sample_v, sample_u]
                neighbour_confidence = confidences[neighbour_index, sample_v, sample_u]
                comparable = (
                    np.isfinite(neighbour_depth)
                    & (neighbour_depth > 1e-6)
                    & np.isfinite(neighbour_confidence)
                    & (neighbour_confidence >= confidence_thresholds[neighbour_index])
                )
                relative_error = np.abs(neighbour_depth - projected_z[visible]) / np.maximum(
                    np.minimum(neighbour_depth, projected_z[visible]),
                    1e-6,
                )
                supported[visible_indices[comparable & (relative_error <= 0.08)]] = True
            if not np.any(supported):
                continue
            v, u = v[supported], u[supported]
            z = z[supported]
            camera_points = camera_points[supported]
            left, right = left[supported], right[supported]
            up, down = up[supported], down[supported]
            tangent_x, tangent_y = tangent_x[supported], tangent_y[supported]
            normal = normal[supported]
            world_points = world_points[supported]

        world_x = tangent_x @ rotation.T
        world_y = tangent_y @ rotation.T
        world_normal = normal @ rotation.T
        rotations = np.stack((world_x, world_y, world_normal), axis=2)
        if rays is None:
            nominal_x = np.abs(z / k[0, 0]) * stride
            nominal_y = np.abs(z / k[1, 1]) * stride
        else:
            nominal_x = (
                np.linalg.norm(rays[v, u + 1] - rays[v, u - 1], axis=1)
                * np.abs(z)
                * 0.5
                * stride
            )
            nominal_y = (
                np.linalg.norm(rays[v + 1, u] - rays[v - 1, u], axis=1)
                * np.abs(z)
                * 0.5
                * stride
            )
        scale_x = np.maximum(np.linalg.norm(right - left, axis=1) * 0.5 * stride, nominal_x)
        scale_y = np.maximum(np.linalg.norm(down - up, axis=1) * 0.5 * stride, nominal_y)
        scale_z = np.maximum(
            np.minimum(scale_x, scale_y) * 0.08,
            np.maximum(scale_x, scale_y) * 0.015,
        )

        seed_points.append(world_points.astype(np.float32))
        seed_colors.append(images[frame_index, v, u].astype(np.uint8))
        seed_scales.append(np.column_stack((scale_x, scale_y, scale_z)).astype(np.float32))
        seed_rotations.append(rotations.astype(np.float32))
        seed_frame_indices.append(
            np.full(len(world_points), frame_index, dtype=np.int32)
        )
        frame_scores[frame_index] = float(np.median(confidence[v, u]))

    if not seed_points:
        raise RuntimeError("LingBot-Map produced no confidence-gated dense geometry")
    points = np.concatenate(seed_points)
    colors = np.concatenate(seed_colors)
    scales = np.concatenate(seed_scales)
    rotations = np.concatenate(seed_rotations)
    source_frame_indices = np.concatenate(seed_frame_indices)
    finite = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(scales).all(axis=1)
        & np.isfinite(rotations).all(axis=(1, 2))
        & (scales > 0.0).all(axis=1)
    )
    points, colors, scales, rotations, source_frame_indices = (
        points[finite],
        colors[finite],
        scales[finite],
        rotations[finite],
        source_frame_indices[finite],
    )
    if len(points) > maximum_seeds:
        indices = np.linspace(0, len(points) - 1, maximum_seeds, dtype=np.int64)
        points, colors, scales, rotations, source_frame_indices = (
            points[indices],
            colors[indices],
            scales[indices],
            rotations[indices],
            source_frame_indices[indices],
        )
    valid_scores = frame_scores[frame_scores > 0.0]
    if len(valid_scores):
        low, high = np.percentile(valid_scores, (10.0, 90.0))
        frame_scores = np.clip((frame_scores - low) / max(high - low, 1e-6), 0.0, 1.0)
        frame_scores = 0.55 + 0.40 * frame_scores
    else:
        frame_scores.fill(0.7)
    return (
        points,
        colors,
        scales,
        _rotation_matrices_to_quaternions(rotations),
        source_frame_indices,
        frame_scores,
    )


def infer_lingbot_geometry(
    image_paths: Sequence[Path],
    *,
    maximum_seeds: int = LINGBOT_MAX_SEEDS,
    normalized_rays: np.ndarray | None = None,
    output_indices: Sequence[int] | None = None,
    progress: ProgressCallback | None = None,
) -> LingbotGeometry:
    import torch

    if torch.cuda.is_available():
        _configure_flashinfer_runtime(torch)
    from lingbot_map.models.gct_stream import GCTStream
    from lingbot_map.utils.geometry import closed_form_inverse_se3_general
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    if not torch.cuda.is_available():
        raise RuntimeError("LingBot-Map requires CUDA")
    paths = [str(Path(path).resolve()) for path in image_paths]
    if len(paths) < 3:
        raise ValueError("LingBot-Map requires at least three ordered views")
    selected_indices = np.asarray(
        list(range(len(paths))) if output_indices is None else output_indices,
        dtype=np.int64,
    )
    if (
        selected_indices.ndim != 1
        or not len(selected_indices)
        or np.any(selected_indices < 0)
        or np.any(selected_indices >= len(paths))
        or np.any(np.diff(selected_indices) <= 0)
    ):
        raise ValueError("LingBot output indices must be unique, increasing, and in range")
    model_path = resolve_lingbot_asset(
        LINGBOT_MODEL_FILENAME,
        "SCANLAN_LINGBOT_MODEL",
    )
    try:
        import flashinfer  # noqa: F401

        use_sdpa = False
        attention_backend = "Windows FlashInfer paged attention"
    except (ImportError, OSError):
        use_sdpa = True
        attention_backend = "PyTorch SDPA fallback"
    backend = f"LingBot-Map long / {attention_backend}"
    if progress:
        progress(
            "lingbot_loading",
            f"Loading LingBot-Map long checkpoint for {len(paths):,} ordered context frames",
            0.09,
            backend,
            {
                "contextFrameCount": len(paths),
                "trainingViewCount": len(selected_indices),
                "flashinfer": not use_sdpa,
            },
        )
    images = load_and_preprocess_images(
        paths,
        mode="crop",
        image_size=LINGBOT_IMAGE_SIZE,
        patch_size=LINGBOT_PATCH_SIZE,
    )
    height, width = int(images.shape[-2]), int(images.shape[-1])
    if normalized_rays is not None and np.asarray(normalized_rays).shape != (height, width, 2):
        raise ValueError(
            "Calibrated camera rays do not match LingBot's actual processed image size"
        )
    device = torch.device("cuda")
    sliding_window, keyframe_interval, max_frame_num = _streaming_configuration(
        torch,
        len(paths),
        use_sdpa=use_sdpa,
    )
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16

    def load_model(sdpa: bool) -> tuple[Any, list[str], list[str]]:
        loaded = GCTStream(
            img_size=LINGBOT_IMAGE_SIZE,
            patch_size=LINGBOT_PATCH_SIZE,
            enable_3d_rope=True,
            max_frame_num=max_frame_num,
            kv_cache_sliding_window=sliding_window,
            kv_cache_scale_frames=LINGBOT_SCALE_FRAMES,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=sdpa,
            camera_num_iterations=4,
        )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        missing_keys, unexpected_keys = loaded.load_state_dict(state, strict=False)
        del state, checkpoint
        loaded = loaded.to(device).eval()
        loaded.aggregator = loaded.aggregator.to(dtype=dtype)
        return loaded, list(missing_keys), list(unexpected_keys)

    model, missing, unexpected = load_model(use_sdpa)
    if progress:
        progress(
            "lingbot_inference",
            f"Estimating a drift-resistant trajectory from {len(paths):,} continuous video frames",
            0.10,
            backend,
            {
                "contextFrameCount": len(paths),
                "trainingViewCount": len(selected_indices),
                "keyframeInterval": keyframe_interval,
                "kvCacheSlidingWindow": sliding_window,
                "maximumFramePositions": max_frame_num,
                "cameraRefinementIterations": 4,
                "processedWidth": width,
                "processedHeight": height,
            },
        )
    torch.cuda.empty_cache()
    def run_inference(loaded: Any, interval: int) -> dict[str, Any]:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            return loaded.inference_streaming(
                images,
                num_scale_frames=min(LINGBOT_SCALE_FRAMES, len(paths)),
                keyframe_interval=interval,
                output_device=torch.device("cpu"),
            )

    try:
        predictions = run_inference(model, keyframe_interval)
    except (ImportError, OSError, RuntimeError) as flashinfer_error:
        if use_sdpa:
            raise
        # A Windows FlashInfer package can import successfully while a JIT
        # kernel still fails for the installed Torch/CUDA/GPU tuple. Rebuild the
        # model with native attention and rerun instead of losing the dataset.
        del model
        gc.collect()
        torch.cuda.empty_cache()
        use_sdpa = True
        attention_backend = "PyTorch SDPA fallback"
        backend = f"LingBot-Map long / {attention_backend}"
        sliding_window, keyframe_interval, max_frame_num = _streaming_configuration(
            torch,
            len(paths),
            use_sdpa=True,
        )
        if progress:
            progress(
                "lingbot_inference",
                "Windows FlashInfer was unavailable at kernel launch; retrying with PyTorch SDPA",
                0.10,
                backend,
                {
                    "contextFrameCount": len(paths),
                    "trainingViewCount": len(selected_indices),
                    "flashinfer": False,
                    "fallbackReason": str(flashinfer_error),
                },
            )
        model, missing, unexpected = load_model(True)
        predictions = run_inference(model, keyframe_interval)
    required_outputs = {"pose_enc", "depth", "depth_conf"}
    absent_outputs = required_outputs.difference(predictions)
    if absent_outputs:
        raise RuntimeError(
            "LingBot checkpoint did not produce required outputs: "
            + ", ".join(sorted(absent_outputs))
        )
    pose_encoding = predictions["pose_enc"]
    camera_from_world, intrinsics = pose_encoding_to_extri_intri(
        pose_encoding,
        (height, width),
    )
    camera_from_world_4x4 = torch.zeros(
        (*camera_from_world.shape[:-2], 4, 4),
        dtype=camera_from_world.dtype,
        device=camera_from_world.device,
    )
    camera_from_world_4x4[..., :3, :4] = camera_from_world
    camera_from_world_4x4[..., 3, 3] = 1.0
    world_from_cameras = closed_form_inverse_se3_general(camera_from_world_4x4)
    depths = predictions["depth"].float().cpu().numpy()
    confidences = predictions["depth_conf"].float().cpu().numpy()
    while depths.ndim > 4 and depths.shape[0] == 1:
        depths = depths[0]
    if depths.shape[-1] == 1:
        depths = depths[..., 0]
    while confidences.ndim > 3 and confidences.shape[0] == 1:
        confidences = confidences[0]
    world_from_cameras_np = world_from_cameras.float().cpu().numpy()
    intrinsics_np = intrinsics.float().cpu().numpy()
    if world_from_cameras_np.ndim == 4:
        world_from_cameras_np = world_from_cameras_np[0]
    if intrinsics_np.ndim == 4:
        intrinsics_np = intrinsics_np[0]
    image_values = predictions.get("images", images)
    image_values = image_values.float().cpu().numpy()
    while image_values.ndim > 4 and image_values.shape[0] == 1:
        image_values = image_values[0]
    image_values = np.rint(np.clip(np.transpose(image_values, (0, 2, 3, 1)), 0.0, 1.0) * 255.0).astype(np.uint8)
    depths = depths[selected_indices]
    confidences = confidences[selected_indices]
    world_from_cameras_np = world_from_cameras_np[selected_indices]
    intrinsics_np = intrinsics_np[selected_indices]
    image_values = image_values[selected_indices]
    del predictions, pose_encoding, camera_from_world, camera_from_world_4x4, intrinsics
    del model, images
    gc.collect()
    torch.cuda.empty_cache()
    if progress:
        progress(
            "lingbot_geometry",
            "Filtering dense geometry by depth confidence, surface continuity, and sky segmentation",
            0.20,
            backend,
            {
                "contextFrameCount": len(paths),
                "trainingViewCount": len(selected_indices),
                "maximumSeedCount": maximum_seeds,
            },
        )
    (
        points,
        colors,
        scales,
        quaternions,
        source_frame_indices,
        frame_confidence,
    ) = _surface_seeds(
        depths,
        confidences,
        image_values,
        intrinsics_np,
        world_from_cameras_np,
        maximum_seeds,
        normalized_rays=normalized_rays,
    )
    return LingbotGeometry(
        world_from_cameras=world_from_cameras_np.astype(np.float64),
        intrinsics=intrinsics_np.astype(np.float64),
        points=points,
        colors=colors,
        scales=scales,
        quaternions=quaternions,
        source_frame_indices=source_frame_indices,
        frame_confidence=frame_confidence,
        backend=backend,
        model_path=str(model_path),
        processed_size=(width, height),
    )
