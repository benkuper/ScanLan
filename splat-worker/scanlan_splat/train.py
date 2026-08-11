from __future__ import annotations

import importlib.metadata
import json
import math
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .dataset import load_dataset
from .depth_loss import depth_weight, masked_robust_depth_loss
from .appearance import (
    compose_material_render_state,
    linear_to_srgb_tensor,
    material_regularization,
    resolve_gaussian_material_seeds,
    srgb_to_linear_tensor,
)
from .export import (
    SH_C0,
    export_3dgs_ply,
    export_material_gaussians,
    export_splat_preview,
    write_splat_sidecars,
)
from .initialization import resolve_initialization_contract
from .pose import (
    constrain_pose_offsets_,
    pose_correction_statistics,
    pose_delta_matrix,
    pose_regularization,
)


FRAME_REUSE_PER_LOAD = 4
LOSS_EMA_ALPHA = 0.005
RGBD_GAUSSIAN_MULTIPLIER = 3
TRAINER_VERSION = "material-aware-components-source-resolution-v11"
RGBD_SURFACE_SCALE_MULTIPLIER = 1.3
RGBD_SURFACE_OPACITY = 0.45
DENSE_PRIOR_INITIAL_OPACITY = 0.01
MAX_METRIC_ITERATIONS = 2_000
MINIMUM_SOURCE_RESOLUTION_FRACTION = 0.20
MINIMUM_PRODUCTION_PSNR_DB = 18.0
MINIMUM_PRODUCTION_SSIM = 0.55
MAXIMUM_PRODUCTION_L1 = 0.15


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(40):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(0.025)
    finally:
        temporary.unlink(missing_ok=True)


def _write_checkpoint_atomic(path: Path, value: Any, save: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        save(value, temporary)
        # torch.save closes its own handle before returning. Flush the completed
        # archive through the OS before publishing it so a reported resumable
        # job never points at a rename that outran the checkpoint data.
        # Windows requires a writable file handle for FlushFileBuffers, which
        # is what os.fsync() uses beneath Python's file descriptor wrapper.
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        for attempt in range(40):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(0.025)
    finally:
        temporary.unlink(missing_ok=True)


def _load_checkpoint_or_quarantine(path: Path, load: Any) -> tuple[Any | None, Path | None]:
    try:
        return load(path), None
    except Exception:
        quarantine = path.with_name(f"splat-checkpoint.corrupt-{time.time_ns()}.pt")
        os.replace(path, quarantine)
        return None, quarantine


def _read_initialization(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        lines: list[str] = []
        while True:
            line = handle.readline().decode("ascii").strip()
            lines.append(line)
            if line == "end_header":
                break
        count = int(next(line.split()[2] for line in lines if line.startswith("element vertex ")))
        properties = [line.split()[2] for line in lines if line.startswith("property ")]
        formats = {"float": "<f4", "float32": "<f4", "uchar": "u1", "uint8": "u1"}
        types = [formats[line.split()[1]] for line in lines if line.startswith("property ")]
        vertices = np.fromfile(handle, dtype=np.dtype(list(zip(properties, types, strict=True))), count=count)
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32)
    if all(name in vertices.dtype.names for name in ("red", "green", "blue")):
        colors = np.column_stack((vertices["red"], vertices["green"], vertices["blue"])).astype(np.float32) / 255.0
    else:
        colors = np.full_like(points, 0.5)
    if len(points) > 500_000:
        indices = np.linspace(0, len(points) - 1, 500_000, dtype=np.int64)
        points, colors = points[indices], colors[indices]
    return points, colors


def _read_seed_parameters(
    root: Path,
    dataset: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    points, colors = _read_initialization(
        root / dataset.get("initialization", "initialization.ply")
    )
    parameters_path = dataset.get("initializationParameters")
    if not parameters_path:
        return points, colors, None, None, None, None
    path = root / parameters_path
    if not path.is_file():
        return points, colors, None, None, None, None
    with np.load(path, allow_pickle=False) as values:
        seeded_points = np.asarray(values["points"], dtype=np.float32)
        seeded_colors = np.asarray(values["colors"], dtype=np.float32)
        seeded_scales = np.asarray(values["scales"], dtype=np.float32)
        seeded_quaternions = np.asarray(values["quaternions"], dtype=np.float32)
        seeded_confidence = (
            np.asarray(values["confidence"], dtype=np.float32)
            if "confidence" in values
            else np.ones(len(seeded_points), dtype=np.float32)
        )
        seeded_opacity = (
            np.asarray(values["opacity"], dtype=np.float32)
            if "opacity" in values
            else None
        )
    if seeded_colors.size and seeded_colors.max() > 1.0:
        seeded_colors /= 255.0
    count = len(seeded_points)
    if (
        seeded_points.shape != (count, 3)
        or seeded_colors.shape != (count, 3)
        or seeded_scales.shape != (count, 3)
        or seeded_quaternions.shape != (count, 4)
        or seeded_confidence.shape != (count,)
        or (seeded_opacity is not None and seeded_opacity.shape != (count,))
        or not all(
            np.isfinite(value).all()
            for value in (
                seeded_points,
                seeded_colors,
                seeded_scales,
                seeded_quaternions,
                seeded_confidence,
                *(() if seeded_opacity is None else (seeded_opacity,)),
            )
        )
        or np.any(seeded_scales <= 0.0)
        or np.any((seeded_confidence < 0.0) | (seeded_confidence > 1.0))
        or (
            seeded_opacity is not None
            and np.any((seeded_opacity < 0.0) | (seeded_opacity > 1.0))
        )
    ):
        raise ValueError("RGB-D Gaussian initialization parameters are invalid")
    return (
        seeded_points,
        seeded_colors,
        seeded_scales,
        seeded_quaternions,
        seeded_confidence,
        seeded_opacity,
    )


def _sh_preview_colors(sh0: Any) -> Any:
    import torch

    return torch.clamp(sh0[:, 0, :] * SH_C0 + 0.5, 0.0, 1.0)


def _inverse_sigmoid_numpy(value: np.ndarray) -> np.ndarray:
    bounded = np.clip(np.asarray(value, dtype=np.float32), 1e-4, 1.0 - 1e-4)
    return np.log(bounded / (1.0 - bounded)).astype(np.float32)


def _material_parameter_initialization(
    colors_srgb: np.ndarray,
    material_seeds: Any | None,
) -> dict[str, np.ndarray]:
    from scanlan_material.radiometry import srgb_to_linear

    count = len(colors_srgb)
    observed_linear = srgb_to_linear(np.clip(colors_srgb, 0.0, 1.0)).astype(np.float32)
    if material_seeds is None:
        confidence = np.zeros(count, dtype=np.float32)
        # Preserve the established display-space trainer exactly when no
        # intrinsic prior is declared. A neutral branch must be a true no-op.
        diffuse = np.asarray(colors_srgb, dtype=np.float32)
        roughness = np.ones(count, dtype=np.float32)
        metallic = np.zeros(count, dtype=np.float32)
        transmission = np.full(count, 1e-4, dtype=np.float32)
        emission = np.full((count, 3), math.exp(-16.0), dtype=np.float32)
    else:
        confidence = np.asarray(material_seeds.confidence, dtype=np.float32)
        diffuse = (
            confidence[:, None] * material_seeds.albedo_linear
            + (1.0 - confidence[:, None]) * observed_linear
        ).astype(np.float32)
        roughness = confidence * material_seeds.roughness + (1.0 - confidence)
        metallic = confidence * material_seeds.metallic
        transmission = np.clip(confidence * material_seeds.transmission, 1e-4, 1.0 - 1e-4)
        emission = np.maximum(
            confidence[:, None] * material_seeds.emission_linear,
            math.exp(-16.0),
        ).astype(np.float32)
    transmission_logits = _inverse_sigmoid_numpy(transmission)
    emission_log = np.log(emission).astype(np.float32)
    return {
        "diffuse_sh0": ((diffuse - 0.5) / SH_C0).astype(np.float32)[:, None, :],
        "view_shN": np.zeros((count, 15, 3), dtype=np.float32),
        "emission_log": emission_log,
        "transmission_logits": transmission_logits,
        "roughness_logits": _inverse_sigmoid_numpy(roughness),
        "metallic_logits": _inverse_sigmoid_numpy(metallic),
        "material_confidence": confidence.astype(np.float32),
        "emission_anchor_log": emission_log.copy(),
        "transmission_anchor_logits": transmission_logits.copy(),
    }


def _material_preview_colors(parameters: Any, material_aware: bool) -> Any:
    if not material_aware:
        return _sh_preview_colors(parameters["diffuse_sh0"])
    import torch

    diffuse_linear = torch.clamp(
        parameters["diffuse_sh0"][:, 0, :] * SH_C0 + 0.5,
        0.0,
        1.0,
    )
    emission_linear = torch.exp(parameters["emission_log"].clamp(-16.0, 4.0))
    return linear_to_srgb_tensor(diffuse_linear + emission_linear).clamp(0.0, 1.0)


def _preview_log_scales(log_scales: Any, flatten_2d: bool = True) -> Any:
    """Flatten 2D discs when exporting to the interoperable 3DGS PLY format."""
    import torch

    if not flatten_2d:
        return log_scales
    scales = torch.exp(log_scales)
    thickness = torch.clamp(torch.amin(scales[:, :2], dim=1, keepdim=True) * 0.08, min=5e-4)
    return torch.log(torch.cat((scales[:, :2], thickness), dim=1))


def _logit(value: Any) -> Any:
    import torch

    value = torch.as_tensor(value).clamp(1e-4, 1 - 1e-4)
    return torch.log(value / (1 - value))


def _training_limits(vram_gib: float) -> tuple[int, int]:
    """Return image and Gaussian limits that leave headroom for CUDA kernels."""
    if vram_gib <= 8.5:
        return 720, 1_000_000
    if vram_gib <= 12.5:
        return 960, 2_000_000
    if vram_gib <= 16.5:
        return 1280, 3_000_000
    return 1600, 4_000_000


def _initial_log_scales(points: np.ndarray, device: Any, scene_scale: float) -> np.ndarray:
    """Estimate standard 3DGS scales from local COLMAP point spacing.

    The reference set is bounded so a large photo reconstruction cannot create
    an N-squared host allocation. CUDA handles the chunked distance queries.
    """
    import torch

    if len(points) < 2:
        return np.full((len(points), 3), math.log(max(scene_scale * 0.003, 1e-5)), dtype=np.float32)
    point_tensor = torch.from_numpy(np.asarray(points, dtype=np.float32)).to(device)
    reference_count = min(len(points), 16_384)
    reference_indices = torch.linspace(
        0,
        len(points) - 1,
        reference_count,
        device=device,
    ).long()
    references = point_tensor[reference_indices]
    distances: list[Any] = []
    for start in range(0, len(points), 2_048):
        pairwise = torch.cdist(point_tensor[start : start + 2_048], references)
        nearest = torch.topk(pairwise, k=min(2, reference_count), largest=False).values
        distance = nearest[:, 0]
        if nearest.shape[1] > 1:
            distance = torch.where(distance < 1e-7, nearest[:, 1], distance)
        distances.append(distance)
    spacing = torch.cat(distances).clamp_min(scene_scale * 1e-5)
    lower = torch.quantile(spacing, 0.02)
    upper = torch.quantile(spacing, 0.98)
    spacing = spacing.clamp(lower, upper) * 0.65
    isotropic = torch.log(spacing[:, None].expand(-1, 3))
    return isotropic.cpu().numpy().astype(np.float32)


def _constrain_appearance_offsets_(offsets: Any, anchor_index: int) -> None:
    import torch

    with torch.no_grad():
        offsets[:, :3].clamp_(-math.log(2.0), math.log(2.0))
        offsets[:, 3:].clamp_(-0.15, 0.15)
        offsets[anchor_index].zero_()


def _update_smoothed_loss(previous: float | None, current: float) -> float:
    if previous is None:
        return current
    return previous + LOSS_EMA_ALPHA * (current - previous)


def _photometric_quality_accepted(metrics: dict[str, float]) -> bool:
    return bool(
        math.isfinite(metrics.get("medianPsnrDb", math.nan))
        and math.isfinite(metrics.get("medianSsim", math.nan))
        and math.isfinite(metrics.get("medianL1", math.nan))
        and metrics["medianPsnrDb"] >= MINIMUM_PRODUCTION_PSNR_DB
        and metrics["medianSsim"] >= MINIMUM_PRODUCTION_SSIM
        and metrics["medianL1"] <= MAXIMUM_PRODUCTION_L1
    )


def _rgbd_gaussian_limit(initial_count: int, hardware_limit: int) -> int:
    """Keep dense metric seeds from immediately expanding to the VRAM ceiling."""
    return min(
        hardware_limit,
        max(500_000, initial_count * RGBD_GAUSSIAN_MULTIPLIER),
    )


def _metric_surface_scale_limit(
    scene_scale: float,
    seeded_scales: np.ndarray | None,
) -> float | None:
    """Bound learned RGB-D discs while leaving room to cover sparse surfaces."""
    if seeded_scales is None or not seeded_scales.size:
        return None
    # A single monocular-depth outlier must not authorize enormous learned
    # Gaussians. The robust upper footprint preserves real large surfaces while
    # preventing the floating shards produced by an unconstrained maximum.
    maximum_seed_radius = float(
        np.percentile(seeded_scales[:, :2], 99.5)
        if len(seeded_scales) >= 100
        else np.max(seeded_scales[:, :2])
    )
    return max(maximum_seed_radius * 2.0, scene_scale * 0.015)


def _dense_prior_scale_limit(
    scene_scale: float,
    seeded_scales: np.ndarray | None,
) -> float | None:
    """Keep a dense monocular prior from saturating views with large splats."""
    if seeded_scales is None or not seeded_scales.size:
        return None
    footprint = np.max(np.asarray(seeded_scales)[:, :2], axis=1)
    return max(float(np.percentile(footprint, 95.0)) * 1.25, scene_scale * 0.003)


def _prepare_dense_seed_scales(
    seeded_scales: np.ndarray,
    surface_scale_limit: float | None,
    *,
    direct_gaussian_prior: bool,
) -> np.ndarray:
    result = np.asarray(seeded_scales, dtype=np.float32).copy()
    if surface_scale_limit is not None:
        np.minimum(result, surface_scale_limit, out=result)
    if not direct_gaussian_prior:
        # Depth-unprojected seeds need an explicit surface-normal axis. A
        # direct GS head already predicts anisotropy and orientation.
        result[:, 2] = np.minimum(
            result[:, 2], np.minimum(result[:, 0], result[:, 1]) * 0.08
        )
    return result


def _exponential_lr_gamma(total_steps: int) -> float:
    """Match gsplat's position schedule, ending at one percent of the initial LR."""
    return 0.01 ** (1.0 / max(total_steps, 1))


def _finish_training_step(
    scaler: Any,
    optimizers: dict[str, Any],
    pose_optimizer: Any | None,
    pose_active: bool,
    strategy: Any,
    parameters: Any,
    strategy_state: dict[str, Any],
    step: int,
    info: dict[str, Any],
    appearance_optimizer: Any | None = None,
    appearance_active: bool = False,
) -> None:
    """Apply gradients before a gsplat strategy can replace the parameters."""
    for optimizer in optimizers.values():
        scaler.step(optimizer)
    if pose_active:
        scaler.step(pose_optimizer)
    if appearance_active:
        scaler.step(appearance_optimizer)
    scaler.update()
    if strategy is not None:
        strategy.step_post_backward(
            parameters,
            optimizers,
            strategy_state,
            step,
            info,
            packed=True,
        )


def _reset_opacity_if_due(
    strategy: Any,
    parameters: Any,
    optimizers: dict[str, Any],
    strategy_state: dict[str, Any],
    step: int,
    reset: Any | None = None,
) -> bool:
    """Apply the opacity reset intended by gsplat 1.5.3's default strategy.

    gsplat 1.5.3 has a chained-comparison typo in its reset condition, so its
    nominal 3,000-step opacity reset never runs. Without that reset, early
    floaters survive and continue to seed aggressive growth.
    """
    if (
        strategy is None
        or step <= 0
        or step % int(strategy.reset_every) != 0
    ):
        return False
    if reset is None:
        from gsplat.strategy.ops import reset_opa

        reset = reset_opa
    reset(
        params=parameters,
        optimizers=optimizers,
        state=strategy_state,
        value=float(strategy.prune_opa) * 2.0,
    )
    return True


def _cache_local_frame_order(
    frame_count: int,
    epoch: int,
    cache_size: int,
) -> np.ndarray:
    """Shuffle views globally, then reuse cache-sized blocks without changing exposure counts."""
    if frame_count <= 0 or cache_size <= 0:
        raise ValueError("Frame count and cache size must be positive")
    shuffled = np.random.default_rng(epoch).permutation(frame_count)
    blocks = [
        np.tile(shuffled[start : start + cache_size], FRAME_REUSE_PER_LOAD)
        for start in range(0, frame_count, cache_size)
    ]
    return np.concatenate(blocks)


def _training_frame_order(
    frames: list[dict[str, Any]],
    epoch: int,
    cache_size: int,
) -> np.ndarray:
    """Give metric anchors and high-quality views equal training exposure."""
    metric = np.asarray(
        [index for index, frame in enumerate(frames) if frame.get("depth")],
        dtype=np.int64,
    )
    appearance = np.asarray(
        [index for index, frame in enumerate(frames) if not frame.get("depth")],
        dtype=np.int64,
    )
    if not len(metric) or not len(appearance):
        return _cache_local_frame_order(len(frames), epoch, cache_size)
    generator = np.random.default_rng(epoch)
    metric = generator.permutation(metric)
    appearance = generator.permutation(appearance)
    group_size = max(len(metric), len(appearance))
    balanced = np.column_stack(
        (np.resize(metric, group_size), np.resize(appearance, group_size))
    ).reshape(-1)
    blocks = [
        np.tile(balanced[start : start + cache_size], FRAME_REUSE_PER_LOAD)
        for start in range(0, len(balanced), cache_size)
    ]
    return np.concatenate(blocks)


def _ssim(predicted: Any, target: Any, mask: Any | None = None) -> Any:
    import torch.nn.functional as functional

    predicted = predicted.permute(2, 0, 1).unsqueeze(0)
    target = target.permute(2, 0, 1).unsqueeze(0)
    mean_x = functional.avg_pool2d(predicted, 3, 1, 1)
    mean_y = functional.avg_pool2d(target, 3, 1, 1)
    variance_x = functional.avg_pool2d(predicted * predicted, 3, 1, 1) - mean_x.square()
    variance_y = functional.avg_pool2d(target * target, 3, 1, 1) - mean_y.square()
    covariance = functional.avg_pool2d(predicted * target, 3, 1, 1) - mean_x * mean_y
    score = ((2 * mean_x * mean_y + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (mean_x.square() + mean_y.square() + 0.01**2) * (variance_x + variance_y + 0.03**2)
    )
    if mask is None:
        return score.mean()
    valid_windows = functional.avg_pool2d(
        mask.to(score.dtype).unsqueeze(0).unsqueeze(0),
        3,
        1,
        1,
    ) > 0.999
    valid_scores = score.masked_select(valid_windows.expand_as(score))
    return valid_scores.mean() if valid_scores.numel() else score.mean()


def _source_resolution_crop(
    frame: dict[str, Any],
    sample: int,
    maximum_raster_dimension: int,
) -> tuple[int, int, int, int] | None:
    """Return a deterministic bounded tile in the source camera pixel grid."""
    intrinsics = frame["intrinsics"]
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    if max(width, height) <= maximum_raster_dimension:
        return None
    tile_width = min(width, maximum_raster_dimension)
    tile_height = min(height, maximum_raster_dimension)
    columns = math.ceil(width / tile_width)
    rows = math.ceil(height / tile_height)
    tile_count = columns * rows
    # A frame-specific rotation prevents synchronized cameras from training on
    # the same corner while sample % tile_count guarantees complete coverage.
    tile_index = (sample + int(frame.get("frameIndex", 0)) * 17) % tile_count
    row, column = divmod(tile_index, columns)
    left = 0 if columns == 1 else round(column * (width - tile_width) / (columns - 1))
    top = 0 if rows == 1 else round(row * (height - tile_height) / (rows - 1))
    return left, top, tile_width, tile_height


def _source_resolution_tile_count(
    frame: dict[str, Any], maximum_raster_dimension: int
) -> int:
    intrinsics = frame["intrinsics"]
    return math.ceil(int(intrinsics["width"]) / maximum_raster_dimension) * math.ceil(
        int(intrinsics["height"]) / maximum_raster_dimension
    )


def _source_resolution_start_step(
    frames: list[dict[str, Any]],
    iterations: int,
    maximum_raster_dimension: int,
) -> int:
    """Reserve enough final iterations to expose every calibrated source tile."""
    minimum_steps = math.ceil(iterations * MINIMUM_SOURCE_RESOLUTION_FRACTION)
    coverage_steps = sum(
        _source_resolution_tile_count(frame, maximum_raster_dimension)
        for frame in frames
    )
    return max(1, iterations - max(minimum_steps, coverage_steps))


def _source_resolution_frame_order(
    frames: list[dict[str, Any]], maximum_raster_dimension: int
) -> np.ndarray:
    """Schedule every source tile once while retaining cache-local camera blocks."""
    camera_order = np.random.default_rng(0x5CA11A).permutation(len(frames))
    return np.concatenate(
        [
            np.full(
                _source_resolution_tile_count(frames[int(index)], maximum_raster_dimension),
                int(index),
                dtype=np.int64,
            )
            for index in camera_order
        ]
    )


def _uses_source_resolution(step: int, start_step: int) -> bool:
    return step >= start_step


def _frame_tensors(
    root: Path,
    frame: dict[str, Any],
    max_dimension: int,
    source_crop: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    import torch

    with Image.open(root / frame["image"]) as source:
        image = source.convert("RGB")
        source_width, source_height = image.size
        declared = frame["intrinsics"]
        if (source_width, source_height) != (
            int(declared["width"]),
            int(declared["height"]),
        ):
            raise ValueError("Canonical RGB dimensions do not match their pinhole intrinsics")
        crop_left = 0
        crop_top = 0
        if source_crop is not None:
            crop_left, crop_top, crop_width, crop_height = source_crop
            if (
                crop_left < 0
                or crop_top < 0
                or crop_width <= 0
                or crop_height <= 0
                or crop_left + crop_width > source_width
                or crop_top + crop_height > source_height
            ):
                raise ValueError("Source-resolution training crop escapes its calibrated image")
            image = image.crop(
                (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)
            )
            requested_scale = 1.0
        else:
            requested_scale = min(1.0, max_dimension / max(image.size))
        if source_crop is None and requested_scale < 1.0:
            image = image.resize(
                (round(image.width * requested_scale), round(image.height * requested_scale)),
                Image.Resampling.LANCZOS,
            )
        rgb = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())
    scale_x = 1.0 if source_crop is not None else image.width / source_width
    scale_y = 1.0 if source_crop is not None else image.height / source_height
    intrinsics = frame["intrinsics"]
    intrinsic = torch.tensor(
        [
            [
                intrinsics["fx"] * scale_x,
                0.0,
                (intrinsics["cx"] + 0.5) * scale_x - 0.5 - crop_left,
            ],
            [
                0.0,
                intrinsics["fy"] * scale_y,
                (intrinsics["cy"] + 0.5) * scale_y - 0.5 - crop_top,
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    world_from_camera = torch.tensor(frame["worldFromRgbCamera"], dtype=torch.float32).reshape(4, 4)
    value: dict[str, Any] = {
        "rgb": rgb,
        "K": intrinsic,
        "cameraToWorld": world_from_camera,
        "view": torch.linalg.inv(world_from_camera),
        "width": image.width,
        "height": image.height,
        "sourceCrop": source_crop,
        "sourceResolution": requested_scale == 1.0,
    }
    if frame.get("depth"):
        with Image.open(root / frame["depth"]) as source:
            if source_crop is not None:
                if source.size != (source_width, source_height):
                    raise ValueError("Source-resolution RGB and registered depth dimensions differ")
                depth_image = source.crop(
                    (
                        crop_left,
                        crop_top,
                        crop_left + image.width,
                        crop_top + image.height,
                    )
                )
            else:
                depth_image = source.resize((image.width, image.height), Image.Resampling.NEAREST)
            depth = torch.from_numpy(np.asarray(depth_image, dtype=np.uint16).copy())
        with Image.open(root / frame["depthMask"]) as source:
            mask_image = (
                source.crop(
                    (
                        crop_left,
                        crop_top,
                        crop_left + image.width,
                        crop_top + image.height,
                    )
                )
                if source_crop is not None
                else source.resize((image.width, image.height), Image.Resampling.NEAREST)
            )
            mask = torch.from_numpy(np.asarray(mask_image, dtype=np.uint8).copy())
        value.update(depth=depth, mask=mask)
        if frame.get("depthConfidence"):
            with Image.open(root / frame["depthConfidence"]) as source:
                confidence_image = (
                    source.crop(
                        (
                            crop_left,
                            crop_top,
                            crop_left + image.width,
                            crop_top + image.height,
                        )
                    )
                    if source_crop is not None
                    else source.resize(
                        (image.width, image.height), Image.Resampling.NEAREST
                    )
                )
                confidence = torch.from_numpy(
                    np.asarray(confidence_image, dtype=np.uint8).copy()
                )
            value["depthConfidence"] = confidence
    for key, tensor in tuple(value.items()):
        if isinstance(tensor, torch.Tensor):
            value[key] = tensor.pin_memory()
    return value


def _frame_to_device(
    frame: dict[str, Any], device: Any, *, linear_rgb: bool = False
) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {}
    for key, value in frame.items():
        if not isinstance(value, torch.Tensor):
            result[key] = value
            continue
        transferred = value.to(device, non_blocking=True)
        if key == "rgb":
            transferred = transferred.float().div_(255.0)
            if linear_rgb:
                transferred = srgb_to_linear_tensor(transferred)
        elif key == "depth":
            transferred = transferred.float().div_(1000.0)
        elif key == "mask":
            transferred = transferred > 0
        elif key == "depthConfidence":
            transferred = transferred.float().div_(255.0)
        result[key] = transferred
    return result


def train_dataset(
    dataset_path: Path,
    project_root: Path,
    iterations: int = 30_000,
    resume: bool = False,
    progress_start: float = 0.0,
) -> dict[str, Any]:
    import torch
    from gsplat.rendering import rasterization, rasterization_2dgs
    from gsplat.strategy import DefaultStrategy

    if not torch.cuda.is_available():
        raise RuntimeError("Gaussian training requires a CUDA-capable PyTorch runtime; current ScanLan features remain available on CPU")
    device = torch.device("cuda")
    device_properties = torch.cuda.get_device_properties(device)
    vram_gib = device_properties.total_memory / (1024**3)
    training_dimension, hardware_maximum_gaussians = _training_limits(vram_gib)
    host_cache_size = max(1, min(int(os.environ.get("SCANLAN_SPLAT_FRAME_CACHE", "4")), 16))
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    root, dataset = load_dataset(dataset_path)
    initialization = resolve_initialization_contract(dataset)
    frames = dataset.get("frames", [])
    if len(frames) < 2:
        raise ValueError("At least two registered RGB views are required for Gaussian training")
    uses_depth = any(frame.get("depth") for frame in frames)
    hybrid = dataset.get("sourceMode") == "hybrid"
    pose_refinement_enabled = bool(
        dataset.get("poseRefinement", False)
        or (
            dataset.get("metric")
            and all(frame.get("depth") and frame.get("depthMask") for frame in frames)
        )
    )
    appearance_optimization_enabled = bool(
        dataset.get("appearanceOptimization", False)
        and (not dataset.get("metric") or hybrid)
    )
    appearance_anchor_index = int(dataset.get("appearanceAnchorIndex", 0))
    appearance_anchor_index = min(max(appearance_anchor_index, 0), len(frames) - 1)
    output_root = project_root / "outputs"
    checkpoint_path = output_root / "splat-checkpoint.pt"
    progress_path = output_root / "splat-progress.json"
    (
        points,
        colors,
        seeded_scales,
        seeded_quaternions,
        seeded_confidence,
        seeded_opacity,
    ) = (
        _read_seed_parameters(root, dataset)
    )
    if len(points) == 0:
        raise ValueError("Sparse or RGB-D initialization contains no points")
    material_seeds = resolve_gaussian_material_seeds(root, dataset, len(points))
    material_aware = material_seeds is not None
    if initialization.is_dense and (
        seeded_scales is None or seeded_quaternions is None
    ):
        raise ValueError("Dense Gaussian initialization parameter sidecar is unavailable")
    if initialization.is_direct and seeded_opacity is None:
        raise ValueError("Direct Gaussian initialization is missing predicted opacity")
    metric_seeded = bool(dataset.get("metric") and initialization.is_dense)
    dense_geometry_prior = initialization.is_dense and not metric_seeded
    direct_gaussian_prior = initialization.is_direct
    uses_2dgs = initialization.uses_2dgs
    requested_iterations = iterations
    if metric_seeded and not hybrid:
        # Dense RGB-D already supplies the measured surface and appearance. A
        # short bounded pass is enough to validate it and conservatively refine
        # camera poses; longer runs add no surface detail in fixed-surface mode.
        iterations = min(iterations, MAX_METRIC_ITERATIONS)
    required_source_steps = sum(
        _source_resolution_tile_count(frame, training_dimension) for frame in frames
    )
    # A requested iteration count is a speed preference, not permission to
    # skip source pixels. Extend the run only when one complete tiled pass
    # cannot fit; the manifest reports requested and effective counts.
    iterations = max(iterations, required_source_steps + 1)
    source_resolution_start_step = _source_resolution_start_step(
        frames, iterations, training_dimension
    )
    source_resolution_frame_order = _source_resolution_frame_order(
        frames, training_dimension
    )

    centred = points - np.median(points, axis=0, keepdims=True)
    scene_scale = max(float(np.percentile(np.linalg.norm(centred, axis=1), 90)), 1e-3)
    maximum_gaussians = (
        len(points)
        if metric_seeded
        else min(
            hardware_maximum_gaussians,
            max(1_000_000, len(points) * 2),
        )
        if dense_geometry_prior
        else _rgbd_gaussian_limit(len(points), hardware_maximum_gaussians)
        if seeded_scales is not None
        else hardware_maximum_gaussians
    )
    surface_scale_limit = (
        _dense_prior_scale_limit(scene_scale, seeded_scales)
        if dense_geometry_prior
        else _metric_surface_scale_limit(scene_scale, seeded_scales)
    )
    training_seeded_scales = seeded_scales
    if dense_geometry_prior and seeded_scales is not None:
        training_seeded_scales = _prepare_dense_seed_scales(
            seeded_scales,
            surface_scale_limit,
            direct_gaussian_prior=direct_gaussian_prior,
        )
    metric_opacity_bounds = (
        (float(_logit(0.01)), float(_logit(0.8)))
        if metric_seeded
        else None
    )
    initial_scale = max(scene_scale * 0.003, 0.004 if dataset.get("metric") else scene_scale * 0.001)
    scales = (
        np.log(
            np.maximum(
                training_seeded_scales
                * (RGBD_SURFACE_SCALE_MULTIPLIER if metric_seeded else 1.0),
                1e-5,
            )
        )
        if seeded_scales is not None
        else (
            np.full((len(points), 3), math.log(initial_scale), dtype=np.float32)
            if uses_2dgs
            else _initial_log_scales(points, device, scene_scale)
        )
    )
    quaternions = (
        seeded_quaternions
        if seeded_quaternions is not None
        else np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (len(points), 1),
        )
    )
    appearance_parameters = _material_parameter_initialization(colors, material_seeds)
    initial_opacity = (
        np.clip(seeded_opacity, 1e-4, 1.0 - 1e-4)
        if direct_gaussian_prior and seeded_opacity is not None
        else
        np.clip(seeded_confidence, 0.05, 1.0) * RGBD_SURFACE_OPACITY
        if metric_seeded and seeded_confidence is not None
        else np.full(
            len(points),
            RGBD_SURFACE_OPACITY
            if metric_seeded
            else DENSE_PRIOR_INITIAL_OPACITY
            if dense_geometry_prior
            else 0.1,
            dtype=np.float32,
        )
    )
    parameter_values = {
        "means": torch.from_numpy(points),
        "scales": torch.from_numpy(np.asarray(scales, dtype=np.float32)),
        "quats": torch.from_numpy(np.asarray(quaternions, dtype=np.float32)),
        "opacities": _logit(
            torch.from_numpy(np.asarray(initial_opacity, dtype=np.float32))
        ),
        **{
            name: torch.from_numpy(value)
            for name, value in appearance_parameters.items()
        },
    }
    start_step = 0
    checkpoint = None
    if resume and checkpoint_path.is_file():
        checkpoint, quarantined_checkpoint = _load_checkpoint_or_quarantine(
            checkpoint_path,
            lambda path: torch.load(path, map_location=device, weights_only=False),
        )
        if quarantined_checkpoint is not None:
            _write_json_atomic(
                progress_path,
                {
                    "stage": "splat_training",
                    "detail": "Unreadable checkpoint preserved; restarting Gaussian training cleanly",
                    "progress": progress_start,
                    "stageProgress": 0.0,
                    "iteration": 0,
                    "totalIterations": iterations,
                    "etaSeconds": None,
                    "stageEtaSeconds": None,
                    "elapsedSeconds": 0,
                    "computeBackend": f"{torch.cuda.get_device_name(device)} · CUDA AMP / gsplat",
                },
            )
    if checkpoint is not None:
        if checkpoint.get("trainerVersion") != TRAINER_VERSION:
            raise RuntimeError("Splat checkpoint belongs to an older trainer; start a new job")
        if checkpoint.get("fingerprint") != dataset.get("fingerprint"):
            raise RuntimeError("Splat checkpoint source fingerprint no longer matches")
        if set(checkpoint.get("parameters", {})) != set(parameter_values):
            raise RuntimeError("Splat checkpoint belongs to an incompatible trainer version; start a new job")
        parameter_values = checkpoint["parameters"]
        start_step = int(checkpoint["step"]) + 1
    if len(parameter_values["means"]) > maximum_gaussians:
        raise RuntimeError(
            f"Splat checkpoint contains {len(parameter_values['means']):,} Gaussians, "
            f"above this GPU's {maximum_gaussians:,} safety limit; start a new job"
        )
    parameters = torch.nn.ParameterDict(
        {name: torch.nn.Parameter(value.to(device)) for name, value in parameter_values.items()}
    )
    pose_offset_values = torch.zeros((len(frames), 9), dtype=torch.float32)
    if checkpoint is not None and checkpoint.get("poseOffsets") is not None:
        restored_pose_offsets = checkpoint["poseOffsets"]
        if tuple(restored_pose_offsets.shape) != tuple(pose_offset_values.shape):
            raise RuntimeError(
                "Splat checkpoint belongs to an incompatible camera-pose layout; start a new job"
            )
        pose_offset_values = restored_pose_offsets.detach().cpu().to(torch.float32)
    pose_offsets = torch.nn.Parameter(
        pose_offset_values.to(device),
        requires_grad=pose_refinement_enabled,
    )
    pose_anchor_mask = torch.tensor(
        [
            bool(frame.get("poseAnchor", frame.get("metricAnchor", False)))
            for frame in frames
        ],
        dtype=torch.bool,
        device=device,
    )
    pose_refinement_mask = (~pose_anchor_mask).to(torch.float32).unsqueeze(-1)
    appearance_offset_values = torch.zeros((len(frames), 6), dtype=torch.float32)
    if checkpoint is not None and checkpoint.get("appearanceOffsets") is not None:
        restored_appearance_offsets = checkpoint["appearanceOffsets"]
        if tuple(restored_appearance_offsets.shape) != tuple(appearance_offset_values.shape):
            raise RuntimeError(
                "Splat checkpoint belongs to an incompatible appearance layout; start a new job"
            )
        appearance_offset_values = restored_appearance_offsets.detach().cpu().to(torch.float32)
    appearance_offsets = torch.nn.Parameter(
        appearance_offset_values.to(device),
        requires_grad=appearance_optimization_enabled,
    )
    learning_rates = {
        "means": 0.0 if metric_seeded else (8e-5 if dense_geometry_prior else 1.6e-4) * scene_scale,
        "scales": 0.0 if metric_seeded else 2.5e-3 if dense_geometry_prior else 5e-3,
        "quats": 0.0 if metric_seeded else 1e-3,
        "opacities": 1e-2 if hybrid and metric_seeded else 0.0 if metric_seeded else 5e-2,
        "diffuse_sh0": 2.5e-3 if hybrid and metric_seeded else 0.0 if metric_seeded else 2.5e-3,
        "view_shN": 1.25e-4 if hybrid and metric_seeded else 0.0 if metric_seeded else 1.25e-4,
        "emission_log": 1e-3 if material_aware else 0.0,
        "transmission_logits": 5e-4 if material_aware else 0.0,
        "roughness_logits": 0.0,
        "metallic_logits": 0.0,
        "material_confidence": 0.0,
        "emission_anchor_log": 0.0,
        "transmission_anchor_logits": 0.0,
    }
    optimizers = {
        name: torch.optim.Adam(
            [{"params": [parameters[name]], "lr": learning_rate, "name": name}],
            eps=1e-15,
            fused=True,
        )
        for name, learning_rate in learning_rates.items()
    }
    pose_optimizer = (
        torch.optim.Adam([pose_offsets], lr=1e-5, weight_decay=1e-6)
        if pose_refinement_enabled
        else None
    )
    appearance_optimizer = (
        torch.optim.Adam([appearance_offsets], lr=1e-3, weight_decay=1e-5)
        if appearance_optimization_enabled
        else None
    )
    neighbor_pairs = torch.tensor(
        [
            (index - 1, index)
            for index in range(1, len(frames))
            if frames[index].get("phaseId") == frames[index - 1].get("phaseId")
        ],
        dtype=torch.long,
        device=device,
    ).reshape(-1, 2)
    pose_refine_start = max(500, iterations // 20)
    strategy = None
    strategy_state: dict[str, Any] = {}
    if initialization.adaptive_densification:
        if uses_2dgs:
            strategy = DefaultStrategy(
                # gsplat's 2DGS densification signal is `gradient_2dgs`; unlike the
                # projected 3DGS means, it does not expose an AbsGrad accumulator.
                absgrad=False,
                grow_grad2d=0.0008,
                refine_start_iter=500,
                refine_stop_iter=min(15_000, max(1_000, iterations // 2)),
                pause_refine_after_reset=len(frames),
                key_for_gradient="gradient_2dgs",
            )
        else:
            strategy = DefaultStrategy(
                absgrad=False,
                grow_grad2d=0.00035 if dense_geometry_prior else 0.0002,
                grow_scale3d=0.02 if dense_geometry_prior else 0.01,
                prune_scale3d=0.10 if dense_geometry_prior else 0.12,
                refine_start_iter=1_000 if dense_geometry_prior else 500,
                refine_stop_iter=(
                    min(12_000, max(3_000, iterations // 2))
                    if dense_geometry_prior
                    else min(20_000, max(2_000, (iterations * 2) // 3))
                ),
                pause_refine_after_reset=len(frames),
                key_for_gradient="means2d",
            )
        strategy.check_sanity(parameters, optimizers)
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    scaler = torch.amp.GradScaler("cuda")
    if checkpoint is not None:
        for name, value in checkpoint.get("optimizers", {}).items():
            if name in optimizers:
                optimizers[name].load_state_dict(value)
        strategy_state = checkpoint.get("strategy", strategy_state)
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        if pose_optimizer is not None and checkpoint.get("poseOptimizer"):
            pose_optimizer.load_state_dict(checkpoint["poseOptimizer"])
        if appearance_optimizer is not None and checkpoint.get("appearanceOptimizer"):
            appearance_optimizer.load_state_dict(checkpoint["appearanceOptimizer"])

    def save_checkpoint(step: int) -> None:
        _write_checkpoint_atomic(
            checkpoint_path,
            {
                "step": step,
                "trainerVersion": TRAINER_VERSION,
                "fingerprint": dataset.get("fingerprint"),
                "parameters": {key: value.detach().cpu() for key, value in parameters.items()},
                "optimizers": {key: value.state_dict() for key, value in optimizers.items()},
                "poseOffsets": pose_offsets.detach().cpu() if pose_refinement_enabled else None,
                "poseOptimizer": pose_optimizer.state_dict() if pose_optimizer is not None else None,
                "appearanceOffsets": (
                    appearance_offsets.detach().cpu()
                    if appearance_optimization_enabled
                    else None
                ),
                "appearanceOptimizer": (
                    appearance_optimizer.state_dict()
                    if appearance_optimizer is not None
                    else None
                ),
                "strategy": strategy_state,
                "scaler": scaler.state_dict(),
                "sourceResolutionSamples": source_resolution_samples,
            },
            torch.save,
        )

    # RGB-D keyframes stay in a small pinned host-memory LRU. Keeping an entire
    # capture resident on CUDA made VRAM usage grow with session duration and
    # was the main source of 12 GB laptop-GPU failures.
    cached_frames: OrderedDict[
        tuple[int, tuple[int, int, int, int] | None], dict[str, Any]
    ] = OrderedDict()
    started = time.perf_counter()
    last_live_preview_at = 0.0
    last_loss = 0.0
    smoothed_loss: float | None = None
    frame_epoch = -1
    frame_order = _training_frame_order(frames, 0, host_cache_size)
    densification_stopped_at: int | None = None
    source_resolution_steps = 0
    source_resolution_tiles: set[tuple[int, tuple[int, int, int, int] | None]] = set()
    source_resolution_samples = [0] * len(frames)
    if checkpoint is not None and checkpoint.get("sourceResolutionSamples") is not None:
        restored_source_samples = [
            int(value) for value in checkpoint["sourceResolutionSamples"]
        ]
        if len(restored_source_samples) != len(frames) or any(
            value < 0 for value in restored_source_samples
        ):
            raise RuntimeError(
                "Splat checkpoint belongs to an incompatible source-resolution schedule; start a new job"
            )
        source_resolution_samples = restored_source_samples
        source_resolution_steps = sum(restored_source_samples)
        for restored_frame_index, restored_count in enumerate(restored_source_samples):
            for restored_sample in range(restored_count):
                source_resolution_tiles.add(
                    (
                        restored_frame_index,
                        _source_resolution_crop(
                            frames[restored_frame_index],
                            restored_sample,
                            training_dimension,
                        ),
                    )
                )

    def publish_live_preview() -> None:
        nonlocal last_live_preview_at
        with torch.no_grad():
            preview_count = min(len(parameters["means"]), 200_000)
            preview_indices = (
                torch.linspace(0, len(parameters["means"]) - 1, preview_count, device=device).long()
                if len(parameters["means"]) > preview_count
                else slice(None)
            )
            preview_parameters = {
                name: value[preview_indices] for name, value in parameters.items()
            }
            _preview_coefficients, preview_optical_opacity, _preview_material = (
                compose_material_render_state(
                    preview_parameters, material_aware, SH_C0
                )
            )
            export_splat_preview(
                output_root / "room-splat.preview.splat",
                parameters["means"][preview_indices].detach().cpu().numpy(),
                _material_preview_colors(
                    preview_parameters,
                    material_aware,
                ).detach().cpu().numpy(),
                _logit(preview_optical_opacity).detach().cpu().numpy(),
                _preview_log_scales(
                    parameters["scales"][preview_indices],
                    flatten_2d=uses_2dgs,
                ).detach().cpu().numpy(),
                torch.nn.functional.normalize(parameters["quats"][preview_indices], dim=-1).detach().cpu().numpy(),
            )
        last_live_preview_at = time.perf_counter()

    for step in range(start_step, iterations):
        if (output_root / "cancel.flag").exists():
            save_checkpoint(step - 1)
            raise RuntimeError("Gaussian training cancelled; checkpoint saved")
        source_resolution_active = _uses_source_resolution(
            step, source_resolution_start_step
        )
        if source_resolution_active:
            frame_index = int(
                source_resolution_frame_order[
                    (step - source_resolution_start_step)
                    % len(source_resolution_frame_order)
                ]
            )
        else:
            next_epoch = step // len(frame_order)
            if next_epoch != frame_epoch:
                frame_epoch = next_epoch
                frame_order = _training_frame_order(
                    frames,
                    frame_epoch,
                    host_cache_size,
                )
            frame_index = int(frame_order[step % len(frame_order)])
        source_crop = None
        if source_resolution_active:
            source_crop = _source_resolution_crop(
                frames[frame_index],
                source_resolution_samples[frame_index],
                training_dimension,
            )
            source_resolution_samples[frame_index] += 1
        cache_key = (frame_index, source_crop)
        if source_resolution_active:
            source_resolution_steps += 1
            source_resolution_tiles.add(cache_key)
        if cache_key not in cached_frames:
            cached_frames[cache_key] = _frame_tensors(
                root,
                frames[frame_index],
                training_dimension,
                source_crop,
            )
            if len(cached_frames) > host_cache_size:
                cached_frames.popitem(last=False)
        else:
            cached_frames.move_to_end(cache_key)
        frame = _frame_to_device(
            cached_frames[cache_key], device, linear_rgb=material_aware
        )
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        pose_active = pose_optimizer is not None and step >= pose_refine_start
        if pose_optimizer is not None:
            pose_optimizer.zero_grad(set_to_none=True)
        appearance_active = appearance_optimizer is not None
        if appearance_optimizer is not None:
            appearance_optimizer.zero_grad(set_to_none=True)
        refined_view = frame["view"]
        if pose_active:
            correction = pose_delta_matrix(
                pose_offsets[frame_index] * pose_refinement_mask[frame_index]
            )
            refined_camera_to_world = frame["cameraToWorld"] @ correction
            refined_view = torch.linalg.inv(refined_camera_to_world)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            sh_degree = min(3, step // max(iterations // 4, 1))
            pose_regularization_value = torch.zeros((), device=device)
            appearance_regularization_value = torch.zeros((), device=device)
            material_regularization_value = torch.zeros((), device=device)
            normalized_quaternions = torch.nn.functional.normalize(
                parameters["quats"], dim=-1
            )
            render_normals = None
            surface_normals = None
            render_distortion = None
            render_coefficients, optical_opacities, _material_state = (
                compose_material_render_state(parameters, material_aware, SH_C0)
            )
            if uses_2dgs:
                (
                    rendering,
                    render_alpha,
                    render_normals,
                    surface_normals,
                    render_distortion,
                    _render_median,
                    info,
                ) = rasterization_2dgs(
                    means=parameters["means"],
                    quats=normalized_quaternions,
                    scales=torch.exp(parameters["scales"]).clamp(1e-5, scene_scale),
                    opacities=optical_opacities,
                    colors=render_coefficients,
                    viewmats=refined_view.unsqueeze(0),
                    Ks=frame["K"].unsqueeze(0),
                    width=frame["width"],
                    height=frame["height"],
                    packed=True,
                    absgrad=False,
                    render_mode="RGB+ED",
                    sh_degree=sh_degree,
                    distloss=True,
                    depth_mode="expected",
                )
            else:
                rendering, render_alpha, info = rasterization(
                    means=parameters["means"],
                    quats=normalized_quaternions,
                    scales=torch.exp(parameters["scales"]).clamp(1e-5, scene_scale),
                    opacities=optical_opacities,
                    colors=render_coefficients,
                    viewmats=refined_view.unsqueeze(0),
                    Ks=frame["K"].unsqueeze(0),
                    width=frame["width"],
                    height=frame["height"],
                    packed=True,
                    absgrad=False,
                    render_mode="RGB",
                    sh_degree=sh_degree,
                )
            if strategy is not None:
                strategy.step_pre_backward(parameters, optimizers, strategy_state, step, info)
            predicted_rgb = rendering[0, ..., :3].clamp(0.0, 1.0)
            predicted_for_loss = predicted_rgb
            if appearance_active:
                appearance = appearance_offsets[frame_index]
                predicted_for_loss = (
                    predicted_rgb * torch.exp(appearance[:3]) + appearance[3:]
                ).clamp(0.0, 1.0)
                appearance_regularization_value = appearance_offsets.square().mean()
            rgb_mask = frame.get("mask")
            rgb_residual = torch.abs(predicted_for_loss - frame["rgb"])
            rgb_l1 = (
                rgb_residual[rgb_mask].mean()
                if rgb_mask is not None and torch.any(rgb_mask)
                else rgb_residual.mean()
            )
            observation_confidence = min(
                1.0,
                max(0.05, float(frames[frame_index].get("poseConfidence", 1.0))),
            )
            loss = observation_confidence * (
                0.8 * rgb_l1
                + 0.2 * (1.0 - _ssim(predicted_for_loss, frame["rgb"], rgb_mask))
            )
            if appearance_active:
                loss = loss + 1e-4 * appearance_regularization_value
            material_regularization_value, _material_loss_parts = material_regularization(
                parameters, material_aware
            )
            loss = loss + material_regularization_value
            depth_value = torch.zeros((), device=device)
            normal_value = torch.zeros((), device=device)
            distortion_value = torch.zeros((), device=device)
            if "depth" in frame and rendering.shape[-1] > 3:
                depth_value = masked_robust_depth_loss(
                    rendering[0, ..., 3],
                    frame["depth"],
                    frame["mask"],
                    frame.get("depthConfidence"),
                )
                loss = loss + depth_weight(step, iterations) * depth_value
            if uses_2dgs and not metric_seeded and step >= max(1_000, iterations // 4):
                alpha = render_alpha[0, ..., 0]
                valid_normals = alpha > 0.5
                if "mask" in frame:
                    valid_normals &= frame["mask"]
                if torch.any(valid_normals):
                    normal_error = 1.0 - torch.sum(
                        render_normals[0] * surface_normals[0] * alpha[..., None].detach(),
                        dim=-1,
                    )
                    normal_value = normal_error[valid_normals].clamp(0.0, 2.0).mean()
                    loss = loss + 0.05 * normal_value
            if uses_2dgs and not metric_seeded and step >= max(1_000, iterations // 10):
                distortion_value = render_distortion.mean()
                loss = loss + 0.01 * distortion_value
            if pose_active:
                pose_regularization_value = pose_regularization(
                    pose_offsets,
                    neighbor_pairs,
                )
                loss = loss + 1e-4 * pose_regularization_value
        scaler.scale(loss).backward()
        if strategy is not None:
            gradient_scale = scaler.get_scale()
            projected_gradient = info.get("gradient_2dgs" if uses_2dgs else "means2d")
            if projected_gradient is not None and projected_gradient.grad is not None:
                projected_gradient.grad.div_(gradient_scale)
        # DefaultStrategy can at most double the set in one grow cycle. Stop
        # before that worst case can cross the hardware profile's hard bound.
        if (
            strategy is not None
            and densification_stopped_at is None
            and len(parameters["means"]) > maximum_gaussians // 2
        ):
            # DefaultStrategy treats refine_stop_iter as inclusive. Put the
            # boundary behind the current step so a checkpoint resumed exactly
            # on a refinement interval cannot schedule one final doubling.
            strategy.refine_stop_iter = min(strategy.refine_stop_iter, step - 1)
            densification_stopped_at = len(parameters["means"])
        # gsplat topology updates replace the Parameters and resize optimizer
        # state. Apply this iteration's gradients before that replacement; the
        # strategy can still consume the retained, explicitly unscaled 2D
        # projection gradient afterward.
        _finish_training_step(
            scaler,
            optimizers,
            pose_optimizer,
            pose_active,
            strategy,
            parameters,
            strategy_state,
            step,
            info,
            appearance_optimizer,
            appearance_active,
        )
        _reset_opacity_if_due(
            strategy,
            parameters,
            optimizers,
            strategy_state,
            step,
        )
        learning_rate_decay = _exponential_lr_gamma(iterations) ** (step + 1)
        optimizers["means"].param_groups[0]["lr"] = (
            learning_rates["means"] * learning_rate_decay
        )
        if pose_optimizer is not None:
            pose_optimizer.param_groups[0]["lr"] = 1e-5 * learning_rate_decay
        if surface_scale_limit is not None:
            with torch.no_grad():
                parameters["scales"].clamp_(max=math.log(surface_scale_limit))
        if metric_opacity_bounds is not None:
            with torch.no_grad():
                parameters["opacities"].clamp_(
                    min=metric_opacity_bounds[0],
                    max=metric_opacity_bounds[1],
                )
        if material_aware:
            with torch.no_grad():
                parameters["emission_log"].clamp_(-16.0, math.log(16.0))
                parameters["transmission_logits"].clamp_(
                    float(_logit(1e-4)), float(_logit(1.0 - 1e-4))
                )
                parameters["diffuse_sh0"].clamp_(
                    min=-0.5 / SH_C0,
                    max=0.5 / SH_C0,
                )
        if len(parameters["means"]) > maximum_gaussians:
            raise RuntimeError(
                f"Adaptive densification exceeded the {maximum_gaussians:,}-Gaussian "
                "GPU safety limit"
            )
        if pose_active:
            constrain_pose_offsets_(pose_offsets)
            with torch.no_grad():
                pose_offsets[pose_anchor_mask].zero_()
        if appearance_active:
            _constrain_appearance_offsets_(appearance_offsets, appearance_anchor_index)
        last_loss = float(loss.detach())
        smoothed_loss = _update_smoothed_loss(smoothed_loss, last_loss)
        if step % 20 == 0 or step + 1 == iterations:
            elapsed = time.perf_counter() - started
            completed = step - start_step + 1
            eta = round(elapsed / max(completed, 1) * (iterations - step - 1))
            stage_progress = (step + 1) / iterations
            maximum_pose_translation_mm = (
                float(
                    torch.linalg.vector_norm(pose_offsets[:, :3], dim=-1).max().detach()
                    * 1000.0
                )
                if pose_refinement_enabled
                else 0.0
            )
            progress_action = (
                "Validated calibrated RGB-D splats"
                if metric_seeded
                else "Refined dense LingBot 3DGS"
                if dense_geometry_prior
                else "Optimized 2DGS" if uses_2dgs else "Optimized photoreal 3DGS"
            )
            raster_label = (
                f"source {frame['width']}×{frame['height']} tile"
                if frame["sourceResolution"]
                else f"global {frame['width']}×{frame['height']}"
            )
            _write_json_atomic(
                progress_path,
                {
                    "stage": "splat_training",
                    "detail": f"{progress_action} iteration {step + 1:,} of {iterations:,} · loss {last_loss:.4f} · rolling {smoothed_loss:.4f} · {raster_label} · {len(parameters['means']):,} Gaussians",
                    "progress": progress_start + (1.0 - progress_start) * stage_progress,
                    "stageProgress": stage_progress,
                    "iteration": step + 1,
                    "totalIterations": iterations,
                    "loss": last_loss,
                    "smoothedLoss": smoothed_loss,
                    "rgbLoss": float(rgb_l1.detach()),
                    "depthLoss": float(depth_value.detach()),
                    "normalLoss": float(normal_value.detach()),
                    "distortionLoss": float(distortion_value.detach()),
                    "poseRegularizationLoss": float(pose_regularization_value.detach()),
                    "appearanceRegularizationLoss": float(appearance_regularization_value.detach()),
                    "materialRegularizationLoss": float(material_regularization_value.detach()),
                    "maximumPoseTranslationMm": maximum_pose_translation_mm,
                    "gaussianCount": len(parameters["means"]),
                    "maximumGaussians": maximum_gaussians,
                    "sourceResolution": frame["sourceResolution"],
                    "sourceCrop": frame["sourceCrop"],
                    "etaSeconds": eta,
                    "stageEtaSeconds": eta,
                    "elapsedSeconds": round(elapsed),
                    "computeBackend": f"{torch.cuda.get_device_name(device)} · CUDA AMP / gsplat {'2DGS' if uses_2dgs else '3DGS'}",
                },
            )
        preview_due = step == start_step or step + 1 == iterations or (
            step > 0
            and step % 250 == 0
            and time.perf_counter() - last_live_preview_at >= 4.0
        )
        if preview_due:
            publish_live_preview()
        if step > 0 and step % 1000 == 0:
            save_checkpoint(step)

    expected_source_tiles = len(source_resolution_frame_order)
    source_resolution_coverage = len(source_resolution_tiles) / max(
        expected_source_tiles, 1
    )
    if source_resolution_coverage < 1.0:
        raise RuntimeError(
            "Gaussian optimization did not cover every calibrated source-resolution tile"
        )

    quality_report: dict[str, Any] = {
        "schemaVersion": 1,
        "status": "not_applicable" if metric_seeded else "evaluating",
        "reason": (
            "fixed calibrated metric surface"
            if metric_seeded
            else "bounded multiview training-camera photometric gate"
        ),
        "trainingColorSpace": "linear-srgb" if material_aware else "srgb",
        "metricColorSpace": "srgb",
        "thresholds": {
            "minimumMedianPsnrDb": MINIMUM_PRODUCTION_PSNR_DB,
            "minimumMedianSsim": MINIMUM_PRODUCTION_SSIM,
            "maximumMedianL1": MAXIMUM_PRODUCTION_L1,
        },
    }
    if not metric_seeded:
        evaluation_indices = np.linspace(
            0,
            len(frames) - 1,
            min(5, len(frames)),
            dtype=np.int64,
        )
        evaluation_records: list[dict[str, float | int]] = []
        with torch.no_grad():
            for evaluation_index in evaluation_indices:
                frame_index = int(evaluation_index)
                evaluation_frame = _frame_to_device(
                    _frame_tensors(root, frames[frame_index], training_dimension),
                    device,
                    linear_rgb=material_aware,
                )
                evaluation_camera_to_world = evaluation_frame["cameraToWorld"]
                if pose_refinement_enabled:
                    evaluation_camera_to_world = evaluation_camera_to_world @ pose_delta_matrix(
                        pose_offsets[frame_index] * pose_refinement_mask[frame_index]
                    )
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    evaluation_coefficients, evaluation_opacity, _evaluation_material = (
                        compose_material_render_state(
                            parameters, material_aware, SH_C0
                        )
                    )
                    evaluation_render, _evaluation_alpha, _evaluation_info = rasterization(
                        means=parameters["means"],
                        quats=torch.nn.functional.normalize(parameters["quats"], dim=-1),
                        scales=torch.exp(parameters["scales"]).clamp(1e-5, scene_scale),
                        opacities=evaluation_opacity,
                        colors=evaluation_coefficients,
                        viewmats=torch.linalg.inv(evaluation_camera_to_world).unsqueeze(0),
                        Ks=evaluation_frame["K"].unsqueeze(0),
                        width=evaluation_frame["width"],
                        height=evaluation_frame["height"],
                        packed=True,
                        render_mode="RGB",
                        sh_degree=3,
                    )
                    predicted = evaluation_render[0, ..., :3].clamp(0.0, 1.0)
                    target = evaluation_frame["rgb"]
                    quality_predicted = (
                        linear_to_srgb_tensor(predicted).clamp(0.0, 1.0)
                        if material_aware
                        else predicted
                    )
                    quality_target = (
                        linear_to_srgb_tensor(target).clamp(0.0, 1.0)
                        if material_aware
                        else target
                    )
                    difference = quality_predicted - quality_target
                    mse = float(difference.square().mean())
                    l1 = float(difference.abs().mean())
                    ssim = float(_ssim(quality_predicted, quality_target))
                evaluation_records.append(
                    {
                        "frameIndex": frame_index,
                        "psnrDb": -10.0 * math.log10(max(mse, 1e-10)),
                        "ssim": ssim,
                        "l1": l1,
                    }
                )
        quality_metrics = {
            "medianPsnrDb": float(
                np.median([record["psnrDb"] for record in evaluation_records])
            ),
            "medianSsim": float(
                np.median([record["ssim"] for record in evaluation_records])
            ),
            "medianL1": float(
                np.median([record["l1"] for record in evaluation_records])
            ),
        }
        quality_accepted = _photometric_quality_accepted(quality_metrics)
        quality_report.update(
            status="accepted" if quality_accepted else "rejected",
            metrics=quality_metrics,
            frames=evaluation_records,
        )
        _write_json_atomic(output_root / "splat-quality-report.json", quality_report)
        if not quality_accepted:
            save_checkpoint(iterations - 1)
            raise RuntimeError(
                "Gaussian candidate failed the production photometric quality gate; "
                "the final checkpoint was preserved for additional optimization"
            )
    else:
        _write_json_atomic(output_root / "splat-quality-report.json", quality_report)

    with torch.no_grad():
        _material_coefficients, optical_opacity, final_material = (
            compose_material_render_state(parameters, material_aware, SH_C0)
        )
        colors_out = _material_preview_colors(parameters, material_aware).cpu().numpy()
        diffuse_linear_out = torch.clamp(
            parameters["diffuse_sh0"][:, 0, :] * SH_C0 + 0.5,
            0.0,
            1.0,
        ).cpu().numpy()
        view_sh_out = parameters["view_shN"].cpu().numpy()
        geometric_opacity_out = torch.sigmoid(parameters["opacities"]).cpu().numpy()
        optical_opacity_out = optical_opacity.cpu().numpy()
        opacities_out = _logit(optical_opacity).cpu().numpy()
        if material_aware:
            emission_out = final_material["emission"].cpu().numpy()
            transmission_out = final_material["transmission"].cpu().numpy()
            roughness_out = final_material["roughness"].cpu().numpy()
            metallic_out = final_material["metallic"].cpu().numpy()
            material_confidence_out = final_material["confidence"].cpu().numpy()
            base_linear = np.maximum(diffuse_linear_out + emission_out, 0.0)
            display_derivative = np.where(
                base_linear <= 0.0031308,
                12.92,
                (1.055 / 2.4)
                * np.power(np.maximum(base_linear, 1e-6), 1.0 / 2.4 - 1.0),
            ).astype(np.float32)
            display_view_sh = view_sh_out * display_derivative[:, None, :]
        else:
            emission_out = np.zeros_like(diffuse_linear_out)
            transmission_out = np.zeros(len(diffuse_linear_out), dtype=np.float32)
            roughness_out = np.ones(len(diffuse_linear_out), dtype=np.float32)
            metallic_out = np.zeros(len(diffuse_linear_out), dtype=np.float32)
            material_confidence_out = np.zeros(len(diffuse_linear_out), dtype=np.float32)
            display_view_sh = view_sh_out
        # The standard PLY is a first-order display-space compatibility
        # projection. The adjacent NPZ retains the exact linear decomposition.
        display_dc = ((colors_out - 0.5) / SH_C0).astype(np.float32)[:, None, :]
        sh_out = np.concatenate((display_dc, display_view_sh), axis=1)
    means_out = parameters["means"].detach().cpu().numpy()
    scales_out = _preview_log_scales(
        parameters["scales"],
        flatten_2d=uses_2dgs,
    ).detach().cpu().numpy()
    quaternions_out = torch.nn.functional.normalize(parameters["quats"], dim=-1).detach().cpu().numpy()
    export_3dgs_ply(
        output_root / "room-splat.ply",
        means_out,
        colors_out,
        opacities_out,
        scales_out,
        quaternions_out,
        sh_coefficients=sh_out,
    )
    export_splat_preview(
        output_root / "room-splat.preview.splat",
        means_out,
        colors_out,
        opacities_out,
        scales_out,
        quaternions_out,
        limit=500_000,
    )
    if material_aware:
        export_material_gaussians(
            output_root / "room-splat-material.npz",
            diffuse_linear=diffuse_linear_out,
            view_sh_linear=view_sh_out,
            emission_linear=emission_out,
            transmission=transmission_out,
            roughness=roughness_out,
            metallic=metallic_out,
            confidence=material_confidence_out,
            geometric_opacity=geometric_opacity_out,
            optical_opacity=optical_opacity_out,
        )
    else:
        (output_root / "room-splat-material.npz").unlink(missing_ok=True)
    with torch.no_grad():
        pose_corrections = pose_delta_matrix(pose_offsets).detach().cpu().numpy()
        appearance_corrections = appearance_offsets.detach().cpu().numpy()
    pose_statistics = pose_correction_statistics(pose_offsets)
    pose_statistics.update(
        enabled=pose_refinement_enabled,
        startIteration=pose_refine_start if pose_refinement_enabled else None,
        regularization="identity and same-phase temporal smoothness",
    )
    refined_camera_records = []
    for frame, correction, appearance in zip(
        frames,
        pose_corrections,
        appearance_corrections,
        strict=True,
    ):
        original = np.asarray(frame["worldFromRgbCamera"], dtype=np.float64).reshape(4, 4)
        refined = original @ correction.astype(np.float64)
        refined_camera_records.append(
            {
                "image": frame.get("image"),
                "depth": frame.get("depth"),
                "phaseId": frame.get("phaseId"),
                "frameIndex": frame.get("frameIndex"),
                "timestampUs": frame.get("timestampUs"),
                "poseCorrectionLocalCamera": correction.reshape(-1).tolist(),
                "worldFromRgbCamera": refined.reshape(-1).tolist(),
                "appearanceLogGainRgb": appearance[:3].tolist(),
                "appearanceBiasRgb": appearance[3:].tolist(),
            }
        )
    _write_json_atomic(
        output_root / "room-splat-cameras.json",
        {
            "schemaVersion": 1,
            "sourceFingerprint": dataset.get("fingerprint", ""),
            "coordinateConvention": dataset.get("coordinateConvention", {}),
            "poseRefinement": pose_statistics,
            "frames": refined_camera_records,
        },
    )
    versions = {}
    for package in (
        "torch",
        "gsplat",
        "pycolmap",
        "av",
        "lingbot-map",
        "mdm",
        "flashinfer-python",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    versions["splatfactoReference"] = (
        "gsplat 1.5.3 native 2DGS with Splatfacto densification"
        if uses_2dgs
        else "gsplat 1.5.3 3DGS with Splatfacto densification"
    )
    training = {
        "trainerVersion": TRAINER_VERSION,
        "iterations": iterations,
        "requestedIterations": requested_iterations,
        "finalLoss": last_loss,
        "smoothedFinalLoss": smoothed_loss,
        "usesDepth": uses_depth,
        "device": torch.cuda.get_device_name(device),
        "vramGiB": round(vram_gib, 2),
        "trainingMaxDimension": training_dimension,
        "trainingRasterPolicy": "global bounded previews then calibrated source-resolution tiles",
        "sourceResolutionStartIteration": source_resolution_start_step,
        "sourceResolutionSteps": source_resolution_steps,
        "sourceResolutionTileCount": len(source_resolution_tiles),
        "expectedSourceResolutionTileCount": expected_source_tiles,
        "sourceResolutionCoverage": source_resolution_coverage,
        "photometricQuality": quality_report,
        "hostFrameCache": host_cache_size,
        "frameReusePerLoad": FRAME_REUSE_PER_LOAD,
        "maximumGaussians": maximum_gaussians,
        "densificationStoppedAt": densification_stopped_at,
        "precision": "CUDA AMP fp16 with TF32 matmul",
        "optimizer": "fused Adam",
        "representation": (
            "2D Gaussian surface discs" if uses_2dgs else "anisotropic 3D Gaussians"
        ),
        "rasterization": f"packed native gsplat {'2DGS' if uses_2dgs else '3DGS'}",
        "adaptiveStrategy": (
            "disabled; calibrated dense RGB-D surface is preserved"
            if metric_seeded
            else "bounded refinement around a dense LingBot geometry prior"
            if dense_geometry_prior
            else "bounded gsplat DefaultStrategy densification and pruning"
        ),
        "geometryOptimization": (
            "fixed metric RGB-D surface"
            if metric_seeded
            else "learned around confidence-gated LingBot depth"
            if dense_geometry_prior
            else "learned"
        ),
        "denseGeometryPrior": dense_geometry_prior,
        "directGaussianPrior": direct_gaussian_prior,
        "initializationContract": {
            "kind": str(initialization.kind),
            "representation": str(initialization.representation),
            "adaptiveDensification": initialization.adaptive_densification,
            "source": initialization.source,
        },
        "surfaceScaleMultiplier": (
            RGBD_SURFACE_SCALE_MULTIPLIER if metric_seeded else None
        ),
        "surfaceOpacity": RGBD_SURFACE_OPACITY if metric_seeded else None,
        "gaussianCount": len(parameters["means"]),
        "sphericalHarmonicsDegree": 3,
        "rgbLoss": (
            "linear-sRGB L1+SSIM with bounded per-view exposure compensation"
            if material_aware
            else "L1+SSIM with bounded per-view exposure compensation"
        ),
        "depthLoss": "masked robust Huber with annealed weight" if uses_depth else "disabled",
        "normalLoss": (
            "disabled for fixed metric surface"
            if metric_seeded
            else "rendered-normal/depth-normal consistency"
            if uses_2dgs
            else "disabled for photoreal 3DGS"
        ),
        "distortionLoss": (
            "disabled for fixed metric surface"
            if metric_seeded
            else "2DGS ray-splat distortion regularization"
            if uses_2dgs
            else "disabled for photoreal 3DGS"
        ),
        "frameSampling": "deterministically shuffled cache-local keyframe blocks",
        "poseRefinement": pose_statistics,
        "appearanceOptimization": {
            "enabled": appearance_optimization_enabled,
            "anchorFrameIndex": appearance_anchor_index if appearance_optimization_enabled else None,
            "model": "per-view RGB log-gain and bias",
        },
        "materialDecomposition": {
            "contract": "scanlan-gaussian-material-v1",
            "enabled": material_aware,
            "priorAvailable": material_aware,
            "trainingColorSpace": "linear-srgb" if material_aware else "srgb compatibility",
            "components": ["diffuse", "view-dependent", "emissive", "transmissive"],
            "opacity": "geometric occupancy separated from optical transmission",
            "specularGradientGate": material_aware,
            "meanTransmission": float(np.mean(transmission_out)),
            "emissiveGaussianCount": int(
                np.count_nonzero(np.max(emission_out, axis=1) > 1e-3)
            ),
            "supportedGaussianCount": int(
                np.count_nonzero(material_confidence_out > 0.0)
            ),
        },
        "sourceQuality": dataset.get("quality"),
    }
    material_manifest = {
        "schemaVersion": 1,
        "contract": "scanlan-gaussian-material-v1",
        "path": "room-splat-material.npz",
        "alignedWith": "room-splat.ply vertex order",
        "colorSpace": "linear-srgb",
        "components": {
            "diffuse": "diffuse_linear Nx3",
            "viewDependent": "view_sh_linear Nx15x3",
            "emissive": "emission_linear Nx3",
            "transmissive": "transmission N",
        },
        "opacity": {
            "geometric": "geometric_opacity N",
            "optical": "optical_opacity = geometric_opacity * (1 - transmission)",
        },
        "priorAvailable": material_aware,
        "plyCompatibility": "first-order linear-to-sRGB projection; NPZ is lossless",
    }
    write_splat_sidecars(
        output_root,
        dataset.get("fingerprint", ""),
        bool(dataset.get("metric")),
        versions,
        training,
        material=material_manifest if material_aware else None,
    )
    project_path = project_root / "project.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    project.setdefault("artifacts", {})["gaussianSplat"] = {
        "path": "outputs/room-splat.ply",
        "refinedCameraPath": "outputs/room-splat-cameras.json",
        "status": "ready",
        "sourceFingerprint": dataset.get("fingerprint", ""),
        "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metric": bool(dataset.get("metric")),
        "stale": False,
    }
    if material_aware:
        project["artifacts"]["gaussianSplat"]["materialPath"] = (
            "outputs/room-splat-material.npz"
        )
    project["processingStatus"] = "complete"
    project.pop("processingError", None)
    _write_json_atomic(project_path, project)
    checkpoint_path.unlink(missing_ok=True)
    published_description = (
        "material-aware Gaussian components"
        if material_aware
        else "calibrated RGB-D splats"
        if metric_seeded
        else "optimized 2D Gaussians" if uses_2dgs else "photoreal 3D Gaussians"
    )
    _write_json_atomic(progress_path, {"stage": "complete", "detail": f"Published {len(parameters['means']):,} {published_description}", "progress": 1.0, "stageProgress": 1.0, "iteration": iterations, "totalIterations": iterations, "loss": last_loss, "smoothedLoss": smoothed_loss, "etaSeconds": 0, "stageEtaSeconds": 0, "elapsedSeconds": round(time.perf_counter() - started), "computeBackend": f"{torch.cuda.get_device_name(device)} · CUDA AMP / gsplat {'2DGS' if uses_2dgs else '3DGS'}"})
    return training
