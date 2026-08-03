from __future__ import annotations

import importlib.metadata
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .dataset import load_dataset
from .depth_loss import depth_weight, masked_robust_depth_loss
from .export import SH_C0, export_3dgs_ply, export_splat_preview, write_splat_sidecars
from .pose import (
    constrain_pose_offsets_,
    pose_correction_statistics,
    pose_delta_matrix,
    pose_regularization,
)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    try:
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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        save(value, temporary)
        os.replace(temporary, path)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    points, colors = _read_initialization(
        root / dataset.get("initialization", "initialization.ply")
    )
    parameters_path = dataset.get("initializationParameters")
    if not parameters_path:
        return points, colors, None, None
    path = root / parameters_path
    if not path.is_file():
        return points, colors, None, None
    with np.load(path, allow_pickle=False) as values:
        seeded_points = np.asarray(values["points"], dtype=np.float32)
        seeded_colors = np.asarray(values["colors"], dtype=np.float32)
        seeded_scales = np.asarray(values["scales"], dtype=np.float32)
        seeded_quaternions = np.asarray(values["quaternions"], dtype=np.float32)
    if seeded_colors.size and seeded_colors.max() > 1.0:
        seeded_colors /= 255.0
    count = len(seeded_points)
    if (
        seeded_points.shape != (count, 3)
        or seeded_colors.shape != (count, 3)
        or seeded_scales.shape != (count, 3)
        or seeded_quaternions.shape != (count, 4)
        or not all(
            np.isfinite(value).all()
            for value in (seeded_points, seeded_colors, seeded_scales, seeded_quaternions)
        )
        or np.any(seeded_scales <= 0.0)
    ):
        raise ValueError("RGB-D Gaussian initialization parameters are invalid")
    return seeded_points, seeded_colors, seeded_scales, seeded_quaternions


def _sh_preview_colors(sh0: Any) -> Any:
    import torch

    return torch.clamp(sh0[:, 0, :] * SH_C0 + 0.5, 0.0, 1.0)


def _preview_log_scales(log_scales: Any) -> Any:
    """Flatten 2D discs only when exporting to the legacy 3DGS viewer format."""
    import torch

    scales = torch.exp(log_scales)
    thickness = torch.clamp(torch.amin(scales[:, :2], dim=1, keepdim=True) * 0.08, min=5e-4)
    return torch.log(torch.cat((scales[:, :2], thickness), dim=1))


def _logit(value: Any) -> Any:
    import torch

    value = torch.as_tensor(value).clamp(1e-4, 1 - 1e-4)
    return torch.log(value / (1 - value))


def _ssim(predicted: Any, target: Any) -> Any:
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
    return score.mean()


def _frame_tensors(root: Path, frame: dict[str, Any], device: Any, max_dimension: int = 960) -> dict[str, Any]:
    import torch

    with Image.open(root / frame["image"]) as source:
        image = source.convert("RGB")
        scale = min(1.0, max_dimension / max(image.size))
        if scale < 1.0:
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
        rgb = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).to(device) / 255.0
    intrinsics = frame["intrinsics"]
    intrinsic = torch.tensor(
        [[intrinsics["fx"] * scale, 0.0, intrinsics["cx"] * scale], [0.0, intrinsics["fy"] * scale, intrinsics["cy"] * scale], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    world_from_camera = torch.tensor(frame["worldFromRgbCamera"], dtype=torch.float32, device=device).reshape(4, 4)
    value: dict[str, Any] = {
        "rgb": rgb,
        "K": intrinsic,
        "cameraToWorld": world_from_camera,
        "view": torch.linalg.inv(world_from_camera),
        "width": image.width,
        "height": image.height,
    }
    if frame.get("depth"):
        with Image.open(root / frame["depth"]) as source:
            depth_image = source.resize((image.width, image.height), Image.Resampling.NEAREST)
            depth = torch.from_numpy(np.asarray(depth_image, dtype=np.float32).copy()).to(device) / 1000.0
        with Image.open(root / frame["depthMask"]) as source:
            mask_image = source.resize((image.width, image.height), Image.Resampling.NEAREST)
            mask = torch.from_numpy(np.asarray(mask_image, dtype=np.uint8).copy()).to(device) > 0
        value.update(depth=depth, mask=mask)
    return value


def train_dataset(
    dataset_path: Path,
    project_root: Path,
    iterations: int = 30_000,
    resume: bool = False,
    progress_start: float = 0.0,
) -> dict[str, Any]:
    import torch
    from gsplat.rendering import rasterization_2dgs
    from gsplat.strategy import DefaultStrategy

    if not torch.cuda.is_available():
        raise RuntimeError("Gaussian training requires a CUDA-capable PyTorch runtime; current ScanLan features remain available on CPU")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    root, dataset = load_dataset(dataset_path)
    frames = dataset.get("frames", [])
    if len(frames) < 2:
        raise ValueError("At least two registered RGB views are required for Gaussian training")
    uses_depth = any(frame.get("depth") for frame in frames)
    pose_refinement_enabled = bool(
        dataset.get("metric")
        and all(frame.get("depth") and frame.get("depthMask") for frame in frames)
    )
    output_root = project_root / "outputs"
    checkpoint_path = output_root / "splat-checkpoint.pt"
    progress_path = output_root / "splat-progress.json"
    points, colors, seeded_scales, seeded_quaternions = _read_seed_parameters(root, dataset)
    if len(points) == 0:
        raise ValueError("Sparse or RGB-D initialization contains no points")

    centred = points - np.median(points, axis=0, keepdims=True)
    scene_scale = max(float(np.percentile(np.linalg.norm(centred, axis=1), 90)), 1e-3)
    initial_scale = max(scene_scale * 0.003, 0.004 if dataset.get("metric") else scene_scale * 0.001)
    scales = (
        np.log(np.maximum(seeded_scales, 1e-5))
        if seeded_scales is not None
        else np.full((len(points), 3), math.log(initial_scale), dtype=np.float32)
    )
    quaternions = (
        seeded_quaternions
        if seeded_quaternions is not None
        else np.random.default_rng(0).normal(size=(len(points), 4)).astype(np.float32)
    )
    sh0 = ((colors - 0.5) / SH_C0).astype(np.float32)[:, None, :]
    parameter_values = {
        "means": torch.from_numpy(points),
        "scales": torch.from_numpy(np.asarray(scales, dtype=np.float32)),
        "quats": torch.from_numpy(np.asarray(quaternions, dtype=np.float32)),
        "opacities": torch.full((len(points),), float(_logit(0.5)), dtype=torch.float32),
        "sh0": torch.from_numpy(sh0),
        "shN": torch.zeros((len(points), 15, 3), dtype=torch.float32),
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
        if checkpoint.get("fingerprint") != dataset.get("fingerprint"):
            raise RuntimeError("Splat checkpoint source fingerprint no longer matches")
        if set(checkpoint.get("parameters", {})) != set(parameter_values):
            raise RuntimeError("Splat checkpoint belongs to an incompatible trainer version; start a new job")
        parameter_values = checkpoint["parameters"]
        start_step = int(checkpoint["step"]) + 1
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
    learning_rates = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 1.25e-4,
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
    strategy = DefaultStrategy(
        # gsplat's 2DGS densification signal is `gradient_2dgs`; unlike the
        # projected 3DGS means, it does not expose an AbsGrad accumulator.
        absgrad=False,
        grow_grad2d=0.0008,
        refine_stop_iter=min(15_000, max(1_000, iterations // 2)),
        pause_refine_after_reset=len(frames),
        key_for_gradient="gradient_2dgs",
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

    def save_checkpoint(step: int) -> None:
        _write_checkpoint_atomic(
            checkpoint_path,
            {
                "step": step,
                "fingerprint": dataset.get("fingerprint"),
                "parameters": {key: value.detach().cpu() for key, value in parameters.items()},
                "optimizers": {key: value.state_dict() for key, value in optimizers.items()},
                "poseOffsets": pose_offsets.detach().cpu() if pose_refinement_enabled else None,
                "poseOptimizer": pose_optimizer.state_dict() if pose_optimizer is not None else None,
                "strategy": strategy_state,
                "scaler": scaler.state_dict(),
            },
            torch.save,
        )

    cached_frames: dict[int, dict[str, Any]] = {}
    started = time.perf_counter()
    last_live_preview_at = 0.0
    last_loss = 0.0
    frame_epoch = -1
    frame_order = np.arange(len(frames), dtype=np.int64)

    def publish_live_preview() -> None:
        nonlocal last_live_preview_at
        with torch.no_grad():
            preview_count = min(len(parameters["means"]), 200_000)
            preview_indices = (
                torch.linspace(0, len(parameters["means"]) - 1, preview_count, device=device).long()
                if len(parameters["means"]) > preview_count
                else slice(None)
            )
            export_splat_preview(
                output_root / "room-splat.preview.splat",
                parameters["means"][preview_indices].detach().cpu().numpy(),
                _sh_preview_colors(parameters["sh0"][preview_indices]).detach().cpu().numpy(),
                parameters["opacities"][preview_indices].detach().cpu().numpy(),
                _preview_log_scales(parameters["scales"][preview_indices]).detach().cpu().numpy(),
                torch.nn.functional.normalize(parameters["quats"][preview_indices], dim=-1).detach().cpu().numpy(),
            )
        last_live_preview_at = time.perf_counter()

    for step in range(start_step, iterations):
        if (output_root / "cancel.flag").exists():
            save_checkpoint(step)
            raise RuntimeError("Gaussian training cancelled; checkpoint saved")
        next_epoch = step // len(frames)
        if next_epoch != frame_epoch:
            frame_epoch = next_epoch
            frame_order = np.random.default_rng(frame_epoch).permutation(len(frames))
        frame_index = int(frame_order[step % len(frames)])
        if frame_index not in cached_frames:
            cached_frames[frame_index] = _frame_tensors(root, frames[frame_index], device)
        frame = cached_frames[frame_index]
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        pose_active = pose_optimizer is not None and step >= pose_refine_start
        if pose_optimizer is not None:
            pose_optimizer.zero_grad(set_to_none=True)
        refined_view = frame["view"]
        if pose_active:
            correction = pose_delta_matrix(pose_offsets[frame_index])
            refined_camera_to_world = frame["cameraToWorld"] @ correction
            refined_view = torch.linalg.inv(refined_camera_to_world)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            sh_degree = min(3, step // max(iterations // 4, 1))
            pose_regularization_value = torch.zeros((), device=device)
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
                quats=torch.nn.functional.normalize(parameters["quats"], dim=-1),
                scales=torch.exp(parameters["scales"]).clamp(1e-5, scene_scale),
                opacities=torch.sigmoid(parameters["opacities"]),
                colors=torch.cat((parameters["sh0"], parameters["shN"]), dim=1),
                viewmats=refined_view.unsqueeze(0),
                Ks=frame["K"].unsqueeze(0),
                width=frame["width"],
                height=frame["height"],
                packed=True,
                absgrad=False,
                # Expected depth is also required for 2DGS distortion and
                # surface-normal regularization on RGB-only datasets.
                render_mode="RGB+ED",
                sh_degree=sh_degree,
                distloss=True,
                depth_mode="expected",
            )
            strategy.step_pre_backward(parameters, optimizers, strategy_state, step, info)
            predicted_rgb = rendering[0, ..., :3].clamp(0.0, 1.0)
            rgb_l1 = torch.mean(torch.abs(predicted_rgb - frame["rgb"]))
            loss = 0.8 * rgb_l1 + 0.2 * (1.0 - _ssim(predicted_rgb, frame["rgb"]))
            depth_value = torch.zeros((), device=device)
            normal_value = torch.zeros((), device=device)
            distortion_value = torch.zeros((), device=device)
            if "depth" in frame and rendering.shape[-1] > 3:
                depth_value = masked_robust_depth_loss(rendering[0, ..., 3], frame["depth"], frame["mask"])
                loss = loss + depth_weight(step, iterations) * depth_value
            if step >= max(500, iterations // 10):
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
            if step >= max(250, iterations // 20):
                distortion_value = render_distortion.mean()
                loss = loss + 0.01 * distortion_value
            if pose_active:
                pose_regularization_value = pose_regularization(
                    pose_offsets,
                    neighbor_pairs,
                )
                loss = loss + 1e-4 * pose_regularization_value
        scaler.scale(loss).backward()
        gradient_scale = scaler.get_scale()
        projected_gradient = info.get("gradient_2dgs")
        if projected_gradient is not None and projected_gradient.grad is not None:
            projected_gradient.grad.div_(gradient_scale)
        for optimizer in optimizers.values():
            scaler.step(optimizer)
        if pose_active:
            scaler.step(pose_optimizer)
        scaler.update()
        if pose_active:
            constrain_pose_offsets_(pose_offsets)
        strategy.step_post_backward(
            parameters,
            optimizers,
            strategy_state,
            step,
            info,
            packed=True,
        )
        last_loss = float(loss.detach())
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
            _write_json_atomic(progress_path, {"stage": "splat_training", "detail": f"Optimized 2DGS iteration {step + 1:,} of {iterations:,} · loss {last_loss:.4f}", "progress": progress_start + (1.0 - progress_start) * stage_progress, "stageProgress": stage_progress, "iteration": step + 1, "totalIterations": iterations, "loss": last_loss, "rgbLoss": float(rgb_l1.detach()), "depthLoss": float(depth_value.detach()), "normalLoss": float(normal_value.detach()), "distortionLoss": float(distortion_value.detach()), "poseRegularizationLoss": float(pose_regularization_value.detach()), "maximumPoseTranslationMm": maximum_pose_translation_mm, "etaSeconds": eta, "stageEtaSeconds": eta, "elapsedSeconds": round(elapsed), "computeBackend": f"{torch.cuda.get_device_name(device)} · CUDA AMP / gsplat 2DGS"})
        preview_due = step == start_step or step + 1 == iterations or (
            step > 0
            and step % 250 == 0
            and time.perf_counter() - last_live_preview_at >= 4.0
        )
        if preview_due:
            publish_live_preview()
        if step > 0 and step % 1000 == 0:
            save_checkpoint(step)

    colors_out = _sh_preview_colors(parameters["sh0"]).detach().cpu().numpy()
    sh_out = torch.cat((parameters["sh0"], parameters["shN"]), dim=1).detach().cpu().numpy()
    means_out = parameters["means"].detach().cpu().numpy()
    opacities_out = parameters["opacities"].detach().cpu().numpy()
    scales_out = _preview_log_scales(parameters["scales"]).detach().cpu().numpy()
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
    with torch.no_grad():
        pose_corrections = pose_delta_matrix(pose_offsets).detach().cpu().numpy()
    pose_statistics = pose_correction_statistics(pose_offsets)
    pose_statistics.update(
        enabled=pose_refinement_enabled,
        startIteration=pose_refine_start if pose_refinement_enabled else None,
        regularization="identity and same-phase temporal smoothness",
    )
    refined_camera_records = []
    for frame, correction in zip(frames, pose_corrections, strict=True):
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
    for package in ("torch", "gsplat"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    versions["splatfactoReference"] = "gsplat 1.5.3 native 2DGS with Splatfacto densification"
    training = {"iterations": iterations, "finalLoss": last_loss, "usesDepth": uses_depth, "device": torch.cuda.get_device_name(device), "precision": "CUDA AMP fp16 with TF32 matmul", "optimizer": "fused Adam", "representation": "2D Gaussian surface discs", "rasterization": "packed native gsplat 2DGS", "adaptiveStrategy": "gsplat DefaultStrategy with native 2DGS densification gradient and pruning", "gaussianCount": len(parameters["means"]), "sphericalHarmonicsDegree": 3, "rgbLoss": "L1+SSIM", "depthLoss": "masked robust Huber with annealed weight", "normalLoss": "rendered-normal/depth-normal consistency", "distortionLoss": "2DGS ray-splat distortion regularization", "frameSampling": "deterministically shuffled keyframe epochs", "poseRefinement": pose_statistics}
    write_splat_sidecars(output_root, dataset.get("fingerprint", ""), bool(dataset.get("metric")), versions, training)
    project_path = project_root / "project.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    project.setdefault("artifacts", {})["gaussianSplat"] = {"path": "outputs/room-splat.ply", "refinedCameraPath": "outputs/room-splat-cameras.json", "status": "ready", "sourceFingerprint": dataset.get("fingerprint", ""), "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "metric": bool(dataset.get("metric")), "stale": False}
    _write_json_atomic(project_path, project)
    checkpoint_path.unlink(missing_ok=True)
    _write_json_atomic(progress_path, {"stage": "complete", "detail": f"Published {len(parameters['means']):,} optimized 2D Gaussians", "progress": 1.0, "stageProgress": 1.0, "iteration": iterations, "totalIterations": iterations, "loss": last_loss, "etaSeconds": 0, "stageEtaSeconds": 0, "elapsedSeconds": round(time.perf_counter() - started), "computeBackend": f"{torch.cuda.get_device_name(device)} · CUDA AMP / gsplat 2DGS"})
    return training
