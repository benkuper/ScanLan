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
from .export import export_3dgs_ply, write_splat_sidecars


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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
    value: dict[str, Any] = {"rgb": rgb, "K": intrinsic, "view": torch.linalg.inv(world_from_camera), "width": image.width, "height": image.height}
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
    from gsplat import rasterization
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
    output_root = project_root / "outputs"
    checkpoint_path = output_root / "splat-checkpoint.pt"
    progress_path = output_root / "splat-progress.json"
    points, colors = _read_initialization(root / dataset.get("initialization", "initialization.ply"))
    if len(points) == 0:
        raise ValueError("Sparse or RGB-D initialization contains no points")

    centred = points - np.median(points, axis=0, keepdims=True)
    scene_scale = max(float(np.percentile(np.linalg.norm(centred, axis=1), 90)), 1e-3)
    initial_scale = max(scene_scale * 0.003, 0.004 if dataset.get("metric") else scene_scale * 0.001)
    parameter_values = {
        "means": torch.from_numpy(points),
        "scales": torch.full((len(points), 3), math.log(initial_scale), dtype=torch.float32),
        "quats": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32).repeat(len(points), 1),
        "opacities": torch.full((len(points),), float(_logit(0.1)), dtype=torch.float32),
        "colors": _logit(torch.from_numpy(colors)),
    }
    start_step = 0
    checkpoint = None
    if resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("fingerprint") != dataset.get("fingerprint"):
            raise RuntimeError("Splat checkpoint source fingerprint no longer matches")
        if set(checkpoint.get("parameters", {})) != set(parameter_values):
            raise RuntimeError("Splat checkpoint belongs to an incompatible trainer version; start a new job")
        parameter_values = checkpoint["parameters"]
        start_step = int(checkpoint["step"]) + 1
    parameters = torch.nn.ParameterDict(
        {name: torch.nn.Parameter(value.to(device)) for name, value in parameter_values.items()}
    )
    learning_rates = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "colors": 2.5e-3,
    }
    optimizers = {
        name: torch.optim.Adam(
            [{"params": [parameters[name]], "lr": learning_rate, "name": name}],
            eps=1e-15,
            fused=True,
        )
        for name, learning_rate in learning_rates.items()
    }
    strategy = DefaultStrategy(
        absgrad=True,
        grow_grad2d=0.0008,
        refine_stop_iter=min(15_000, max(1_000, iterations // 2)),
        pause_refine_after_reset=len(frames),
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

    def save_checkpoint(step: int) -> None:
        torch.save(
            {
                "step": step,
                "fingerprint": dataset.get("fingerprint"),
                "parameters": {key: value.detach().cpu() for key, value in parameters.items()},
                "optimizers": {key: value.state_dict() for key, value in optimizers.items()},
                "strategy": strategy_state,
                "scaler": scaler.state_dict(),
            },
            checkpoint_path,
        )

    cached_frames: dict[int, dict[str, Any]] = {}
    started = time.perf_counter()
    last_loss = 0.0
    uses_depth = any(frame.get("depth") for frame in frames)
    for step in range(start_step, iterations):
        if (output_root / "cancel.flag").exists():
            save_checkpoint(step)
            raise RuntimeError("Gaussian training cancelled; checkpoint saved")
        frame_index = step % len(frames)
        if frame_index not in cached_frames:
            cached_frames[frame_index] = _frame_tensors(root, frames[frame_index], device)
        frame = cached_frames[frame_index]
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            rendering, _, info = rasterization(
                means=parameters["means"],
                quats=torch.nn.functional.normalize(parameters["quats"], dim=-1),
                scales=torch.exp(parameters["scales"]).clamp(1e-5, scene_scale),
                opacities=torch.sigmoid(parameters["opacities"]),
                colors=torch.sigmoid(parameters["colors"]),
                viewmats=frame["view"].unsqueeze(0),
                Ks=frame["K"].unsqueeze(0),
                width=frame["width"],
                height=frame["height"],
                packed=True,
                absgrad=True,
                render_mode="RGB+ED" if uses_depth else "RGB",
            )
            strategy.step_pre_backward(parameters, optimizers, strategy_state, step, info)
            predicted_rgb = rendering[0, ..., :3].clamp(0.0, 1.0)
            rgb_l1 = torch.mean(torch.abs(predicted_rgb - frame["rgb"]))
            loss = 0.8 * rgb_l1 + 0.2 * (1.0 - _ssim(predicted_rgb, frame["rgb"]))
            depth_value = torch.zeros((), device=device)
            if "depth" in frame and rendering.shape[-1] > 3:
                depth_value = masked_robust_depth_loss(rendering[0, ..., 3], frame["depth"], frame["mask"])
                loss = loss + depth_weight(step, iterations) * depth_value
        scaler.scale(loss).backward()
        gradient_scale = scaler.get_scale()
        means_2d = info.get("means2d")
        if means_2d is not None:
            if means_2d.grad is not None:
                means_2d.grad.div_(gradient_scale)
            absolute_gradient = getattr(means_2d, "absgrad", None)
            if absolute_gradient is not None:
                absolute_gradient.div_(gradient_scale)
        for optimizer in optimizers.values():
            scaler.step(optimizer)
        scaler.update()
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
            _write_json_atomic(progress_path, {"stage": "splat_training", "detail": f"Optimized iteration {step + 1:,} of {iterations:,} · loss {last_loss:.4f}", "progress": progress_start + (1.0 - progress_start) * stage_progress, "stageProgress": stage_progress, "iteration": step + 1, "totalIterations": iterations, "loss": last_loss, "rgbLoss": float(rgb_l1.detach()), "depthLoss": float(depth_value.detach()), "etaSeconds": eta, "stageEtaSeconds": eta, "elapsedSeconds": round(elapsed), "computeBackend": f"{torch.cuda.get_device_name(device)} · CUDA AMP / gsplat"})
        if step > 0 and step % 1000 == 0:
            save_checkpoint(step)

    colors_out = torch.sigmoid(parameters["colors"]).detach().cpu().numpy()
    export_3dgs_ply(
        output_root / "room-splat.ply",
        parameters["means"].detach().cpu().numpy(),
        colors_out,
        parameters["opacities"].detach().cpu().numpy(),
        parameters["scales"].detach().cpu().numpy(),
        torch.nn.functional.normalize(parameters["quats"], dim=-1).detach().cpu().numpy(),
    )
    versions = {}
    for package in ("torch", "gsplat"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    versions["splatfactoReference"] = "Nerfstudio 1.1.5 strategy adapted to gsplat 1.5.3"
    training = {"iterations": iterations, "finalLoss": last_loss, "usesDepth": uses_depth, "device": torch.cuda.get_device_name(device), "precision": "CUDA AMP fp16 with TF32 matmul", "optimizer": "fused Adam", "rasterization": "packed", "adaptiveStrategy": "gsplat DefaultStrategy with AbsGrad densification/pruning", "gaussianCount": len(parameters["means"]), "rgbLoss": "L1+SSIM", "depthLoss": "masked robust Huber with annealed weight"}
    write_splat_sidecars(output_root, dataset.get("fingerprint", ""), bool(dataset.get("metric")), versions, training)
    project_path = project_root / "project.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    project.setdefault("artifacts", {})["gaussianSplat"] = {"path": "outputs/room-splat.ply", "status": "ready", "sourceFingerprint": dataset.get("fingerprint", ""), "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "metric": bool(dataset.get("metric")), "stale": False}
    _write_json_atomic(project_path, project)
    checkpoint_path.unlink(missing_ok=True)
    _write_json_atomic(progress_path, {"stage": "complete", "detail": f"Published {len(parameters['means']):,} optimized Gaussians", "progress": 1.0, "stageProgress": 1.0, "iteration": iterations, "totalIterations": iterations, "loss": last_loss, "etaSeconds": 0, "stageEtaSeconds": 0, "elapsedSeconds": round(time.perf_counter() - started), "computeBackend": f"{torch.cuda.get_device_name(device)} · CUDA AMP / gsplat"})
    return training
