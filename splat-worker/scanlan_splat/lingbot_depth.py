from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


LINGBOT_DEPTH_CODE_REVISION = "f3a237e434ae987bc38281476d6cfb5df3e4d739"
LINGBOT_DEPTH_MODEL_REPOSITORY = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"
LINGBOT_DEPTH_MODEL_REVISION = "79204ed6b837f4fdd192cf563e59481fecfa0295"
LINGBOT_DEPTH_MODEL_FILENAME = "lingbot-depth-v0.5.pt"
LINGBOT_DEPTH_MODEL_SHA256 = "b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23"
LINGBOT_DEPTH_REQUEST_SCHEMA = 1


class DepthPredictor(Protocol):
    backend: str

    def infer(
        self,
        color: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]: ...


def _runtime_asset(filename: str, environment_name: str) -> Path:
    configured = os.environ.get(environment_name)
    executable_root = Path(sys.executable).resolve().parent
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
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
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"{filename} is not installed; run npm run prepare:splat or set {environment_name}"
    )


def resolve_lingbot_depth_model(*, verify: bool = True) -> Path:
    path = _runtime_asset(
        LINGBOT_DEPTH_MODEL_FILENAME,
        "SCANLAN_LINGBOT_DEPTH_MODEL",
    )
    if verify:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != LINGBOT_DEPTH_MODEL_SHA256:
            raise RuntimeError(
                "The installed LingBot-Depth checkpoint does not match ScanLan's pinned digest"
            )
    return path


def lingbot_depth_runtime_status(
    *,
    verify_model: bool = False,
    smoke_test: bool = False,
) -> dict[str, Any]:
    import importlib.util

    try:
        package_available = importlib.util.find_spec("mdm.model.v2") is not None
    except (ImportError, ModuleNotFoundError):
        package_available = False
    model_path: str | None = None
    error: str | None = None
    runtime_validated = False
    backend: str | None = None
    try:
        model_path = str(resolve_lingbot_depth_model(verify=verify_model))
    except (FileNotFoundError, RuntimeError) as caught:
        error = str(caught)
    if smoke_test and package_available and model_path is not None:
        try:
            predictor = LingbotDepthPredictor.load()
            height, width = 24, 32
            color = np.full((height, width, 3), 127, dtype=np.uint8)
            depth = np.full((height, width), 2.0, dtype=np.float32)
            depth[8:16, 11:21] = 0.0
            intrinsics = np.asarray(
                [[0.9, 0.0, 0.5], [0.0, 1.2, 0.5], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            prediction, mask = predictor.infer(color, depth, intrinsics)
            if prediction.shape != depth.shape or mask.shape != depth.shape:
                raise RuntimeError("LingBot-Depth smoke test returned a misaligned raster")
            if not np.any(mask & np.isfinite(prediction) & (prediction > 0.0)):
                raise RuntimeError("LingBot-Depth smoke test returned no valid metric depth")
            runtime_validated = True
            backend = predictor.backend
        except Exception as caught:  # Diagnostic boundary returns a structured failure.
            error = str(caught)
    return {
        "available": package_available
        and model_path is not None
        and (runtime_validated or not smoke_test),
        "packageAvailable": package_available,
        "modelPath": model_path,
        "runtimeValidated": runtime_validated,
        "backend": backend,
        "codeRevision": LINGBOT_DEPTH_CODE_REVISION,
        "modelRevision": LINGBOT_DEPTH_MODEL_REVISION,
        "modelSha256": LINGBOT_DEPTH_MODEL_SHA256,
        "error": error,
    }


@dataclass
class LingbotDepthPredictor:
    model: Any
    torch: Any
    device: Any
    use_mixed_precision: bool
    backend: str

    @classmethod
    def load(cls) -> "LingbotDepthPredictor":
        # The release optionally imports xFormers built for Torch 2.6. ScanLan
        # uses native SDPA in its newer isolated CUDA runtime instead.
        os.environ.setdefault("XFORMERS_DISABLED", "1")
        import torch
        from mdm.model.v2 import MDMModel

        if not torch.cuda.is_available():
            raise RuntimeError("LingBot-Depth refinement requires CUDA")
        device = torch.device("cuda")
        model_path = resolve_lingbot_depth_model(verify=True)
        model = MDMModel.from_pretrained(model_path)
        model.enable_pytorch_native_sdpa()
        _enable_single_frame_nested_sdpa(model)
        model = model.to(device).eval()
        use_mixed_precision = bool(torch.cuda.is_bf16_supported())
        precision = "BF16" if use_mixed_precision else "FP32"
        return cls(
            model=model,
            torch=torch,
            device=device,
            use_mixed_precision=use_mixed_precision,
            backend=f"LingBot-Depth v0.5 / PyTorch SDPA {precision} / {torch.cuda.get_device_name(device)}",
        )

    def infer(
        self,
        color: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        image = torch.from_numpy(np.ascontiguousarray(color)).to(
            self.device, dtype=torch.float32
        )
        image = image.permute(2, 0, 1).div_(255.0)
        depth = torch.from_numpy(np.ascontiguousarray(depth_m)).to(
            self.device, dtype=torch.float32
        )
        normalized_intrinsics = torch.from_numpy(
            np.ascontiguousarray(intrinsics, dtype=np.float32)
        ).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.model.infer(
                image,
                depth_in=depth,
                intrinsics=normalized_intrinsics,
                apply_mask=True,
                # Upstream's historical flag name selects BF16 autocast.
                use_fp16=self.use_mixed_precision,
            )
        prediction = output["depth"].float().cpu().numpy()
        mask_value = output.get("mask")
        mask = (
            np.ones(prediction.shape, dtype=bool)
            if mask_value is None
            else mask_value.bool().cpu().numpy()
        )
        del image, depth, normalized_intrinsics, output
        return prediction.astype(np.float32, copy=False), mask.astype(bool, copy=False)


def _enable_single_frame_nested_sdpa(model: Any) -> None:
    """Preserve invalid-depth token masking without requiring xFormers.

    LingBot's DINO encoder represents each batch item as a separate token list
    after dropping invalid depth patches. Its upstream nested block needs an
    xFormers block-diagonal bias. ScanLan deliberately infers one frame at a
    time, where that bias is mathematically identical to ordinary attention on
    the one remaining sequence. Running the base block on that sequence keeps
    the trained masking semantics and uses PyTorch's native SDPA kernels.
    """
    from mdm.model.dinov2_rgbd.layers.block import Block

    def forward_single_sequence(block: Any, value: Any) -> Any:
        if isinstance(value, list):
            if len(value) != 1:
                raise RuntimeError(
                    "ScanLan's native-SDPA LingBot path requires one frame per inference"
                )
            return [Block.forward(block, value[0])]
        return Block.forward(block, value)

    for block in model.encoder.backbone.blocks:
        block.forward = types.MethodType(forward_single_sequence, block)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2)
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


def _load_frame(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = int(frame["width"])
    height = int(frame["height"])
    color_path = Path(frame["colorPath"])
    depth_path = Path(frame["depthPath"])
    expected_pixels = width * height
    color = np.fromfile(color_path, dtype=np.uint8)
    depth = np.fromfile(depth_path, dtype="<u2")
    if color.size != expected_pixels * 3:
        raise ValueError(f"Aligned RGB frame {color_path} has an unexpected size")
    if depth.size != expected_pixels:
        raise ValueError(f"Depth frame {depth_path} has an unexpected size")
    color = color.reshape(height, width, 3)
    depth_m = depth.reshape(height, width).astype(np.float32) / float(frame["depthScale"])
    intrinsics = np.asarray(
        [
            [float(frame["fx"]) / width, 0.0, float(frame["cx"]) / width],
            [0.0, float(frame["fy"]) / height, float(frame["cy"]) / height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return color, depth_m, intrinsics


def refine_depth_request(
    request_path: Path,
    progress_path: Path,
    *,
    predictor: DepthPredictor | None = None,
) -> dict[str, Any]:
    request_path = request_path.resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if int(request.get("schemaVersion", 0)) != LINGBOT_DEPTH_REQUEST_SCHEMA:
        raise ValueError("Unsupported LingBot-Depth request schema")
    frames = request.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("LingBot-Depth request contains no frames")
    cancel_value = str(request.get("cancelPath", "")).strip()
    cancel_path = Path(cancel_value) if cancel_value else None
    if predictor is None:
        _write_json(
            progress_path,
            {
                "stage": "lingbot_depth_loading",
                "detail": "Loading pinned LingBot-Depth v0.5 checkpoint",
                "progress": 0.0,
            },
        )
        predictor = LingbotDepthPredictor.load()
    outputs: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        if cancel_path is not None and cancel_path.is_file():
            raise RuntimeError("LingBot-Depth refinement cancelled")
        _write_json(
            progress_path,
            {
                "stage": "lingbot_depth_inference",
                "detail": f"Refining aligned RGB-D keyframe {index + 1} of {len(frames)}",
                "progress": index / len(frames),
                "frameIndex": index,
                "frameCount": len(frames),
                "computeBackend": predictor.backend,
            },
        )
        color, depth_m, intrinsics = _load_frame(frame)
        prediction, mask = predictor.infer(color, depth_m, intrinsics)
        if prediction.shape != depth_m.shape or mask.shape != depth_m.shape:
            raise RuntimeError("LingBot-Depth returned a raster with incompatible alignment")
        prediction_path = Path(frame["predictionPath"])
        mask_path = Path(frame["modelMaskPath"])
        _save_array(prediction_path, prediction.astype(np.float32, copy=False))
        _save_array(mask_path, mask.astype(np.uint8, copy=False))
        outputs.append(
            {
                "key": str(frame["key"]),
                "predictionPath": str(prediction_path),
                "modelMaskPath": str(mask_path),
            }
        )
    result = {
        "schemaVersion": LINGBOT_DEPTH_REQUEST_SCHEMA,
        "status": "complete",
        "backend": predictor.backend,
        "codeRevision": LINGBOT_DEPTH_CODE_REVISION,
        "modelRevision": LINGBOT_DEPTH_MODEL_REVISION,
        "modelSha256": LINGBOT_DEPTH_MODEL_SHA256,
        "frames": outputs,
    }
    result_path = Path(request["resultPath"])
    _write_json(result_path, result)
    _write_json(
        progress_path,
        {
            "stage": "lingbot_depth_inference",
            "detail": f"Refined {len(frames)} aligned RGB-D keyframes",
            "progress": 1.0,
            "frameCount": len(frames),
            "computeBackend": predictor.backend,
        },
    )
    return result
