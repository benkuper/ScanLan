from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .media import (
    MediaPreparationOptions,
    _write_json_atomic,
    adaptive_frame_selection_status,
    prepare_media_dataset,
    prepare_media_observations,
)
from .runtime import pycolmap_feature_runtime
from .train import train_dataset


def _cuda_smoke_test() -> None:
    import torch
    from gsplat.rendering import rasterization_2dgs

    means = torch.tensor([[0.0, 0.0, 2.0]], device="cuda", requires_grad=True)
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
    scales = torch.full((1, 3), 0.05, device="cuda")
    opacities = torch.full((1,), 0.8, device="cuda")
    colors = torch.tensor([[0.3, 0.6, 0.9]], device="cuda")
    view = torch.eye(4, device="cuda").unsqueeze(0)
    intrinsics = torch.tensor(
        [[[8.0, 0.0, 3.5], [0.0, 8.0, 3.5], [0.0, 0.0, 1.0]]],
        device="cuda",
    )
    rendering, _, normals, _surface_normals, distortion, _median, _ = rasterization_2dgs(
        means=means,
        quats=quaternions,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=view,
        Ks=intrinsics,
        width=8,
        height=8,
        packed=True,
        render_mode="RGB+ED",
        distloss=True,
    )
    (rendering.sum() + normals.sum() + distortion.sum()).backward()
    torch.cuda.synchronize()
    if means.grad is None or not torch.isfinite(means.grad).all():
        raise RuntimeError("gsplat CUDA backward smoke test produced invalid gradients")


def _publish_failure(project_root: Path, error: Exception) -> None:
    """Leave a structured failure for the desktop process instead of only stderr."""
    progress_path = project_root / "outputs" / "splat-progress.json"
    try:
        payload = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_path.is_file()
            else {}
        )
        payload.update(
            {
                "stage": "failed",
                "detail": str(error),
                "error": str(error),
                "etaSeconds": None,
                "stageEtaSeconds": None,
            }
        )
        _write_json_atomic(progress_path, payload)
    except Exception:
        # Preserve the original worker failure even if status publication also
        # encounters an unexpected filesystem error.
        pass


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="scanlan-splat")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--project", type=Path, required=True)
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--iterations", type=int, default=30_000)
    train.add_argument("--resume", action="store_true")
    prepare = commands.add_parser("prepare-media")
    prepare.add_argument("--project", type=Path, required=True)
    prepare.add_argument("--source", type=Path, action="append", default=[])
    prepare.add_argument("--video-fps", type=float, default=15.0)
    prepare.add_argument("--maximum-video-frames", type=int, default=3_000)
    prepare.add_argument("--maximum-image-dimension", type=int, default=2560)
    prepare.add_argument("--geometry-worker", type=Path, default=None)
    prepare.add_argument("--progressive-rgb-preview", action="store_true")
    observations = commands.add_parser("extract-media")
    observations.add_argument("--project", type=Path, required=True)
    observations.add_argument("--source", type=Path, action="append", default=[])
    observations.add_argument("--video-fps", type=float, default=15.0)
    observations.add_argument("--maximum-video-frames", type=int, default=3_000)
    observations.add_argument("--maximum-image-dimension", type=int, default=2560)
    neural_sdf = commands.add_parser("refine-sdf")
    neural_sdf.add_argument("--input", type=Path, required=True)
    neural_sdf.add_argument("--output", type=Path, required=True)
    neural_sdf.add_argument("--report", type=Path, required=True)
    neural_sdf.add_argument("--progress", type=Path, default=None)
    neural_sdf.add_argument("--iterations", type=int, default=1_600)
    neural_sdf.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    material = commands.add_parser("prepare-material")
    material.add_argument("--dataset", type=Path, required=True)
    material.add_argument("--output", type=Path, required=True)
    material.add_argument("--frame-index", type=int, action="append", default=None)
    pack = commands.add_parser("material-pack")
    pack.add_argument("--pack", choices=["commercial", "research"], required=True)
    pack.add_argument("--output", type=Path, required=True)
    diagnostics = commands.add_parser("diagnostics")
    diagnostics.add_argument("--require-cuda", action="store_true")
    diagnostics.add_argument("--require-learned-features", action="store_true")
    diagnostics.add_argument("--require-adaptive-frames", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "diagnostics":
            import torch
            import gsplat
            from .neural_sdf import NEURAL_SDF_VERSION
            from scanlan_material import (
                ANALYSIS_VERSION,
                CONTRACT_VERSION,
                GEOMETRY_POLICY_VERSION,
                GEOMETRY_RESULT_VERSION,
                RADIOMETRY_VERSION,
                SURFACE_CONTRACT_VERSION,
            )

            cuda_available = torch.cuda.is_available()
            if arguments.require_cuda and cuda_available:
                _cuda_smoke_test()
            feature_runtime = pycolmap_feature_runtime()
            adaptive_frames = adaptive_frame_selection_status()
            print(
                json.dumps(
                    {
                        "version": __version__,
                        "cuda": cuda_available,
                        "cudaSmokeTest": cuda_available and arguments.require_cuda,
                        "device": torch.cuda.get_device_name(0) if cuda_available else None,
                        "cudaCapability": (
                            ".".join(map(str, torch.cuda.get_device_capability(0)))
                            if cuda_available
                            else None
                        ),
                        "torch": torch.__version__,
                        "gsplat": getattr(gsplat, "__version__", "unknown"),
                        "pycolmap": feature_runtime,
                        "learnedGeometryIsolation": "scanlan-geometry",
                        "adaptiveFrames": adaptive_frames,
                        "neuralSdf": {
                            "version": NEURAL_SDF_VERSION,
                            "cudaAvailable": cuda_available,
                        },
                        "materialFoundation": {
                            "contractVersion": CONTRACT_VERSION,
                            "radiometryVersion": RADIOMETRY_VERSION,
                            "analysisVersion": ANALYSIS_VERSION,
                            "surfaceContractVersion": SURFACE_CONTRACT_VERSION,
                            "geometryPolicyVersion": GEOMETRY_POLICY_VERSION,
                            "geometryResultVersion": GEOMETRY_RESULT_VERSION,
                        },
                    }
                )
            )
            return (
                2
                if arguments.require_cuda
                and (not cuda_available or not feature_runtime["cudaValidated"])
                or arguments.require_learned_features
                and not feature_runtime["learnedValidated"]
                or arguments.require_adaptive_frames
                and not adaptive_frames["enabled"]
                else 0
            )
        if arguments.command == "prepare-material":
            from scanlan_material import prepare_dataset_radiometry

            result = prepare_dataset_radiometry(
                arguments.dataset.resolve(strict=True),
                arguments.output.resolve(),
                frame_indices=arguments.frame_index,
            )
            print(json.dumps(result))
            return 0
        if arguments.command == "material-pack":
            from scanlan_material import write_pack_manifest

            result = write_pack_manifest(arguments.output.resolve(), arguments.pack)
            print(json.dumps(result))
            return 0
        if arguments.command in {"prepare-media", "extract-media"}:
            media_options = MediaPreparationOptions(
                video_fps=max(0.1, min(arguments.video_fps, 30.0)),
                maximum_video_frames=max(3, min(arguments.maximum_video_frames, 5_000)),
                maximum_image_dimension=max(720, min(arguments.maximum_image_dimension, 8_192)),
            )
            prepare = (
                prepare_media_observations
                if arguments.command == "extract-media"
                else prepare_media_dataset
            )
            result = (
                prepare(
                    arguments.project.resolve(),
                    arguments.source,
                    media_options,
                    geometry_worker=arguments.geometry_worker,
                    progressive_rgb_preview=arguments.progressive_rgb_preview,
                )
                if arguments.command == "prepare-media"
                else prepare(
                    arguments.project.resolve(),
                    arguments.source,
                    media_options,
                )
            )
            print(json.dumps(result))
            return 0
        if arguments.command == "refine-sdf":
            from .neural_sdf import run_refinement

            result = run_refinement(
                arguments.input.resolve(strict=True),
                arguments.output.resolve(),
                arguments.report.resolve(),
                arguments.progress.resolve() if arguments.progress is not None else None,
                max(200, min(arguments.iterations, 5_000)),
                arguments.device,
            )
            print(json.dumps(result))
            return 0 if result["status"] == "accepted" else 3
        result = train_dataset(
            arguments.dataset.resolve(),
            arguments.project.resolve(),
            max(1_000, min(arguments.iterations, 100_000)),
            arguments.resume,
        )
        print(json.dumps(result))
        return 0
    except Exception as error:
        project_root = getattr(arguments, "project", None)
        if isinstance(project_root, Path):
            _publish_failure(project_root.resolve(), error)
        print(f"scanlan-splat: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
